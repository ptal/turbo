// Copyright 2021 Pierre Talbot, Frédéric Pinel

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <new>
#include <chrono>

#include "solver.cuh"
#include "vstore.cuh"
#include "constraints.cuh"
#include "cuda_helper.hpp"
#include "statistics.cuh"
#include "status.cuh"
#include "search.cuh"

#ifdef SEQUENTIAL

template <typename T>
bool propagate(std::vector<T>& constraints, VStore& vstore, PropagatorsStatus& pstatus) {
  bool has_changed = false;
  for(auto p : constraints) {
    bool has_changed2 = p.propagate(vstore);
    has_changed |= has_changed2;
    Status s = has_changed2 ? UNKNOWN : IDLE;
    if(p.is_entailed(vstore)) {
      s = ENTAILED;
    }
    if(p.is_disentailed(vstore)) {
      s = DISENTAILED;
    }
    pstatus.inplace_join(p.uid, s);
  }
  return has_changed;
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x)
{
  INFO(constraints.print(*vstore));
  Statistics stats;
  VStore best_sol = VStore(vstore->size());
  Var* temporal_vars = constraints.temporal_vars(vstore->size());
  SharedData shared_data = SharedData(vstore, constraints.size());

  auto t1 = std::chrono::high_resolution_clock::now();

  shared_data.into_device_mem();
  Stack stack(*(shared_data.vstore));
  Interval best_bound = {limit_min(), limit_max()};
  INFO(printf("starting search with %p\n", shared_data.vstore));

  while(shared_data.exploring) {
    // I. Propagation
    VStore& vstore = *(shared_data.vstore);
    PropagatorsStatus& pstatus = *(shared_data.pstatus);
    bool has_changed = true;
    while(has_changed && pstatus.join() < ENTAILED) {
      has_changed = propagate(constraints.temporal, vstore, pstatus);
      has_changed |= propagate(constraints.reifiedLogicalAnd, vstore, pstatus);
      has_changed |= propagate(constraints.linearIneq, vstore, pstatus);
    }
    // We propagate once more to verify that all propagators are really entailed.
    if(pstatus.join() == ENTAILED) {
      propagate(constraints.temporal, vstore, pstatus);
      propagate(constraints.reifiedLogicalAnd, vstore, pstatus);
      propagate(constraints.linearIneq, vstore, pstatus);
    }
    // II. Branching
    one_step(stack, best_bound, shared_data.pstatus->join(),
      &shared_data, &stats, &best_sol, minimize_x, temporal_vars);
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  stats.print();
  std::cout << "solveTime=" << duration << std::endl;
}

#else

const int PROPS_TYPE = 3;

template<typename T>
CUDA_GLOBAL void propagate_k(SharedData* shared_data, T* props) {
  size_t id = threadIdx.x + blockIdx.x*blockDim.x;
  T& p = props[id];
  while (shared_data->exploring) {
    // The order of reading pstatus and vstore is important w.r.t. backtracking in the search thread.
    // It is ok to write in the old pstatus array accordingly to a new store, but not to write in a new pstatus array according to the old store.
    // NOTE: Actually might not be good enough, if the write operations executed sequentially in the search thread can be observed in a different order here.
    PropagatorsStatus& pstatus = *(shared_data->pstatus);
    VStore& vstore = *(shared_data->vstore);
    Status s = p.propagate(vstore) ? UNKNOWN : IDLE;
    if(p.is_entailed(vstore)) {
      s = ENTAILED;
    }
    if(p.is_disentailed(vstore)) {
      // INFO(printf("%lu disentailed in (%p,%p).\n", p.uid, &vstore, &pstatus));
      s = DISENTAILED;
    }
    pstatus.inplace_join(p.uid, s);
  }
}

// The variables pstatus and vstore are shared among all propagators of all types.
// The UID inside a propagator, e.g., `TemporalProp::uid`, refers to the index of the propagator in the status array of `pstatus`.
template<typename T>
T* launch(SharedData* shared_data, std::vector<T> &c, cudaStream_t s)
{
  // printf("launching %lu threads on stream %p\n", c.size(), s[0]);

  T* props;
  CUDIE(cudaMallocManaged(&props, c.size() * sizeof(T)));
  for (int i=0; i < c.size(); ++i) {
    props[i] = c[i];
  }

  propagate_k<T><<<1, c.size(), 0, s>>>(shared_data, props);
  CUDIE0();
  return props;
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x)
{
  INFO(constraints.print(*vstore));

  void* stats_raw;
  CUDIE(cudaMallocManaged(&stats_raw, sizeof(Statistics)));
  Statistics* stats = new(stats_raw) Statistics();


  void* best_sol_raw;
  CUDIE(cudaMallocManaged(&best_sol_raw, sizeof(VStore)));
  VStore* best_sol = new(best_sol_raw) VStore(vstore->size());

  Var* temporal_vars = constraints.temporal_vars(vstore->size());

  cudaStream_t monitor;
  CUDIE(cudaStreamCreate(&monitor));
  cudaStream_t streams[PROPS_TYPE];
  for (int i=0; i < PROPS_TYPE; ++i) {
    CUDIE(cudaStreamCreate(&streams[i]));
  }

  void *raw_shared_data;
  CUDIE(cudaMallocManaged(&raw_shared_data, sizeof(SharedData)));
  SharedData* shared_data = new(raw_shared_data) SharedData(vstore, constraints.size());

  auto t1 = std::chrono::high_resolution_clock::now();

  search<<<1,1,0,monitor>>>(shared_data, stats, best_sol, minimize_x, temporal_vars);
  CUDIE0();

  auto props1 = launch<TemporalProp>(shared_data, constraints.temporal, streams[0]);
  CUDIE0();
  auto props2 = launch<ReifiedLogicalAnd>(shared_data, constraints.reifiedLogicalAnd, streams[1]);
  CUDIE0();
  auto props3 = launch<LinearIneq>(shared_data, constraints.linearIneq, streams[2]);
  CUDIE0();

  CUDIE(cudaDeviceSynchronize());

  auto t2 = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  stats->print();
  std::cout << "solveTime=" << duration << std::endl;

  best_sol->~VStore();
  CUDIE(cudaFree(best_sol_raw));

  CUDIE(cudaFree(temporal_vars));

  shared_data->~SharedData();
  CUDIE(cudaFree(raw_shared_data));

  CUDIE(cudaFree(props1));
  CUDIE(cudaFree(props2));
  CUDIE(cudaFree(props3));

  CUDIE(cudaStreamDestroy(monitor));
  for (int i=0; i < PROPS_TYPE; ++i) {
    CUDIE(cudaStreamDestroy(streams[i]));
  }
}

#endif
