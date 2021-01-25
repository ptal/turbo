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

#include "solver.cuh"
#include "vstore.cuh"
#include "constraints.cuh"
#include "cuda_helper.hpp"
#include "status.cuh"
#include "search.cuh"

const int PROPS_TYPE = 3;

template <typename T>
struct SharedData {
  PropagatorsStatus* pstatus;
  VStore* vstore;
  T* props;

  CUDA SharedData(PropagatorsStatus* pstatus, VStore* vstore, T* props)
    : pstatus(pstatus), vstore(vstore), props(props) {}
};

template<typename T>
CUDA_GLOBAL void propagate_k(SharedData<T>* shared_data) {
  size_t id = threadIdx.x + blockIdx.x*blockDim.x;
  PropagatorsStatus& pstatus = *(shared_data->pstatus);
  T& p = shared_data->props[id];
  while (pstatus.is_exploring()) {
    Status s = p.propagate(*(shared_data->vstore)) ? UNKNOWN : IDLE;
    if(p.is_entailed(*(shared_data->vstore))) {
      s = ENTAILED;
    }
    if(p.is_disentailed(*(shared_data->vstore))) {
      s = DISENTAILED;
    }
    pstatus.inplace_join(p.uid, s);
  }
}

// The variables pstatus and vstore are shared among all propagators of all types.
// The UID inside a propagator, e.g., `TemporalProp::uid`, refers to the index of the propagator in the status array of `pstatus`.
template<typename T>
SharedData<T>* launch(PropagatorsStatus* pstatus, VStore* vstore, std::vector<T> &c, cudaStream_t s)
{
  // printf("launching %lu threads on stream %p\n", c.size(), s[0]);

  T* props;
  CUDIE(cudaMallocManaged(&props, c.size() * sizeof(T)));
  for (int i=0; i < c.size(); ++i) {
    props[i] = c[i];
  }

  SharedData<T> *shared_data;
  CUDIE(cudaMallocManaged(&shared_data, sizeof(SharedData<T>)));
  *shared_data = SharedData<T>(pstatus, vstore, props);

  propagate_k<T><<<1, c.size(), 0, s>>>(shared_data);
  CUDIE0();
  return shared_data;
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x)
{
  // std::cout << "Before propagation: " << std::endl;
  // vstore->print(var2name_raw);

  void* pstatus_raw;
  CUDIE(cudaMallocManaged(&pstatus_raw, sizeof(PropagatorsStatus)));
  PropagatorsStatus* pstatus = new(pstatus_raw) PropagatorsStatus(constraints.size());

  void* best_sol_raw;
  CUDIE(cudaMallocManaged(&best_sol_raw, sizeof(VStore)));
  VStore* best_sol = new(best_sol_raw) VStore(*vstore);

  constraints.print(*vstore);

  Var* temporal_vars = constraints.temporal_vars(vstore->size());

  cudaStream_t monitor;
  CUDIE(cudaStreamCreate(&monitor));
  cudaStream_t streams[PROPS_TYPE];
  for (int i=0; i < PROPS_TYPE; ++i) {
    CUDIE(cudaStreamCreate(&streams[i]));
  }

  search<<<1,1,0,monitor>>>(pstatus, vstore, best_sol, minimize_x, temporal_vars);
  CUDIE0();

  auto shared_data_0 = launch<TemporalProp>(pstatus, vstore, constraints.temporal, streams[0]);
  CUDIE0();
  auto shared_data_1 = launch<ReifiedLogicalAnd>(pstatus, vstore, constraints.reifiedLogicalAnd, streams[1]);
  CUDIE0();
  auto shared_data_2 = launch<LinearIneq>(pstatus, vstore, constraints.linearIneq, streams[2]);
  CUDIE0();

  CUDIE(cudaDeviceSynchronize());

  if(best_sol->size() == 0) {
    printf("Could not find a solution.\n");
  }
  else {
    printf("Best bound found is %d..%d.\n",
      (*best_sol)[minimize_x].lb, (*best_sol)[minimize_x].ub);
    best_sol->free();
  }
  CUDIE(cudaFree(best_sol_raw));
  CUDIE(cudaFree(temporal_vars));

  pstatus->free();
  CUDIE(cudaFree(pstatus_raw));

  CUDIE(cudaFree(shared_data_0->props));
  CUDIE(cudaFree(shared_data_0));
  CUDIE(cudaFree(shared_data_1->props));
  CUDIE(cudaFree(shared_data_1));
  CUDIE(cudaFree(shared_data_2->props));
  CUDIE(cudaFree(shared_data_2));

  CUDIE(cudaStreamDestroy(monitor));
  for (int i=0; i < PROPS_TYPE; ++i) {
    CUDIE(cudaStreamDestroy(streams[i]));
  }
}
