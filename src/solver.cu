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

template<typename T>
T* cons_alloc(std::vector<T> &c)
{
  T* props;
  CUDIE(cudaMallocManaged(&props, c.size() * sizeof(T)));
  for (int i=0; i < c.size(); ++i) {
    new(props + i) T(c[i]);
  }
  return props;
}

template <typename T>
CUDA_DEVICE
bool propagate(T* constraints, int nc, VStore& vstore, PropagatorsStatus& pstatus) {
  bool has_changed = false;
  for(int i=0; i<nc; ++i) {
    T& p = constraints[i];
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

CUDA_GLOBAL void propagate_nodes_k(
    TreeData* td,
    TemporalProp* tem_p, int nt,
    ReifiedLogicalAnd* rei_p, int nr,
    LinearIneq* lin_p, int nl) {
  int nid = threadIdx.x + blockIdx.x * blockDim.x;
  bool has_changed = true;
  PropagatorsStatus& pstatus = *(td->node_array[nid].pstatus);
  VStore& vstore = *(td->node_array[nid].vstore);
  while(has_changed && pstatus.join() < ENTAILED) {
    has_changed = propagate(tem_p, nt, vstore, pstatus);
    has_changed |= propagate(rei_p, nr, vstore, pstatus);
    has_changed |= propagate(lin_p, nl, vstore, pstatus);
  }
  // We propagate once more to verify that all propagators are really entailed.
  if(pstatus.join() == ENTAILED) {
    propagate(tem_p, nt, vstore, pstatus);
    propagate(rei_p, nr, vstore, pstatus);
    propagate(lin_p, nl, vstore, pstatus);
  }
  /*
  propagate_k<TemporalProp><<<constraints.temporal.size(), 1>>>(shared_data, tem_p);
  propagate_k<LinearIneq><<<constraints.linearIneq.size(), 1>>>(shared_data, lin_p);
  propagate_k<ReifiedLogicalAnd><<<constraints.reifiedLogicalAnd.size(), 1>>>(shared_data, rei_p);
  CUDIE(cudaDeviceSynchronize());
  */
}

CUDA_GLOBAL void transfer_search(TreeData* td) {
    td->transferFromSearch();
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x, int timeout)
{
  INFO(constraints.print(*vstore));

  Var* temporal_vars = constraints.temporal_vars(vstore->size());

  TreeData *tree_data;
  CUDIE(cudaMallocManaged(&tree_data, sizeof(*tree_data)));
  new(tree_data) TreeData(temporal_vars, minimize_x, *vstore, constraints.size());

  auto tem_p = cons_alloc<TemporalProp>(constraints.temporal);
  auto rei_p = cons_alloc<ReifiedLogicalAnd>(constraints.reifiedLogicalAnd);
  auto lin_p = cons_alloc<LinearIneq>(constraints.linearIneq);
  auto t1 = std::chrono::high_resolution_clock::now();

  while (!tree_data->stack.is_empty()) {
    auto current = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(current - t1).count() > timeout) {
      break;
    }
    tree_data->transferFromSearch();

    propagate_nodes_k<<<tree_data->node_array.size(), 1>>>(
        tree_data,
        tem_p, constraints.temporal.size(),
        rei_p, constraints.reifiedLogicalAnd.size(),
        lin_p ,constraints.linearIneq.size());
    CUDIE(cudaDeviceSynchronize());
    tree_data->transferToSearch();
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  CUDIE(cudaFree(tem_p));
  CUDIE(cudaFree(rei_p));
  CUDIE(cudaFree(lin_p));
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  tree_data->stats.print();
  if(duration > timeout * 1000) {
    std::cout << "solveTime=timeout" << std::endl;
  }
  else {
    std::cout << "solveTime=" << duration << std::endl;
  }

  CUDIE(cudaFree(tree_data));
  CUDIE(cudaFree(temporal_vars));
}
