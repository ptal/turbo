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
#include "propagators.cuh"
#include "cuda_helper.hpp"
#include "statistics.cuh"
#include "status.cuh"
#include "search.cuh"

CUDA_DEVICE
bool propagate(Propagator** props, int nc, VStore& vstore, PropagatorsStatus& pstatus) {
  bool has_changed = false;
  for(int i=nc-1; i>= 0; --i) {
    Propagator* p = props[i];
    bool has_changed2 = p->propagate(vstore);
    has_changed |= has_changed2;
    Status s = has_changed2 ? UNKNOWN : IDLE;
    if(p->is_entailed(vstore)) {
      s = ENTAILED;
    }
    if(p->is_disentailed(vstore)) {
      s = DISENTAILED;
    }
    pstatus.inplace_join(p->uid, s);
  }
  return has_changed;
}

CUDA_GLOBAL void propagate_nodes_k(
    TreeData* td, 
    Propagator** props, 
    int nc)
{
  int nid = threadIdx.x + blockIdx.x * blockDim.x;
  if (nid >= nc) { return; }

  bool has_changed = true;
  PropagatorsStatus& pstatus = *(td->node_array->pstatus);
  VStore& vstore = *(td->node_array->vstore);
  while(has_changed && pstatus.join() < ENTAILED) {
    has_changed = propagate(props, nc, vstore, pstatus);
  }
  // We propagate once more to verify that all propagators are really entailed.
  if(pstatus.join() == ENTAILED) {
    propagate(props, nc, vstore, pstatus);
  }
}

CUDA_GLOBAL void explore(
    TreeData *tree_data, 
    Propagator **props, 
    int cons_sz)
{
  int tid = blockIdx.x + threadIdx.x*blockDim.x;
  while (!tree_data->stack.is_empty()) {
    tree_data->transferFromSearch();
    printf("nodes %d\n", tree_data->node_array.size());
    propagate_nodes_k<<<1, cons_sz>>>(tree_data, props, cons_sz);
    CUDIE(cudaDeviceSynchronize());
    tree_data->transferToSearch();
  }
}

CUDA_GLOBAL void new_tree(
    TreeData *tree_data, 
    Var* temporal_vars, 
    Var minimize_x, 
    VStore* vstore,
    size_t csize)
{
  new(tree_data) TreeData(temporal_vars, minimize_x, *vstore, csize);
}

CUDA_GLOBAL void tree_stats(TreeData *tree_data)
{
  tree_data->stats.print();
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x, int timeout)
{
  INFO(constraints.print(*vstore));

  Var* temporal_vars = constraints.temporal_vars(vstore->size());

  std::cout << "Start transfering propagator to device memory." << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  Propagator** props;
  CUDIE(cudaMallocManaged(&props, constraints.size() * sizeof(Propagator*)));
  for (auto p : constraints.propagators) {
    // std::cout << "Transferring " << p->uid << std::endl;
    props[p->uid] = p->to_device();
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Finish transfering propagators to device memory (" << duration << " ms)" << std::endl;

  // process one tree (subtree later), within a CUDA thread
  TreeData *tree_data;
  CUDIE(cudaMalloc(&tree_data, sizeof(*tree_data)));
  new_tree<<<1, 1>>>(tree_data, temporal_vars, minimize_x, vstore, constraints.size());
  CUDIE(cudaDeviceSynchronize());
  //new(tree_data) TreeData(temporal_vars, minimize_x, *vstore, constraints.size());

  t1 = std::chrono::high_resolution_clock::now();

  explore<<<1,1>>>(tree_data, props, constraints.size());
  CUDIE(cudaDeviceSynchronize());

  t2 = std::chrono::high_resolution_clock::now();
  CUDIE(cudaFree(props));
  duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  tree_stats<<<1,1>>>(tree_data);
  CUDIE(cudaDeviceSynchronize());
  //tree_data->stats.print();

  if(duration > timeout * 1000) {
    std::cout << "solveTime=timeout" << std::endl;
  }
  else {
    std::cout << "solveTime=" << duration << std::endl;
  }

  CUDIE(cudaFree(tree_data));
  CUDIE(cudaFree(temporal_vars));
}
