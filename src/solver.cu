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
    TreeData* td, Propagator** props, int nc) {
  int nid = threadIdx.x + blockIdx.x * blockDim.x;
  if (nid >= td->node_array.size()) { return; }

  bool has_changed = true;
  PropagatorsStatus& pstatus = *(td->node_array[nid].pstatus);
  VStore& vstore = *(td->node_array[nid].vstore);
  while(has_changed && pstatus.join() < ENTAILED) {
    has_changed = propagate(props, nc, vstore, pstatus);
  }
  // We propagate once more to verify that all propagators are really entailed.
  if(pstatus.join() == ENTAILED) {
    propagate(props, nc, vstore, pstatus);
  }
}

CUDA_GLOBAL void transfer_from_search(TreeData* td) {
  td->transferFromSearch();
}
CUDA_GLOBAL void transfer_to_search(TreeData* td) {
  //int i = threadIdx.x + blockIdx.x*blockDim.x;
  //td->transferToSearch_i(i);  // doesn't work, because write access to stack/node_array
  td->transferToSearch();
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x, int timeout)
{
  INFO(constraints.print(*vstore));


  Var* temporal_vars = constraints.temporal_vars(vstore->size());

  TreeData *tree_data;
  CUDIE(cudaMallocManaged(&tree_data, sizeof(*tree_data)));
  new(tree_data) TreeData(temporal_vars, minimize_x, *vstore, constraints.size());

  std::cout << "Start transfering propagator to device memory." << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  Propagator** props;
  CUDIE(cudaMallocManaged(&props, constraints.size() * sizeof(Propagator*)));
  for (auto p : constraints.propagators) {
    //std::cout << "Transferring " << p->uid << std::endl;
    props[p->uid] = p->to_device();
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Finish transfering propagators to device memory (" << duration << " ms)." << std::endl;

  int64_t durf (0);
  int64_t durk (0);
  int64_t durt (0);

  int loops (0);
  int device;
  CUDIE(cudaGetDevice(&device));
  cudaDeviceProp cudaProperties;
  CUDIE(cudaGetDeviceProperties(&cudaProperties, device));
  int nproc = cudaProperties.multiProcessorCount;
  std::cout<<"processors "<<nproc<<'\n';
  int threads, blocks;

  while (!tree_data->stack.is_empty()) {
    loops++;
    auto current = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(current - t1).count() > timeout) {
      break;
    }
    auto ts = std::chrono::high_resolution_clock::now();
    tree_data->transferFromSearch();
    //transfer_from_search<<<1,1>>>(tree_data);
    //CUDIE(cudaDeviceSynchronize());
    auto tf = std::chrono::high_resolution_clock::now();
    durf += std::chrono::duration_cast<std::chrono::milliseconds>( tf - ts ).count();

    //propagate_nodes_k<<<tree_data->node_array.size(), 1>>>(
    threads = 4;
    blocks = (1 + tree_data->node_array.size()/threads);
    ts = std::chrono::high_resolution_clock::now();
    propagate_nodes_k<<<blocks, threads>>>(
        tree_data, props, constraints.size());
    CUDIE(cudaDeviceSynchronize());
    tf = std::chrono::high_resolution_clock::now();
    durk += std::chrono::duration_cast<std::chrono::milliseconds>( tf - ts ).count();

    ts = std::chrono::high_resolution_clock::now();
    tree_data->transferToSearch();
    //transfer_to_search<<<1, 1>>>(tree_data);
    //CUDIE(cudaDeviceSynchronize());
    tf = std::chrono::high_resolution_clock::now();
    durt += std::chrono::duration_cast<std::chrono::milliseconds>( tf - ts ).count();
  }

  t2 = std::chrono::high_resolution_clock::now();
  CUDIE(cudaFree(props));
  duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  std::cout <<"[x"<<loops<<"] TPB="<<threads<<", fromsearch="<<durf<<", kernels="<<durk<<", tosearch="<<durt<<" (ms)\n";

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
