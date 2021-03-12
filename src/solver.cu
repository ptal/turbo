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

#define OR_NODES 1

CUDA_GLOBAL void search_k(
    Pointer<TreeAndPar>* tree;
    VStore root,
    Vector<Pointer<Propagator>> props,
    Vector<Var> branching_vars,
    Pointer<Interval>* best_bound,
    VStore* best_sols,
    Var min_x)
{
  extern __shared__ int shmem[];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nodeid = blockIdx.x;
  int stride = gridDim.x * blockDim.x;

  if (tid == 0) {
    SharedAllocator allocator(shmem);
    tree.reset(new(allocator) TreeAndPar(
      root, props, branching_vars, *best_bound, min_x, allocator));
  }
  __syncthreads();
  tree->search(tid, stride);
  best_sols[blockIdx.x].reset(tree->best());
  __syncthreads();

}

void solve(VStore* vstore, Constraints constraints, Var minimize_x, int timeout)
{
  INFO(constraints.print(*vstore));

  Array<Var> temporal_vars = constraints.temporal_vars(vstore->size());

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

  t1 = std::chrono::high_resolution_clock::now();

  search_k<<<OR_NODES, 1>>>(tree_data, props, constraints.size());
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
