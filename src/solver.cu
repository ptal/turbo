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
    Array<Pointer<TreeAndPar>>* trees,
    VStore* root,
    Array<Pointer<Propagator>>* props,
    Array<Var>* branching_vars,
    Pointer<Interval>* best_bound,
    Array<VStore>* best_sols,
    Var minimize_x)
{
  extern __shared__ int shmem[];
  // __shared__ int shmem[1000];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nodeid = blockIdx.x;
  int stride = gridDim.x * blockDim.x;

  if (tid == 0) {
    SharedAllocator allocator(shmem);
    (*trees)[nodeid].reset(new(allocator) TreeAndPar(
      *root, *props, *branching_vars, **best_bound, minimize_x, allocator));
  }
  __syncthreads();
  (*trees)[nodeid]->search(tid, stride);
  if (tid == 0) {
    (*best_sols)[nodeid].reset((*trees)[nodeid]->best());
  }
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x, int timeout)
{
  // INFO(constraints.print(*vstore));

  Array<Var>* branching_vars = constraints.branching_vars();

  std::cout << "Start transfering propagator to device memory." << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  Array<Pointer<Propagator>>* props = new(managed_allocator) Array<Pointer<Propagator>>(constraints.size());
  std::cout << "props created " << props->size() << std::endl;
  for (auto p : constraints.propagators) {
    p->print(*vstore);
    std::cout << std::endl;
    (*props)[p->uid].reset(p->to_device());
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Finish transfering propagators to device memory (" << duration << " ms)" << std::endl;

  t1 = std::chrono::high_resolution_clock::now();

  Array<Pointer<TreeAndPar>>* trees = new(managed_allocator) Array<Pointer<TreeAndPar>>(OR_NODES);
  Array<VStore>* best_sols = new(managed_allocator) Array<VStore>(*vstore, OR_NODES);
  Pointer<Interval>* best_bound = new(managed_allocator) Pointer<Interval>(Interval());

  search_k<<<OR_NODES, 1, sizeof(int) * vstore->size() * 20>>>(trees, vstore, props, branching_vars,
    best_bound, best_sols, minimize_x);
  CUDIE(cudaDeviceSynchronize());

  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  //tree_data->stats.print();
  if(duration > timeout * 1000) {
    std::cout << "solveTime=timeout" << std::endl;
  }
  else {
    std::cout << "solveTime=" << duration << std::endl;
  }

  // delete(managed_allocator) best_bound;
  // delete(managed_allocator) props;
  // delete(managed_allocator) trees;
  // delete(managed_allocator) branching_vars;
  // delete(managed_allocator) best_bound;
  // delete(managed_allocator) best_sols;
}
