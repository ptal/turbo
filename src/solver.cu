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
#define AND_NODES 512
// #define SHMEM_SIZE 65536
#define SHMEM_SIZE 44000

CUDA_GLOBAL void search_k(
    Array<Pointer<TreeAndPar>>* trees,
    VStore* root,
    Array<Pointer<Propagator>>* props,
    Array<Var>* branching_vars,
    Pointer<Interval>* best_bound,
    Array<VStore>* best_sols,
    Var minimize_x,
    Array<Statistics>* stats)
{
  extern __shared__ int shmem[];
  const int n = SHMEM_SIZE;
  // __shared__ int shmem[n];
  int tid = threadIdx.x;
  int nodeid = blockIdx.x;
  int stride = blockDim.x;

  if (tid == 0) {
    SharedAllocator allocator(shmem, n);
    (*trees)[nodeid].reset(new(allocator) TreeAndPar(
      *root, *props, *branching_vars, **best_bound, minimize_x, allocator));
  }
  __syncthreads();
  (*trees)[nodeid]->search(tid, stride);
  if (tid == 0) {
    (*best_sols)[nodeid].reset((*trees)[nodeid]->best());
    (*stats)[nodeid] = (*trees)[nodeid]->statistics();
  }
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x, int timeout)
{
  // INFO(constraints.print(*vstore));

  Array<Var>* branching_vars = constraints.branching_vars();

  LOG(std::cout << "Start transfering propagator to device memory." << std::endl);
  auto t1 = std::chrono::high_resolution_clock::now();
  Array<Pointer<Propagator>>* props = new(managed_allocator) Array<Pointer<Propagator>>(constraints.size());
  LOG(std::cout << "props created " << props->size() << std::endl);
  for (auto p : constraints.propagators) {
    LOG(p->print(*vstore));
    LOG(std::cout << std::endl);
    (*props)[p->uid].reset(p->to_device());
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  LOG(std::cout << "Finish transfering propagators to device memory (" << duration << " ms)" << std::endl);

  t1 = std::chrono::high_resolution_clock::now();

  Array<Pointer<TreeAndPar>>* trees = new(managed_allocator) Array<Pointer<TreeAndPar>>(OR_NODES);
  Pointer<Interval>* best_bound = new(managed_allocator) Pointer<Interval>(Interval());
  Array<VStore>* best_sols = new(managed_allocator) Array<VStore>(*vstore, OR_NODES);
  Array<Statistics>* stats = new(managed_allocator) Array<Statistics>(OR_NODES);

  // cudaFuncSetAttribute(search_k, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SIZE);
  search_k<<<OR_NODES, min((int)props->size(), AND_NODES), SHMEM_SIZE>>>(trees, vstore, props, branching_vars,
    best_bound, best_sols, minimize_x, stats);
  CUDIE(cudaDeviceSynchronize());

  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

  // Gather statistics and best bound.
  Statistics statistics;
  for(int i = 0; i < stats->size(); ++i) {
    statistics.nodes += (*stats)[i].nodes;
    statistics.fails += (*stats)[i].fails;
    statistics.sols += (*stats)[i].sols;
    statistics.best_bound = statistics.best_bound == -1 ? (*stats)[i].best_bound : min(statistics.best_bound, (*stats)[i].best_bound);
    statistics.peak_depth = max(statistics.peak_depth, (*stats)[i].peak_depth);
  }

  statistics.print();
  // if(timeout != INT_MAX && duration > timeout * 1000) {
  std::cout << "solveTime=" << duration << std::endl;
  // if(statistics.nodes == NODES_LIMIT) {
  //   std::cout << "solveTime=timeout (" << duration/1000 << "." << duration % 1000 << "s)" << std::endl;
  // }
  // else {
  //   std::cout << "solveTime=" << duration/1000 << "." << duration % 1000 << "s" << std::endl;
  // }

  operator delete(best_bound, managed_allocator);
  operator delete(props, managed_allocator);
  operator delete(trees, managed_allocator);
  operator delete(branching_vars, managed_allocator);
  operator delete(best_bound, managed_allocator);
  operator delete(best_sols, managed_allocator);
}
