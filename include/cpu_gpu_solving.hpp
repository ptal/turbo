// Copyright 2024 Pierre Talbot

#ifndef TURBO_CPU_GPU_SOLVING_HPP
#define TURBO_CPU_GPU_SOLVING_HPP

#include "common_solving.hpp"

namespace bt = ::battery;

#ifdef __CUDACC__

template <class Universe>
using CP_CPUGPU = AbstractDomains<Universe,
  battery::statistics_allocator<battery::standard_allocator>,
  battery::statistics_allocator<UniqueLightAlloc<battery::managed_allocator, 0>>,
  battery::statistics_allocator<UniqueLightAlloc<battery::managed_allocator, 1>>>;

using FPEngineBlock = BlockAsynchronousIterationGPU<bt::pool_allocator>;
using FPEngineGrid = GridAsynchronousIterationGPU<bt::global_allocator>;

template <class AD>
__global__ void fixpoint_kernel_block(AD* ad, size_t shared_bytes) {
  assert(blockIdx.x == 0);
  extern __shared__ unsigned char shared_mem[];
  auto group = cooperative_groups::this_thread_block();
  bt::unique_ptr<bt::pool_allocator, bt::global_allocator> shared_mem_pool_ptr;
  bt::pool_allocator& shared_mem_pool = bt::make_unique_block(shared_mem_pool_ptr, shared_mem, shared_bytes);
  bt::unique_ptr<FPEngineBlock, bt::global_allocator> fp_engine_ptr;
  FPEngineBlock& fp_engine = bt::make_unique_block(fp_engine_ptr, group, shared_mem_pool);
  fp_engine.fixpoint(*ad);
  group.sync();
}

template <class AD>
__global__ void fixpoint_kernel_grid(AD* ad) {
  auto group = cooperative_groups::this_grid();
  bt::unique_ptr<FPEngineGrid, bt::global_allocator> fp_engine_ptr;
  FPEngineGrid& fp_engine = bt::make_unique_grid(fp_engine_ptr, group);
  fp_engine.fixpoint(*ad);
  group.sync();
}

#endif

void cpu_gpu_solve(const Configuration<battery::standard_allocator>& config) {
#ifndef __CUDACC__
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
#else
  auto start = std::chrono::high_resolution_clock::now();
  CP_CPUGPU<Itv> cp(config);
  cp.preprocess();

  /** Some GPU configuration. */
  size_t shared_mem_size = 100;
  int hint_num_blocks;
  int hint_num_threads;
  CUDAE(cudaOccupancyMaxPotentialBlockSize(&hint_num_blocks, &hint_num_threads, (void*) fixpoint_kernel_grid<typename CP_CPUGPU<Itv>::IPC>, shared_mem_size));
  cp.config.and_nodes = (cp.config.and_nodes == 0) ? hint_num_threads : cp.config.and_nodes;
  cp.config.or_nodes = (cp.config.or_nodes == 0) ? hint_num_blocks : cp.config.or_nodes;
  CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, cp.config.stack_kb*1000));

  local::B has_changed = true;
  block_signal_ctrlc();
  while(!must_quit() && check_timeout(cp, start) && has_changed) {
    has_changed = false;
    if(cp.config.or_nodes == 1) {
      fixpoint_kernel_block<<<1,  static_cast<unsigned int>(cp.config.and_nodes), shared_mem_size>>>(cp.ipc.get(), shared_mem_size);
    }
    else {
      auto ipc_ptr = cp.ipc.get();
      void* args[] = {&ipc_ptr};
      dim3 dimBlock(cp.config.and_nodes, 1, 1);
      dim3 dimGrid(cp.config.or_nodes, 1, 1);
      CUDAEX(cudaLaunchCooperativeKernel((void*)fixpoint_kernel_grid<typename CP_CPUGPU<Itv>::IPC>, dimGrid, dimBlock, args));
    }
    CUDAEX(cudaDeviceSynchronize());
    bool must_prune = cp.on_node();
    if(cp.ipc->is_bot()) {
      cp.on_failed_node();
    }
    else if(cp.search_tree->template is_extractable<AtomicExtraction>()) {
      has_changed |= cp.bab->deduce();
      must_prune |= cp.on_solution_node();
    }
    has_changed |= cp.search_tree->deduce();
    if(must_prune) { break; }
  }
  cp.print_final_solution();
  cp.print_mzn_statistics();
#endif
}


#endif