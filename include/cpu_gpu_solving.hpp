// Copyright 2024 Pierre Talbot

#ifndef TURBO_CPU_GPU_SOLVING_HPP
#define TURBO_CPU_GPU_SOLVING_HPP

#include "common_solving.hpp"
// #include "gpu_solving.hpp"

namespace bt = ::battery;

#ifdef __CUDACC__

template <class Universe>
using CP_CPUGPU = AbstractDomains<Universe,
  battery::statistics_allocator<battery::standard_allocator>,
  battery::statistics_allocator<UniqueLightAlloc<battery::managed_allocator, 0>>,
  battery::statistics_allocator<UniqueLightAlloc<battery::managed_allocator, 1>>>;

void print_memory_statistics(const char* key, size_t bytes) {
  printf("%% %s=%zu [", key, bytes);
  if(bytes < 1000 * 1000) {
    printf("%.2fKB", static_cast<double>(bytes) / 1000);
  }
  else if(bytes < 1000 * 1000 * 1000) {
    printf("%.2fMB", static_cast<double>(bytes) / (1000 * 1000));
  }
  else {
    printf("%.2fGB", static_cast<double>(bytes) / (1000 * 1000 * 1000));
  }
  printf("]\n");
}

using Itv0 = Interval<ZLB<int, bt::local_memory>>;
using Itv1 = Interval<ZLB<int, bt::atomic_memory_block>>;
using Itv2 = Interval<ZLB<int, bt::atomic_memory_grid>>;
using AtomicBool = B<bt::atomic_memory_block>;
using FPEngine = BlockAsynchronousIterationGPU<bt::pool_allocator>;

struct GPUState {
  using IStore = VStore<Itv1, bt::pool_allocator>;
  using IPC = PC<IStore, bt::pool_allocator>;
  abstract_ptr<IStore> store_ptr;
  abstract_ptr<IPC> ipc_ptr;

  CUDA GPUState() {}

  template <class StoreType, class PCType>
  CUDA void allocate(StoreType& store, PCType& pc) {
    size_t bytes = store.get_allocator().total_bytes_allocated()
      + pc.get_allocator().total_bytes_allocated()
      + sizeof(IStore) + sizeof(IPC) + 1000;
    void* mem_pool = bt::global_allocator{}.allocate(bytes);
    bt::pool_allocator pool(static_cast<unsigned char*>(mem_pool), bytes);
    AbstractDeps<bt::global_allocator, bt::pool_allocator> deps(bt::global_allocator{}, pool);
    ipc_ptr = bt::allocate_shared<IPC, bt::pool_allocator>(pool, pc, deps);
    store_ptr = deps.template extract<IStore>(store.aty());
  }

  CUDA void deallocate() {
    // NOTE: .reset() does not work because it does not reset the allocator, which is itself allocated in global memory.
    store_ptr = abstract_ptr<IStore>();
    ipc_ptr = abstract_ptr<IPC>();
  }
};

template <class StoreType, class PCType>
__global__ void allocate_gpu_state(GPUState* state, StoreType* store, PCType* pc) {
  state->allocate(*store, *pc);
}

__global__ void deallocate_gpu_state(GPUState* state) {
  state->deallocate();
}

using FPEngineBlock = BlockAsynchronousIterationGPU<bt::pool_allocator>;
using FPEngineGrid = GridAsynchronousIterationGPU<bt::global_allocator>;

template <class StoreType>
__global__ void fixpoint_kernel_block(GPUState* state, StoreType* store, size_t shared_bytes) {
  extern __shared__ unsigned char shared_mem[];
  assert(blockIdx.x == 0);
  auto group = cooperative_groups::this_thread_block();
  store->copy_to(group, *state->store_ptr);
  // No need to sync here, make_unique_block already contains sync.
  bt::unique_ptr<bt::pool_allocator, bt::global_allocator> shared_mem_pool_ptr;
  bt::pool_allocator& shared_mem_pool = bt::make_unique_block(shared_mem_pool_ptr, shared_mem, shared_bytes);
  bt::unique_ptr<FPEngineBlock, bt::global_allocator> fp_engine_ptr;
  FPEngineBlock& fp_engine = bt::make_unique_block(fp_engine_ptr, group, shared_mem_pool);
  fp_engine.fixpoint(*(state->ipc_ptr));
  group.sync();
  state->store_ptr->copy_to(group, *store);
  group.sync();
}

template <class StoreType>
__global__ void fixpoint_kernel_grid(GPUState* state, StoreType* store) {
  auto group = cooperative_groups::this_grid();
  store->copy_to(group, *state->store_ptr);
  bt::unique_ptr<FPEngineGrid, bt::global_allocator> fp_engine_ptr;
  FPEngineGrid& fp_engine = bt::make_unique_grid(fp_engine_ptr, group);
  fp_engine.fixpoint(*(state->ipc_ptr));
  group.sync();
  state->store_ptr->copy_to(group, *store);
  group.sync();
}

#endif

void cpu_gpu_solve(const Configuration<battery::standard_allocator>& config) {
#ifndef __CUDACC__
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
#else
  auto start = std::chrono::high_resolution_clock::now();
  CP_CPUGPU<Itv0> cp(config);
  cp.preprocess();

  /** Some GPU configuration. */
  size_t shared_mem_size = 100;
  int hint_num_blocks;
  int hint_num_threads;
  CUDAE(cudaOccupancyMaxPotentialBlockSize(&hint_num_blocks, &hint_num_threads, (void*) fixpoint_kernel_grid<typename CP_CPUGPU<Itv0>::IStore>, shared_mem_size));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  size_t total_global_mem = deviceProp.totalGlobalMem;
  size_t num_sm = deviceProp.multiProcessorCount;
  cp.config.and_nodes = (cp.config.and_nodes == 0) ? hint_num_threads : cp.config.and_nodes;
  cp.config.or_nodes = (cp.config.or_nodes == 0) ? hint_num_blocks : cp.config.or_nodes;
  // The stack allocated depends on the maximum number of threads per SM, not on the actual number of threads per block.
  size_t total_stack_size = num_sm * deviceProp.maxThreadsPerMultiProcessor * cp.config.stack_kb * 1000;
  size_t remaining_global_mem = total_global_mem - total_stack_size;
  remaining_global_mem -= remaining_global_mem / 10; // We leave 10% of global memory free for CUDA allocations, not sure if it is useful though.
  CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, cp.config.stack_kb*1000));
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, remaining_global_mem));
  if(cp.config.verbose_solving) {
    print_memory_statistics("stack_memory", total_stack_size);
    print_memory_statistics("heap_memory", remaining_global_mem);
    printf("%% and_nodes=%zu\n", cp.config.and_nodes);
    printf("%% or_nodes=%zu\n", cp.config.or_nodes);
  }

  // int num_blocks;
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, fixpoint_kernel_block<typename CP_CPUGPU<Itv0>::IStore>, 256, shared_mem_size);
  // printf("%% max_blocks_per_sm=%d\n", num_blocks);

  /** Allocating the GPU data structures in global memory. */
  using managed_alloc1 = battery::statistics_allocator<UniqueLightAlloc<battery::managed_allocator, 0>>;
  using managed_alloc2 = battery::statistics_allocator<UniqueLightAlloc<battery::managed_allocator, 1>>;
  using IStoreCopy = VStore<Itv0, managed_alloc1>;
  using IPCCopy = PC<IStoreCopy, managed_alloc2>;
  AbstractDeps<bt::standard_allocator, managed_alloc1, managed_alloc2> deps(
    bt::standard_allocator{}, managed_alloc1{}, managed_alloc2{});
  auto ipc_ptr = bt::make_shared<IPCCopy, managed_alloc2>(*cp.ipc.get(), deps);
  auto store_ptr = deps.template extract<IStoreCopy>(cp.store->aty());
  auto gpu_state = bt::make_unique<GPUState, bt::managed_allocator>();
  allocate_gpu_state<<<1, 1>>>(gpu_state.get(), store_ptr.get(), ipc_ptr.get());
  CUDAEX(cudaDeviceSynchronize());

  /** Solving algorithm with propagation on the GPU and search on the CPU. */
  local::B has_changed = true;
  block_signal_ctrlc();
  while(!must_quit() && check_timeout(cp, start) && has_changed) {
    has_changed = false;
    if(cp.config.or_nodes == 1) {
      fixpoint_kernel_block<<<1,  static_cast<unsigned int>(cp.config.and_nodes), shared_mem_size>>>(gpu_state.get(), cp.store.get(), shared_mem_size);
    }
    else {
      auto store_ptr = cp.store.get();
      auto gpu_state_ptr = gpu_state.get();
      void* args[] = {&gpu_state_ptr, &store_ptr};
      dim3 dimBlock(cp.config.and_nodes, 1, 1);
      dim3 dimGrid(cp.config.or_nodes, 1, 1);
      CUDAEX(cudaLaunchCooperativeKernel((void*)fixpoint_kernel_grid<typename CP_CPUGPU<Itv0>::IStore>, dimGrid, dimBlock, args));
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
  deallocate_gpu_state<<<1, 1>>>(gpu_state.get());
  CUDAEX(cudaDeviceSynchronize());
#endif
}

#endif
