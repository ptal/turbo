// Copyright 2025 Pierre Talbot

#ifndef TURBO_BAREBONES_DIVE_AND_SOLVE_HPP
#define TURBO_BAREBONES_DIVE_AND_SOLVE_HPP

#include "common_solving.hpp"
#include "memory_gpu.hpp"
#include "lala/light_branch.hpp"
#include <mutex>
#include <thread>
#include <chrono>

/** This is required in order to guess the usage of global memory, and increase it. */
#define MAX_SEARCH_DEPTH 100000

namespace bt = ::battery;

/**
 * The full GPU version (`gpu_dive_and_solve`) is not compiling on modern GPU hardware (SM >= 9) due to the kernel being too large.
 * We circuvanted this issue by creating an hybrid version where only propagation is executed on the GPU (`hybrid_dive_and_solve`).
 * This has the disadvantage of memory transfer and synchronization overheads.
 * We propose a new "barebones" version which contains less abstractions than the GPU and hybrid versions, but have the same functionalities.
 * In particular, we directly implement the branch-and-bound algorithm here and avoid using `lala::BAB` and `lala::SearchTree` which are nice from a software engineering perspective but bring significant overhead.
 * This version is intended to reach the best possible performance.
 */

#ifdef __CUDACC__

#include <cuda/std/chrono>
#include <cuda/semaphore>

#endif

namespace barebones {

#ifdef __CUDACC__

/** `ConcurrentAllocator` allocates memory available both on CPU and GPU. For non-Linux systems such as Windows pinned memory must be used (see PR #19). */
#ifdef NO_CONCURRENT_MANAGED_MEMORY
  using ConcurrentAllocator = bt::pinned_allocator;
#else
  using ConcurrentAllocator = bt::managed_allocator;
#endif

using GridCP = AbstractDomains<Itv,
  bt::statistics_allocator<ConcurrentAllocator>,
  bt::statistics_allocator<UniqueLightAlloc<ConcurrentAllocator, 0>>,
  bt::statistics_allocator<UniqueLightAlloc<ConcurrentAllocator, 1>>>;

/** Data shared between CPU and GPU. */
struct UnifiedData {
  /** The root node of the problem, useful to backtrack when solving a new subproblem.
   * Also contains the shared information such as statistics and solver configuration.
   */
  GridCP root;

  /** Stop signal from the CPU because of a timeout or CTRL-C. */
  cuda::std::atomic_flag stop;

  /** The memory configuration of each block. */
  MemoryConfig mem_config;

  UnifiedData(const CP<Itv>& cp, MemoryConfig mem_config)
   : root(cp)
   , stop(false)
   , mem_config(mem_config)
  {
    size_t num_subproblems = 1;
    num_subproblems <<= root.config.subproblems_power;
    root.stats.eps_num_subproblems = num_subproblems;
  }
};

using IStore = VStore<Itv, bt::pool_allocator>;
using IProp = PIR<IStore, bt::pool_allocator>;

/** Data private to a single block. */
struct BlockData {
  /** The store of variables at the root of the current subproblem. */
  abstract_ptr<IStore> store_root;

  /** The best solution found so far in this block. */
  abstract_ptr<IStore> best_store;

  /** The current store of variables.
   * We use a `pool_allocator`, this allows to easily switch between global memory and shared memory, if the store of variables can fit inside.
   * */
  abstract_ptr<IStore> store;

  /** The propagators implemented as an array of bytecodes.
   * Similarly, the propagators can be stored in the global or shared memory.
   * If the propagators do not fit in shared memory, the array of propagators is shared among all blocks.
   * It is possible because the propagators are state-less, we avoid duplicating them in each block.
   * */
  abstract_ptr<IProp> iprop;

  /** The statistics of the current block. */
  Statistics<bt::global_allocator> stats;

  /** The path from `UnifiedData::root` to the current subproblem `store_root`. */
  size_t subproblem_idx;

  /** The best bound found so far by this block. */
  Itv best_bound;

  /** The current strategy being used to split the store.
   * It is an index into `GridData::strategies`.
   */
  int current_strategy;

  /** The next unassigned variable in the current strategy.
   * It is an index into `GridData::strategies.vars`.
   */
  int next_unassigned_var;

  /** The decision taken when exploring the tree. */
  bt::vector<LightBranch<Itv>, bt::global_allocator> decisions;

  /** This is used for fast backtracking: `ropes[decisions.size()-1]` is the depth we need to backtrack to. */
  bt::vector<int, bt::global_allocator> ropes;

  __device__ BlockData() {
    decisions.reserve(5000);
    ropes.reserve(5000);
  }
};

/** Data shared among all blocks. */
struct GridData {
  /** The private data of each block. */
  bt::vector<BlockData, bt::global_allocator> blocks;

  /** We generate the subproblems lazily.
   * Suppose we generate `2^3` subproblems, we represent the first subproblem as `000`, the second as `001`, the third as `010`, and so on.
   * A `0` means to turn left in the search tree, and a `1` means to turn right.
   * Incrementing this integer will generate the path of the next subproblem.
   */
  ZLB<size_t, bt::atomic_memory_grid> next_subproblem;

  /** This is an approximation of the best bound found so far, globally, across all threads.
   * It is not necessarily the true best bound at each time `t`.
   * The true best bound is obtained by `meet` over all block's best bounds.
   * It is used to share information among blocks.
   */
  Itv appx_best_bound;

  /** Due to multithreading, we must protect `stdout` when printing.
   * The model of computation in this work is lock-free, but it seems unavoidable for printing.
  */
  cuda::binary_semaphore<cuda::thread_scope_device> print_lock;

  /** The search strategy is immutable and shared among the blocks. */
  bt::vector<StrategyType<bt::global_allocator>, bt::global_allocator> dive_strategies;

  /** The search strategy is immutable and shared among the blocks. */
  bt::vector<StrategyType<bt::global_allocator>, bt::global_allocator> solve_strategies;

  /** The objective variable to minimize.
   * Maximization problem are transformed into minimization problems by negating the objective variable.
   * Equal to -1 if the problem is a satisfaction problem.
   */
  AVar obj_var;

  __device__ GridData(const GridCP& root)
   : blocks(root.config.or_nodes)
   , next_subproblem(root.config.or_nodes)
   , appx_best_bound(Itv::top())
   , print_lock(1)
   , dive_strategies(root.eps_split->strategies_())
   , solve_strategies(root.split->strategies_())
   , obj_var(root.bab->objective_var())
  {
    for(int i = 0; i < blocks.size(); ++i) {
      blocks[i].subproblem_idx = i;
    }
  }
};

MemoryConfig configure_gpu_barebones(CP<Itv>&);
__global__ void initialize_global_data(UnifiedData*, bt::unique_ptr<GridData, bt::global_allocator>*);
__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>*);
__global__ void gpu_barebones_solve(UnifiedData*, GridData*);

void barebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {
  auto start = std::chrono::steady_clock::now();
  check_support_managed_memory();
  check_support_concurrent_managed_memory();
  /** We start with some preprocessing to reduce the number of variables and constraints. */
  CP<Itv> cp(config);
  cp.preprocess();
  if(cp.iprop->is_bot()) {
    cp.print_final_solution();
    cp.print_mzn_statistics();
    return;
  }
  MemoryConfig mem_config = configure_gpu_barebones(cp);
  auto unified_data = bt::make_unique<UnifiedData, ConcurrentAllocator>(cp, mem_config);
  auto grid_data = bt::make_unique<bt::unique_ptr<GridData, bt::global_allocator>, ConcurrentAllocator>();
  initialize_global_data<<<1,1>>>(unified_data.get(), grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
  gpu_barebones_solve
    <<<static_cast<unsigned int>(cp.config.or_nodes),
      CUDA_THREADS_PER_BLOCK,
      mem_config.shared_bytes>>>
    (unified_data.get(), grid_data->get());
  bool interrupted = wait_solving_ends(unified_data->stop, unified_data->root, start);
  CUDAEX(cudaDeviceSynchronize());
  deallocate_global_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
}

/** We configure the GPU according to the user configuration:
 * 1) Configure the size of the shared memory.
 * 2) Guess the "best" number of blocks per SM, if not provided.
 * 3) Increase the global heap memory.
 * 4) Increase the stack size if requested by the user.
 */
MemoryConfig configure_gpu_barebones(CP<Itv>& cp) {
  auto& config = cp.config;

  /** I. Configure the shared memory size. */
  size_t store_bytes = gpu_sizeof<IStore>() + gpu_sizeof<abstract_ptr<IStore>>() + cp.store->vars() * gpu_sizeof<Itv>();
  size_t iprop_bytes = gpu_sizeof<IProp>() + gpu_sizeof<abstract_ptr<IProp>>() + cp.iprop->num_deductions() * gpu_sizeof<bytecode_type>();
  MemoryConfig mem_config((void*) gpu_barebones_solve, store_bytes, iprop_bytes);
  mem_config.print_mzn_statistics(config, cp.stats);

  /** II. Number of blocks per SM. */
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if(config.or_nodes == 0) {
    config.or_nodes = deviceProp.multiProcessorCount;
  }
  if(cp.config.verbose_solving) {
    int num_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, (void*) gpu_barebones_solve, CUDA_THREADS_PER_BLOCK, mem_config.shared_bytes);
    printf("%% max_blocks_per_sm=%d\n", num_blocks);
  }

  /** III. Size of the heap global memory.
   * The estimation is very conservative, normally we should not run out of memory.
   * */
  size_t required_global_mem =
    /** Memory shared among all blocks. Use store_bytes to estimate the strategies. */
    gpu_sizeof<UnifiedData>() + store_bytes * 5 + iprop_bytes +
    gpu_sizeof<GridData>() +
    config.or_nodes * gpu_sizeof<BlockData>() +
    config.or_nodes * store_bytes * 3 + // current, root, best.
    config.or_nodes * iprop_bytes +
    config.or_nodes * (gpu_sizeof<int>() + gpu_sizeof<LightBranch<Itv>>()) * MAX_SEARCH_DEPTH;
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, required_global_mem));
  cp.stats.print_memory_statistics(cp.config.verbose_solving, "heap_memory", required_global_mem);
  if(cp.config.verbose_solving) {
    printf("%% or_nodes=%zu\n", config.or_nodes);
  }
  if(deviceProp.totalGlobalMem < required_global_mem) {
    printf("%% WARNING: The total global memory available is less than the required global memory.\n\
    As our memory estimation is very conservative, it might still work, but it is not guaranteed.\n");
  }

  /** IV. Increase the stack if requested by the user. */
  if(config.stack_kb != 0) {
    CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, config.stack_kb*1024));
    // The stack allocated depends on the maximum number of threads per SM, not on the actual number of threads per block.
    size_t total_stack_size = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor * config.stack_kb * 1000;
    cp.stats.print_memory_statistics(cp.config.verbose_solving, "stack_memory", total_stack_size);
  }
  return mem_config;
}

__global__ void initialize_global_data(
  UnifiedData* unified_data,
  bt::unique_ptr<GridData, bt::global_allocator>* grid_data_ptr)
{
  *grid_data_ptr = bt::make_unique<GridData, bt::global_allocator>(unified_data->root);
}

__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>* grid_data) {
  grid_data->reset();
}

__global__ void gpu_barebones_solve(UnifiedData* unified_data, GridData* grid_data) {
  auto& config = unified_data->root.config;
  if(threadIdx.x == 0 && blockIdx.x == 0 && config.verbose_solving) {
    printf("%% GPU kernel started, starting solving...\n");
  }
}

#else
void barebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
}
#endif // __CUDACC__
} // namespace barebones

#endif // TURBO_HYBRID_DIVE_AND_SOLVE_HPP
