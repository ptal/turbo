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
 * This has the disadvantage of memory transfers between CPU and GPU and synchronization overheads.
 * We propose a new "barebones" version which contains less abstractions than the GPU and hybrid versions, but have the same functionalities.
 * In particular, we directly implement the branch-and-bound algorithm here and avoid using `lala::BAB` and `lala::SearchTree` which are nice from a software engineering perspective but bring significant overhead.
 * This version is intended to reach the best possible performance.
 *
 * Terminology:
 *  * unified data: data available to both the CPU and GPU.
 *  * block data: data used within a single block.
 *  * grid data: data shared among all blocks in the grid.
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
   : root(GridCP::tag_gpu_block_copy{}, false, cp)
   , stop(false)
   , mem_config(mem_config)
  {
    size_t num_subproblems = 1;
    num_subproblems <<= root.config.subproblems_power;
    root.stats.eps_num_subproblems = num_subproblems;
  }
};

struct GridData;
using IStore = VStore<Itv, bt::pool_allocator>;
using IProp = PIR<IStore, bt::pool_allocator>;
using UB = ZUB<typename Itv::LB::value_type, bt::atomic_memory_grid>;
using strategies_type = bt::vector<StrategyType<bt::global_allocator>, bt::global_allocator>;

/** Data private to a single block. */
struct BlockData {
  /** The store of variables at the root of the current subproblem. */
  abstract_ptr<VStore<Itv, bt::global_allocator>> root_store;

  /** The best solution found so far in this block. */
  abstract_ptr<VStore<Itv, bt::global_allocator>> best_store;

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

  /** The path from `UnifiedData::root` to the current subproblem `root_store`. */
  size_t subproblem_idx;

  /** The best bound found so far by this block.
   * We always seek to minimize.
   * Note that if `GridData::appx_best_bound` is better than `best_bound` the current block does not necessarily record its best store in `best_store`.
   */
  UB best_bound;

  /** The current strategy being used to split the store.
   * It is an index into `GridData::strategies`.
   */
  int current_strategy;

  /** The next unassigned variable in the current strategy.
   * It is an index into `GridData::strategies.vars`.
   */
  int next_unassigned_var;

  /** On backtracking, the value to restore `current_strategy`. */
  int snapshot_root_strategy;

  /** On backtracking, the value to restore `next_unassigned_var`. */
  int snapshot_next_unassigned_var;

  /** The decision taken when exploring the tree. */
  bt::vector<LightBranch<Itv>, bt::global_allocator> decisions;

  /** Current depth of the search tree. */
  int depth;

  /** A timer used for computing time statistics. */
  cuda::std::chrono::system_clock::time_point timer;

  __device__ BlockData()
   : subproblem_idx(0)
   , current_strategy(0)
   , next_unassigned_var(0)
   , decisions(5000)
   , depth(0)
  {}

  __device__ void allocate(const UnifiedData& unified_data, const GridData& grid_data, unsigned char* shared_mem) {
    if(threadIdx.x == 0) {
      subproblem_idx = blockIdx.x;
      const MemoryConfig& mem_config = unified_data.mem_config;
      const auto& u_store = *(unified_data.root.store);
      const auto& u_iprop = *(unified_data.root.iprop);
      bt::pool_allocator shared_mem_pool(mem_config.make_shared_pool(shared_mem));
      bt::pool_allocator store_allocator(mem_config.make_store_pool(shared_mem_pool));
      bt::pool_allocator prop_allocator(mem_config.make_prop_pool(shared_mem_pool));
      root_store = bt::make_shared<VStore<Itv, bt::global_allocator>, bt::global_allocator>(u_store);
      best_store = bt::make_shared<VStore<Itv, bt::global_allocator>, bt::global_allocator>(u_store);
      store = bt::allocate_shared<IStore, bt::pool_allocator>(store_allocator, u_store, store_allocator);
      iprop = bt::allocate_shared<IProp, bt::pool_allocator>(prop_allocator, u_iprop, store, prop_allocator);
    }
  }

  /** We must deallocate store and iprop inside the kernel because they might be initialized in shared memory. */
  __device__ void deallocate_shared_data() {
    if(threadIdx.x == 0) {
      // NOTE: .reset() does not work because it does not reset the allocator, which is itself allocated in global memory.
      store = abstract_ptr<IStore>();
      iprop = abstract_ptr<IProp>();
    }
  }

  /** Add a new decision on the `decisions` stack and increase depth.
   * \param has_changed: A Boolean in shared memory.
   * \param strategies: A sequence of non-empty strategies.
   * \precondition: We must not be on a leaf node.
   */
  __device__ INLINE void split(bool& has_changed, const strategies_type& strategies) {
    using LB2 = typename Itv::LB::local_type;
    using UB2 = typename Itv::UB::local_type;
    __shared__ local::ZUB idx;
    decisions[depth].var = AVar{};
    int currentDepth = depth;
    for(int i = current_strategy; i < strategies.size(); ++i) {
      assert(strategies[i].vars.size() > 0);
      switch(strategies[i].var_order) {
        case VariableOrder::INPUT_ORDER: {
          input_order_split(has_changed, idx, strategies[i].val_order, strategies[i].vars.size(),
            [&](int j) { return strategies[i].vars[j]; });
          break;
        }
        case VariableOrder::FIRST_FAIL: {
          lattice_smallest_split(has_changed, idx, strategies[i],
            [](const Itv& u) { return UB2(u.ub().value() - u.lb().value()); });
          break;
        }
        case VariableOrder::ANTI_FIRST_FAIL: {
          lattice_smallest_split(has_changed, idx, strategies[i],
            [](const Itv& u) { return LB2(u.ub().value() - u.lb().value()); });
          break;
        }
        case VariableOrder::LARGEST: {
          lattice_smallest_split(has_changed, idx, strategies[i],
            [](const Itv& u) { return LB2(u.ub().value()); });
          break;
        }
        case VariableOrder::SMALLEST: {
          lattice_smallest_split(has_changed, idx, strategies[i],
            [](const Itv& u) { return UB2(u.lb().value()); });
        }
        default: assert(false);
      }
      __syncthreads();
      if(!decisions[currentDepth].var.is_untyped()) {
        return;
      }
      if(threadIdx.x == 0) {
        current_strategy++;
        next_unassigned_var = 0;
      }
      // `input_order_split` and `lattice_smallest_split` have a `__syncthreads()` before reading next_unassigned_var.
    }
    // Split on the remaining unassigned variables in the store, if any.
    // This is in case the search strategy is not complete.
    AType aty = store->aty();
    input_order_split(has_changed, idx, ValueOrder::MIN, store->vars(), [aty](int j) { return AVar(aty, j); });
  }

  /** Select the next unassigned variable with a finite interval in the array `strategy.vars()`.
   * We ignore infinite variables as splitting on them do not guarantee termination.
   * \param has_changed is a Boolean in shared memory.
   * \param idx is a decreasing integer in shared memory.
   */
  template <class F>
  __device__ INLINE void input_order_split(bool& has_changed, local::ZUB& idx, ValueOrder val_order,
    int num_vars, const F& f)
  {
    has_changed = true;
    if(threadIdx.x == 0) {
      idx = num_vars;
    }
    while(has_changed) {
      __syncthreads();
      // int n = idx.value();
      // __syncthreads();
      has_changed = false;
      for(int i = next_unassigned_var + threadIdx.x; i < num_vars; i += blockDim.x) {
        const auto& dom = store->project(f(i));
        if(dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top()) {
          has_changed |= idx.meet(local::ZUB(i));
        }
      }
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      next_unassigned_var = idx.value();
    }
    if(idx.value() != num_vars) {
      push_decision(val_order, f(idx.value()));
    }
  }

  /** Given an array of variable, select the variable `x` with the smallest value `f(store[x])` where "smallest" is defined according to the lattice order of the return type of `f`.
   * \param has_changed is a Boolean in shared memory.
   * \param idx is a decreasing integer in shared memory.
   * */
  template <class F>
  __device__ INLINE void lattice_smallest_split(bool& has_changed, local::ZUB& idx,
    const StrategyType<bt::global_allocator>& strategy, F f)
  {
    using T = decltype(f(Itv{}));
    __shared__ T value;
    has_changed = true;
    if(threadIdx.x == 0) {
      value = T::top();
      idx = strategy.vars.size();
    }
    /** This fixpoint loop seeks for the smallest `x` according to `f(x)` and the next unassigned variable. */
    while(has_changed) {
      __syncthreads();
      has_changed = false;
      for(int i = next_unassigned_var + threadIdx.x; i < strategy.vars.size(); i += blockDim.x) {
        const auto& dom = store->project(strategy.vars[i]);
        if(dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top()) {
          has_changed |= value.meet(f(dom));
          has_changed |= idx.meet(local::ZUB(i));
        }
      }
      __syncthreads();
    }
    /** If we found a value, we traverse again the variables' array to find its index. */
    if(!value.is_top()) {
      if(threadIdx.x == 0) {
        next_unassigned_var = idx.value();
        idx = strategy.vars.size();
      }
      __syncthreads();
      has_changed = true;
      // This fixpoint loop is not strictly necessary.
      // We keep it for determinism: the variable with the smallest index is selected first.
      while(has_changed) {
        int n = idx.value();
        __syncthreads();
        has_changed = false;
        for(int i = next_unassigned_var + threadIdx.x; i < strategy.vars.size(); i += blockDim.x) {
          const auto& dom = store->project(strategy.vars[i]);
          if(f(dom) == value) {
            has_changed |= idx.meet(local::ZUB(i));
          }
        }
        __syncthreads();
      }
      assert(idx.value() < strategy.vars.size());
      push_decision(strategy.val_order, strategy.vars[idx.value()]);
    }
  }

  /** Push a new decision onto the decisions stack.
   *  \precondition The domain of the variable `var` must not be empty, be a singleton or contain infinite bounds.
  */
  __device__ INLINE void push_decision(ValueOrder val_order, AVar var) {
    using value_type = typename Itv::LB::value_type;
    if(threadIdx.x == 0) {
      decisions[depth].var = var;
      decisions[depth].current_idx = -1;
      const auto& dom = store->project(decisions[depth].var);
      switch(val_order) {
        case ValueOrder::MIN: {
          decisions[depth].children[0] = Itv(dom.lb().value());
          decisions[depth].children[1] = Itv(dom.lb().value() + value_type{1}, dom.ub());
          break;
        }
        case ValueOrder::MAX: {
          decisions[depth].children[0] = Itv(dom.ub().value());
          decisions[depth].children[1] = Itv(dom.lb(), dom.ub().value() - value_type{1});
          break;
        }
        case ValueOrder::SPLIT: {
          auto mid = dom.lb().value() +  (dom.ub().value() - dom.lb().value()) / value_type{2};
          decisions[depth].children[0] = Itv(dom.lb(), mid);
          decisions[depth].children[1] = Itv(mid + value_type{1}, dom.ub());
          break;
        }
        case ValueOrder::REVERSE_SPLIT: {
          auto mid = dom.lb().value() +  (dom.ub().value() - dom.lb().value()) / value_type{2};
          decisions[depth].children[0] = Itv(mid + value_type{1}, dom.ub());
          decisions[depth].children[1] = Itv(dom.lb(), mid);
          break;
        }
        // ValueOrder::MEDIAN is not possible with interval.
        default: assert(false);
      }
      /** Ropes are a mechanisme for fast backtracking.
       * The rope of a left node is always the depth of the right node (also its depth), because after completing the exploration of the left subtree, we must visit the right subtree (rooted at the current depth).
       * The rope of the right node is inherited from its parent, we set -1 if there is no next node to visit.
       */
      decisions[depth].ropes[0] = depth + 1;
      decisions[depth].ropes[1] = depth > 0 ? decisions[depth-1].ropes[decisions[depth-1].current_idx] : -1;
      ++depth;
      // Reallocate decisions if needed.
      if(decisions.size() == depth) {
        decisions.resize(decisions.size() * 2);
      }
    }
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
   * We always seek to minimize.
   */
  UB appx_best_bound;

  /** Due to multithreading, we must protect `stdout` when printing.
   * The model of computation in this work is lock-free, but it seems unavoidable for printing.
  */
  cuda::binary_semaphore<cuda::thread_scope_device> print_lock;

  /** The search strategy is immutable and shared among the blocks. */
  strategies_type dive_strategies;

  /** The search strategy is immutable and shared among the blocks. */
  strategies_type solve_strategies;

  /** The objective variable to minimize.
   * Maximization problem are transformed into minimization problems by negating the objective variable.
   * Equal to -1 if the problem is a satisfaction problem.
   */
  AVar obj_var;

  __device__ GridData(const GridCP& root)
   : blocks(root.config.or_nodes)
   , next_subproblem(root.config.or_nodes)
   , print_lock(1)
   , dive_strategies(root.eps_split->strategies_())
   , solve_strategies(root.split->strategies_())
   , obj_var(root.bab->objective_var())
  {}
};

MemoryConfig configure_gpu_barebones(CP<Itv>&);
__global__ void initialize_global_data(UnifiedData*, bt::unique_ptr<GridData, bt::global_allocator>*);
__global__ void gpu_barebones_solve(UnifiedData*, GridData*);
__global__ void reduce_blocks(UnifiedData*, GridData*);
__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>*);

void barebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {
  if(config.print_intermediate_solutions) {
    printf("%% WARNING: -arch barebones is incompatible with -i and -a (it cannot print intermediate solutions).");
  }
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
  /** We wait that either the solving is interrupted, or that all threads have finished. */
  /** Block the signal CTRL-C to notify the threads if we must exit. */
  block_signal_ctrlc();
  gpu_barebones_solve
    <<<static_cast<unsigned int>(cp.config.or_nodes),
      CUDA_THREADS_PER_BLOCK,
      mem_config.shared_bytes>>>
    (unified_data.get(), grid_data->get());
  bool interrupted = wait_solving_ends(unified_data->stop, unified_data->root, start);
  CUDAEX(cudaDeviceSynchronize());
  reduce_blocks<<<1,1>>>(unified_data.get(), grid_data->get());
  CUDAEX(cudaDeviceSynchronize());
  auto& uroot = unified_data->root;
  if(uroot.stats.solutions > 0) {
    cp.print_solution(*uroot.best);
  }
  uroot.stats.print_mzn_final_separator();
  if(uroot.config.print_statistics) {
    uroot.config.print_mzn_statistics();
    uroot.stats.print_mzn_statistics(uroot.config.or_nodes);
    if(uroot.bab->is_optimization() && uroot.stats.solutions > 0) {
      uroot.stats.print_mzn_objective(uroot.best->project(uroot.bab->objective_var()), true);
    }
    unified_data->root.stats.print_mzn_end_stats();
  }
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
  size_t iprop_bytes = gpu_sizeof<IProp>() + gpu_sizeof<abstract_ptr<IProp>>() + cp.iprop->num_deductions() * gpu_sizeof<bytecode_type>() + gpu_sizeof<typename IProp::bytecodes_type>();
  MemoryConfig mem_config;
  if(config.only_global_memory) {
    mem_config = MemoryConfig(store_bytes, iprop_bytes);
  }
  else {
    mem_config = MemoryConfig((void*) gpu_barebones_solve, store_bytes, iprop_bytes);
  }
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

#define TIMEPOINT(KIND) \
  if(threadIdx.x == 0) { \
    block_data.timer = block_data.stats.stop_timer(Timer::KIND, block_data.timer); \
  }

__global__ void gpu_barebones_solve(UnifiedData* unified_data, GridData* grid_data) {
  extern __shared__ unsigned char shared_mem[];
  auto& config = unified_data->root.config;
  BlockData& block_data = grid_data->blocks[blockIdx.x];
  if(threadIdx.x == 0 && blockIdx.x == 0 && config.verbose_solving) {
    printf("%% GPU kernel started, starting solving...\n");
  }

  /** I. Initialization the block data and the fixpoint engine. */

  block_data.allocate(*unified_data, *grid_data, shared_mem);
  __syncthreads();
  IProp& iprop = *block_data.iprop;
  __shared__ FixpointSubsetGPU<BlockAsynchronousFixpointGPU<true>, bt::global_allocator, CUDA_THREADS_PER_BLOCK> fp_engine;
  fp_engine.init(iprop.num_deductions());
  /** This shared variable is necessary to avoid multiple threads to read into `unified_data.stop.test()`,
   * potentially reading different values and leading to deadlock. */
  __shared__ bool stop;
  __shared__ bool has_changed;
  __shared__ bool is_leaf_node;
  stop = false;
  auto group = cooperative_groups::this_thread_block();
  if(threadIdx.x == 0) {
    block_data.timer = block_data.stats.start_timer_device();
  }
  __syncthreads();

  /** II. Start the main solving loop. */

  size_t num_subproblems = unified_data->root.stats.eps_num_subproblems;
  while(block_data.subproblem_idx < num_subproblems && !stop) {
    if(config.verbose_solving && threadIdx.x == 0) {
      grid_data->print_lock.acquire();
      printf("%% Block %d solves subproblem num %" PRIu64 "\n", blockIdx.x, block_data.subproblem_idx);
      grid_data->print_lock.release();
    }

    // A. Restoring the current state to the root node.

    block_data.current_strategy = 0;
    block_data.next_unassigned_var = 0;
    block_data.depth = 0;
    unified_data->root.store->copy_to(group, *block_data.store);
    fp_engine.reset(iprop.num_deductions());
    __syncthreads();

    // B. Dive into the search tree until we reach the target subproblem.

    // C. Solve the current subproblem.

    while(!stop) {

      // D. Optimize the objective variable.

      if(threadIdx.x == 0 && !grid_data->obj_var.is_untyped()) {
        /** Before propagating, we update the local bound with the best known global bound.
         * Strenghten the objective variable to get a better objective next time.
         */
        block_data.best_bound.meet(grid_data->appx_best_bound);
        if(!block_data.best_bound.is_top()) {
          block_data.store->embed(grid_data->obj_var,
            Itv(Itv::LB::top(), Itv::UB(block_data.best_bound.value() - 1)));
        }
      }
      __syncthreads();
      // Unconstrained objective, can terminate, we will not find a better solution.
      if(block_data.best_bound.is_bot()) {
        stop = true;
        break;
      }
      TIMEPOINT(SEARCH);
      is_leaf_node = false;

      // E. Propagate the current node.

      int fp_iterations;
      switch(config.fixpoint) {
        case FixpointKind::AC1: {
          fp_iterations = fp_engine.fixpoint(
            [&](int i){ return iprop.deduce(i); },
            [&](){ return iprop.is_bot(); });
          break;
        }
        case FixpointKind::WAC1: {
          if(fp_engine.num_active() <= config.wac1_threshold) {
            fp_iterations = fp_engine.fixpoint(
              [&](int i){ return iprop.deduce(i); },
              [&](){ return iprop.is_bot(); });
          }
          else {
            fp_iterations = fp_engine.fixpoint(
              [&](int i){ return warp_fixpoint<CUDA_THREADS_PER_BLOCK>(iprop, i); },
              [&](){ return iprop.is_bot(); });
          }
          break;
        }
      }
      TIMEPOINT(FIXPOINT);

      // F. Analyze the result of propagation

      if(!iprop.is_bot()) {
        fp_engine.select([&](int i) { return !iprop.ask(i); });
        TIMEPOINT(SELECT_FP_FUNCTIONS);
        if(fp_engine.num_active() == 0) {
          is_leaf_node = true;
          if(threadIdx.x == 0) {
            block_data.best_bound.meet(Itv::UB(block_data.store->project(grid_data->obj_var).lb().value()));
            has_changed = grid_data->appx_best_bound.meet(block_data.best_bound);
          }
          __syncthreads();
          // When two blocks find the same bound, we avoid copying the store twice (although it can still happen if they concurrently meet the same bound in `appx_best_bound`).
          if(has_changed) {
            block_data.store->copy_to(group, *block_data.best_store);
            if(threadIdx.x == 0) {
              block_data.stats.solutions++;
              if(config.verbose_solving) {
                grid_data->print_lock.acquire();
                printf("%% objective="); block_data.best_bound.print(); printf("\n");
                grid_data->print_lock.release();
              }
            }
          }
        }
      }
      else {
        is_leaf_node = true;
      }

      if(threadIdx.x == 0) {
        block_data.stats.fixpoint_iterations += fp_iterations;
        block_data.stats.nodes++;
        block_data.stats.fails += (iprop.is_bot() ? 1 : 0);
        block_data.stats.depth_max = battery::max(block_data.stats.depth_max, block_data.depth);

        // G. Checking stopping conditions.

        if(block_data.stats.nodes >= config.stop_after_n_nodes
         || unified_data->stop.test())
        {
          block_data.stats.exhaustive = false;
          stop = true;
        }
      }
      __syncthreads();

      // H. Branching

      if(!is_leaf_node) {
        // If we are at the root of the current subproblem, we create a snapshot for future backtracking.
        if(block_data.depth == 0) {
          block_data.store->copy_to(group, *block_data.root_store);
        }
        block_data.split(has_changed, grid_data->solve_strategies);
        __syncthreads();
        // Split was not able to split a domain. It means that the search strategy is not complete due to unsplittable infinite domains.
        // We trigger backtracking, and set exhaustive to `false`.
        if(block_data.decisions[block_data.depth - 1].var.is_untyped()) {
          is_leaf_node = true;
          block_data.stats.exhaustive = false;
        }
        else if(threadIdx.x == 0) {
          // depth == 1 for root node because we just increased it in `block_data.split`.
          if(block_data.depth == 1) {
            block_data.snapshot_root_strategy = block_data.current_strategy;
            block_data.snapshot_next_unassigned_var = block_data.next_unassigned_var;
          }
          // Apply the decision.
          block_data.store->embed(block_data.decisions[block_data.depth-1].var, block_data.decisions[block_data.depth-1].next());
        }
      }

      // I. Backtracking

      if(is_leaf_node) {
        // Leaf node at root.
        if(block_data.depth == 0) {
          break;
        }
        __syncthreads();
        block_data.depth = block_data.decisions[block_data.depth-1].ropes[block_data.decisions[block_data.depth-1].current_idx];
        // Check if there is no more node to visit.
        if(block_data.depth == -1) {
          break;
        }
        // Restore from root by copying the store and re-applying all decisions from root to block_data.depth-1.
        fp_engine.reset(iprop.num_deductions());
        block_data.root_store->copy_to(group, *block_data.store);
        has_changed = true;
        while(has_changed) {
          __syncthreads();
          has_changed = false;
          for(int i = threadIdx.x; i < block_data.depth - 1; i += blockDim.x) {
            has_changed |= block_data.store->embed(block_data.decisions[i].var, block_data.decisions[i].current());
          }
          __syncthreads();
        }
        if(threadIdx.x == 0) {
          block_data.store->embed(block_data.decisions[block_data.depth - 1].var, block_data.decisions[block_data.depth - 1].next());
          block_data.current_strategy = block_data.snapshot_root_strategy;
          block_data.next_unassigned_var = block_data.snapshot_next_unassigned_var;
        }
      }
    }
  }
  __syncthreads();
  fp_engine.destroy();
  block_data.deallocate_shared_data();
}

__global__ void reduce_blocks(UnifiedData* unified_data, GridData* grid_data) {
  auto& root = unified_data->root;
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    root.stats.meet(grid_data->blocks[i].stats);
  }
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    auto& block = grid_data->blocks[i];
    if(block.stats.solutions > 0) {
      if(root.bab->is_satisfaction()) {
        block.best_store->extract(*root.best);
        break;
      }
      else {
        grid_data->appx_best_bound.meet(block.best_bound);
      }
    }
  }
  // If we found a bound, we copy the best store into the unified data.
  if(!grid_data->appx_best_bound.is_top()) {
    for(int i = 0; i < grid_data->blocks.size(); ++i) {
      auto& block = grid_data->blocks[i];
      if(block.best_bound == grid_data->appx_best_bound
       && block.best_store->project(grid_data->obj_var).lb() == block.best_bound)
      {
        block.best_store->copy_to(*root.best);
        break;
      }
    }
  }
}

__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>* grid_data) {
  grid_data->reset();
}

#else
void barebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
}
#endif // __CUDACC__
} // namespace barebones

#endif // TURBO_BAREBONES_DIVE_AND_SOLVE_HPP
