// Copyright 2025 Pierre Talbot

#ifndef TURBO_BAREBONES_DIVE_AND_SOLVE_HPP
#define TURBO_BAREBONES_DIVE_AND_SOLVE_HPP

#include "common_solving.hpp"
#include "memory_gpu.hpp"
#include "lala/light_branch.hpp"
#include <mutex>
#include <thread>
#include <chrono>

/** This is required in order to guess the usage of global memory, and increase the CUDA default limit. */
#define MAX_SEARCH_DEPTH 10000

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
#ifndef TURBO_IPC_ABSTRACT_DOMAIN

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
   * Invariant: `best_bound == best_store.project(obj_var).lb()`
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

  /** A timer used for computing diving VS search time statistics. */
  cuda::std::chrono::system_clock::time_point dive_timer;

  /** The time at which the kernel was started, useful to compute the time of the best bound. */
  cuda::std::chrono::system_clock::time_point start_time;

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
   * \param strategies: A sequence of strategies.
   * \precondition: We must not be on a leaf node.
   */
  __device__ INLINE void split(bool& has_changed, const strategies_type& strategies) {
    using LB2 = typename Itv::LB::local_type;
    using UB2 = typename Itv::UB::local_type;
    __shared__ local::ZUB idx;
    decisions[depth].var = AVar{};
    int currentDepth = depth;
    for(int i = current_strategy; i < strategies.size(); ++i) {
      switch(strategies[i].var_order) {
        case VariableOrder::RANDOM:
        case VariableOrder::INPUT_ORDER: {
          input_order_split(has_changed, idx, strategies[i]);
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
          break;
        }
        default: assert(false);
      }
      __syncthreads();
      // If we could find a variable with the current strategy, we return.
      if(!decisions[currentDepth].var.is_untyped()) {
        return;
      }
      if(threadIdx.x == 0) {
        current_strategy++;
        next_unassigned_var = 0;
      }
    }
    // `input_order_split` and `lattice_smallest_split` have a `__syncthreads()` before reading next_unassigned_var.
  }

  /** Select the next unassigned variable with a finite interval in the array `strategy.vars()` or `store` if the previous one is empty.
   * We ignore infinite variables as splitting on them do not guarantee termination.
   * \param has_changed is a Boolean in shared memory.
   * \param idx is a decreasing integer in shared memory.
   */
  __device__ INLINE void input_order_split(bool& has_changed, local::ZUB& idx, const StrategyType<bt::global_allocator>& strategy)
  {
    bool split_in_store = strategy.vars.empty();
    int n = split_in_store ? store->vars() : strategy.vars.size();
    if(threadIdx.x == 0) {
      has_changed = true;
      idx = n;
    }
    __syncthreads();
    while(has_changed) {
      __syncthreads();
      // int n = idx.value();
      if(threadIdx.x == 0) {
        has_changed = false;
      }
      __syncthreads();
      for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
        const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
        if(dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top()) {
          if(idx.meet(local::ZUB(i))) {
            has_changed = true;
          }
        }
      }
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      next_unassigned_var = idx.value();
      if(next_unassigned_var != n) {
        push_decision(strategy.val_order, split_in_store ? AVar{store->aty(), next_unassigned_var} : strategy.vars[next_unassigned_var]);
      }
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
    bool split_in_store = strategy.vars.empty();
    int n = split_in_store ? store->vars() : strategy.vars.size();
    __syncthreads();
    if(threadIdx.x == 0) {
      has_changed = true;
      value = T::top();
      idx = n;
    }
    __syncthreads();
    /** This fixpoint loop seeks for the smallest `x` according to `f(x)` and the next unassigned variable. */
    while(has_changed) {
      __syncthreads();
      if(threadIdx.x == 0) {
        has_changed = false;
      }
      __syncthreads();
      for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
        const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
        if(dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top()) {
          if(value.meet(f(dom))) {
            has_changed = true;
          }
          if(idx.meet(local::ZUB(i))) {
            has_changed = true;
          }
        }
      }
      __syncthreads();
    }
    /** If we found a value, we traverse again the variables' array to find its index. */
    if(!value.is_top()) {
      __syncthreads();
      if(threadIdx.x == 0) {
        next_unassigned_var = idx.value();
        idx = n;
        has_changed = true;
      }
      __syncthreads();
      // This fixpoint loop is not strictly necessary.
      // We keep it for determinism: the variable with the smallest index is selected first.
      while(has_changed) {
        // int n = idx.value();
        __syncthreads();
        has_changed = false;
        __syncthreads();
        for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
          const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
          if(dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top() && f(dom) == value) {
            if(idx.meet(local::ZUB(i))) {
              has_changed = true;
            }
          }
        }
        __syncthreads();
      }
      assert(idx.value() < n);
      if(threadIdx.x == 0) {
        if(split_in_store) {
          push_decision(strategy.val_order, AVar{store->aty(), idx.value()});
        }
        else {
          push_decision(strategy.val_order, strategy.vars[idx.value()]);
        }
      }
    }
  }

  /** Push a new decision onto the decisions stack.
   *  \precondition The domain of the variable `var` must not be empty, be a singleton or contain infinite bounds.
   *  \precondition Must be executed by thread 0 only.
  */
  __device__ INLINE void push_decision(ValueOrder val_order, AVar var) {
    using value_type = typename Itv::LB::value_type;
    assert(threadIdx.x == 0);
    decisions[depth].var = var;
    decisions[depth].current_idx = -1;
    const auto& dom = store->project(decisions[depth].var);
    assert(dom.lb().value() != dom.ub().value());
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
    /** Ropes are a mechanism for fast backtracking.
     * The rope of a left node is always the depth of the right node (also its depth), because after completing the exploration of the left subtree, we must visit the right subtree (rooted at the current depth).
     * The rope of the right node is inherited from its parent, we set -1 if there is no next node to visit.
     */
    decisions[depth].ropes[0] = depth + 1;
    decisions[depth].ropes[1] = depth > 0 ? decisions[depth-1].ropes[decisions[depth-1].current_idx] : -1;
    ++depth;
    // printf("depth(%d), var = %d, children = [%d, %d] | [%d, %d], ropes = [%d, %d]\n",
    //   depth, decisions[depth-1].var.vid(),
    //   (int)decisions[depth-1].children[0].lb().value(), (int)decisions[depth-1].children[0].ub().value(),
    //   (int)decisions[depth-1].children[1].lb().value(), (int)decisions[depth-1].children[1].ub().value(),
    //   decisions[depth-1].ropes[0], decisions[depth-1].ropes[1]);
    // Reallocate decisions if needed.
    if(decisions.size() == depth) {
      printf("resize to %d\n", (int)decisions.size() * 2);
      decisions.resize(decisions.size() * 2);
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

  /** A specific strategy is used for the subproblem decomposition during the diving phase. */
  bool has_eps_strategy;

  /** The search strategy is immutable and shared among the blocks. */
  strategies_type search_strategies;

  /** The objective variable to minimize.
   * Maximization problem are transformed into minimization problems by negating the objective variable.
   * Equal to -1 if the problem is a satisfaction problem.
   */
  AVar obj_var;

  __device__ GridData(const GridCP& root)
   : blocks(root.stats.num_blocks)
   , next_subproblem(root.stats.num_blocks)
   , print_lock(1)
   , has_eps_strategy(root.config.eps_var_order != "default")
   , search_strategies(root.split->strategies_())
   , obj_var(root.minimize_obj_var)
  {}
};

MemoryConfig configure_gpu_barebones(CP<Itv>&);
__global__ void initialize_global_data(UnifiedData*, bt::unique_ptr<GridData, bt::global_allocator>*);
__global__ void gpu_barebones_solve(UnifiedData*, GridData*);
template <class FPEngine>
__device__ INLINE void propagate(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data,
   FPEngine& fp_engine, bool& stop, bool& has_changed, bool& is_leaf_node);
__global__ void reduce_blocks(UnifiedData*, GridData*);
__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>*);

void barebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {
  if(config.print_intermediate_solutions) {
    printf("%% WARNING: -arch barebones is incompatible with -i and -a (it cannot print intermediate solutions).\n");
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
    <<<static_cast<unsigned int>(cp.stats.num_blocks),
      CUDA_THREADS_PER_BLOCK,
      mem_config.shared_bytes>>>
    (unified_data.get(), grid_data->get());
  auto now = std::chrono::steady_clock::now();
  int64_t time_to_kernel_start = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
  bool interrupted = wait_solving_ends(unified_data->stop, unified_data->root, start);
  CUDAEX(cudaDeviceSynchronize());
  reduce_blocks<<<1,1>>>(unified_data.get(), grid_data->get());
  CUDAEX(cudaDeviceSynchronize());
  auto& uroot = unified_data->root;
  if(uroot.stats.solutions > 0) {
    // We add the time before the kernel starts to the time needed to find the best bound.
    uroot.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND) += time_to_kernel_start;
    if(uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) != 0) {
      uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) += time_to_kernel_start;
    }
    cp.print_solution(*uroot.best);
  }
  uroot.stats.print_mzn_final_separator();
  if(uroot.config.print_statistics) {
    uroot.config.print_mzn_statistics();
    uroot.stats.print_mzn_statistics(uroot.config.verbose_solving);
    if(uroot.bab->is_optimization() && uroot.stats.solutions > 0) {
      uroot.stats.print_mzn_objective(uroot.best->project(uroot.bab->objective_var()), uroot.bab->is_minimization());
    }
    unified_data->root.stats.print_mzn_end_stats();
  }
  deallocate_global_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
}

/** We configure the GPU according to the user configuration:
 * 1) Guess the "best" number of blocks per SM, if not provided.
 * 2) Update the number of subproblems to at least "30 * B" where B is the number of blocks.
 * 3) Configure the size of the shared memory.
 * 4) Increase the global heap memory.
 * 5) Increase the stack size if requested by the user.
 */
MemoryConfig configure_gpu_barebones(CP<Itv>& cp) {
  auto& config = cp.config;

  /** I. Number of blocks per SM. */
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int max_block_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_block_per_sm, (void*) gpu_barebones_solve, CUDA_THREADS_PER_BLOCK, 0);
  if(cp.config.verbose_solving) {
    printf("%% max_blocks_per_sm=%d\n", max_block_per_sm);
  }
  if(cp.config.or_nodes != 0) {
    cp.stats.num_blocks = std::min(max_block_per_sm * deviceProp.multiProcessorCount, (int)cp.config.or_nodes);
    if(cp.config.verbose_solving >= 1 && cp.stats.num_blocks < cp.config.or_nodes) {
      printf("%% WARNING: -or %d is too high on your GPU architecture, it has been reduced to %d.\n", (int)cp.config.or_nodes, cp.stats.num_blocks);
    }
  }
  else {
    cp.stats.num_blocks = max_block_per_sm * deviceProp.multiProcessorCount;
  }

  /** II. Number of subproblems. */
  cp.stats.print_stat("subproblems_power", cp.config.subproblems_power);
  if(cp.config.subproblems_power == -1) {
    cp.config.subproblems_power = 0;
    while((1 << cp.config.subproblems_power) < 30 * cp.stats.num_blocks) {
      cp.config.subproblems_power++;
    }
  }

  /** III. Size of the heap global memory.
   * The estimation is very conservative, normally we should not run out of memory.
   * */
  size_t store_bytes = gpu_sizeof<IStore>() + gpu_sizeof<abstract_ptr<IStore>>() + cp.store->vars() * gpu_sizeof<Itv>();
  size_t iprop_bytes = gpu_sizeof<IProp>() + gpu_sizeof<abstract_ptr<IProp>>() + cp.iprop->num_deductions() * gpu_sizeof<bytecode_type>() + gpu_sizeof<typename IProp::bytecodes_type>();
  size_t mem_per_block = gpu_sizeof<BlockData>()
    + store_bytes * size_t{3}  // current, root, best.
    + store_bytes * size_t{2}  // search strategies
    + iprop_bytes * size_t{2}
    + cp.iprop->num_deductions() * size_t{4} * gpu_sizeof<int>()  // fixpoint engine
    + (gpu_sizeof<int>() + gpu_sizeof<LightBranch<Itv>>()) * size_t{MAX_SEARCH_DEPTH};
  size_t estimated_global_mem = gpu_sizeof<UnifiedData>() + store_bytes * size_t{5} + iprop_bytes +
    gpu_sizeof<GridData>();

  size_t mem_for_blocks = deviceProp.totalGlobalMem - estimated_global_mem - (deviceProp.totalGlobalMem / 100 * 10);
  cp.stats.num_blocks = std::max(size_t{1}, std::min(mem_for_blocks / mem_per_block, static_cast<size_t>(cp.stats.num_blocks)));
  estimated_global_mem += cp.stats.num_blocks * mem_per_block;
  if(estimated_global_mem > deviceProp.totalGlobalMem / 100 * 90) {
    printf("%% WARNING: The estimated global memory is larger than 90%% of the total global memory.\n\
%% It is possible to run out of memory during solving.\n");
  }
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, deviceProp.totalGlobalMem / 100 * 97));
  cp.stats.print_memory_statistics(cp.config.verbose_solving, "heap_memory", estimated_global_mem);
  cp.stats.print_memory_statistics(cp.config.verbose_solving, "mem_per_block", mem_per_block);
  cp.stats.print_memory_statistics(cp.config.verbose_solving, "total_global_mem_bytes", deviceProp.totalGlobalMem);
  cp.stats.print_stat("num_blocks", cp.stats.num_blocks);

  /** IV. Increase the stack if requested by the user. */
  if(config.stack_kb != 0) {
    CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, config.stack_kb*1000));
    // The stack allocated depends on the maximum number of threads per SM, not on the actual number of threads per block.
    size_t total_stack_size = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor * config.stack_kb * 1000;
    cp.stats.print_memory_statistics(cp.config.verbose_solving, "stack_memory", total_stack_size);
  }

  /** V. Configure the shared memory size. */
  int blocks_per_sm = std::max(1, (cp.stats.num_blocks + deviceProp.multiProcessorCount - 1) / deviceProp.multiProcessorCount);
  MemoryConfig mem_config;
  if(config.only_global_memory) {
    mem_config = MemoryConfig(store_bytes, iprop_bytes);
  }
  else {
    mem_config = MemoryConfig((void*) gpu_barebones_solve, config.verbose_solving, blocks_per_sm, store_bytes, iprop_bytes);
  }
  mem_config.print_mzn_statistics(config, cp.stats);
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

  /** A. Initialization the block data and the fixpoint engine. */

  block_data.allocate(*unified_data, *grid_data, shared_mem);
  __syncthreads();
  IProp& iprop = *block_data.iprop;
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
  __shared__ BlockAsynchronousFixpointGPU<true> fp_engine;
#else
  __shared__ FixpointSubsetGPU<BlockAsynchronousFixpointGPU<true>, bt::global_allocator, CUDA_THREADS_PER_BLOCK> fp_engine;
  fp_engine.init(iprop.num_deductions());
#endif
  /** This shared variable is necessary to avoid multiple threads to read into `unified_data.stop.test()`,
   * potentially reading different values and leading to deadlock. */
  __shared__ bool stop;
  __shared__ bool has_changed;
  __shared__ bool is_leaf_node;
  __shared__ int remaining_depth;
  stop = false;
  auto group = cooperative_groups::this_thread_block();
  if(threadIdx.x == 0) {
    block_data.timer = block_data.stats.start_timer_device();
    block_data.start_time = block_data.timer;
  }
  __syncthreads();

  /** B. Start the main dive and solve loop. */

  size_t num_subproblems = unified_data->root.stats.eps_num_subproblems;
  while(block_data.subproblem_idx < num_subproblems && !stop) {
    if(config.verbose_solving >= 2 && threadIdx.x == 0) {
      grid_data->print_lock.acquire();
      printf("%% Block %d solves subproblem num %" PRIu64 "\n", blockIdx.x, block_data.subproblem_idx);
      grid_data->print_lock.release();
    }

    // C. Restoring the current state to the root node.

    block_data.current_strategy = 0;
    block_data.next_unassigned_var = 0;
    block_data.depth = 0;
    unified_data->root.store->copy_to(group, *block_data.store);
#ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
    fp_engine.reset(iprop.num_deductions());
#endif
    __syncthreads();

    // D. Dive into the search tree until we reach the target subproblem.
    remaining_depth = config.subproblems_power;
    if(threadIdx.x == 0) {
      block_data.dive_timer = block_data.stats.start_timer_device();
      is_leaf_node = false;
    }
    __syncthreads();
    while(remaining_depth > 0 && !is_leaf_node && !stop) {
      __syncthreads();
      propagate(*unified_data, *grid_data, block_data, fp_engine, stop, has_changed, is_leaf_node);
      __syncthreads();
      if(!is_leaf_node) {
        block_data.split(has_changed, grid_data->search_strategies);
        __syncthreads();
        // Split was not able to split a domain. It means that the search strategy is not complete due to unsplittable infinite domains.
        // We skip the subtree, and set exhaustive to `false`.
        if(block_data.decisions[0].var.is_untyped()) {
          is_leaf_node = true;
          block_data.stats.exhaustive = false;
          if(threadIdx.x == 0 && config.verbose_solving >= 1) { printf("%% WARNING: infinite element detected during branching, search is not exhaustive\n");}
        }
        else if(threadIdx.x == 0) {
          --remaining_depth;
          // We do not record the decisions when diving.
          --block_data.depth;
          /** We commit to one of the branches depending on the current value on the path.
           * Suppose the depth is 3, the path is "010" we are currently at `remaining_depth = 1`.
           * We must extract the bit "1", and we do so by standard bitwise manipulation.
           * Whenever the branch_idx is 0 means to take the left branch, and 1 means to take the right branch.
           */
          size_t branch_idx = (block_data.subproblem_idx & (size_t{1} << remaining_depth)) >> remaining_depth;
          /** We immediately commit to the branch. */
          // printf("split on %d (", block_data.decisions[0].var.vid()); block_data.store->project(block_data.decisions[0].var).print(); printf(")\n");
          block_data.store->embed(block_data.decisions[0].var, block_data.decisions[0].children[branch_idx]);
        }
      }
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      block_data.stats.stop_timer(Timer::DIVE, block_data.dive_timer);
    }
    // E. Skip subproblems that are not reachable.

    /** If we reached a leaf node before the subproblem was reached, then it means a whole subtree should be skipped. */
    if(is_leaf_node && !stop) {
       /** To skip all the paths of the subtree obtained, we perform bitwise operations.
       * Suppose the current path is "00" turn left two times, and the following search tree:
       *         *         depth = 0
       *        / \
       *      0    1       depth = 1
       *    / \   / \
       *   00 01 10 11     depth = 2
       *
       * If we detect a leaf node at depth 1, after only one left turn, we must skip the remaining of the subtree, in particular to avoid exploring the path "01".
       * To achieve that, we take the current path "00", shift it to the right by 1 (essentially erasing the path that has not been explored), increment it to go to the next subtree (at the same depth), and shift it back to the left to reach the first subproblem of the subtree.
       * Cool huh?
       */
      if(threadIdx.x == 0) {
        size_t next_subproblem_idx = ((block_data.subproblem_idx >> remaining_depth) + size_t{1}) << remaining_depth;
        // Make sure the subtree is skipped.
        while(grid_data->next_subproblem.meet(ZLB<size_t, bt::local_memory>(next_subproblem_idx))) {}
        /** It is possible that other blocks skip similar subtrees.
          * Hence, we only count the subproblems skipped by the block solving the left most subproblem. */
        if((block_data.subproblem_idx & ((size_t{1} << remaining_depth) - size_t{1})) == size_t{0}) {
          block_data.stats.eps_skipped_subproblems += next_subproblem_idx - block_data.subproblem_idx;
        }
      }
    }
    else if(!stop) {

      // F. Solve the current subproblem.

      // We skip the remaining of the EPS strategy if there is any.
      if(threadIdx.x == 0 && grid_data->has_eps_strategy) {
        block_data.current_strategy = battery::max(1, block_data.current_strategy);
        block_data.next_unassigned_var = 0;
      }

      while(!stop) {

        // I. Optimize the objective variable (only if not diving).

        if(threadIdx.x == 0 && !grid_data->obj_var.is_untyped()) {
          /** Before propagating, we update the local bound with the best known global bound.
           * Strenghten the objective variable to get a better objective next time.
           */
          if(!grid_data->appx_best_bound.is_top()) {
            block_data.store->embed(grid_data->obj_var,
              Itv(Itv::LB::top(), Itv::UB(grid_data->appx_best_bound.value() - 1)));
            block_data.store->embed(grid_data->obj_var,
              Itv(Itv::LB::top(), Itv::UB(block_data.best_bound.value() - 1)));
          }
          // Unconstrained objective, can terminate, we will not find a better solution.
          if(grid_data->appx_best_bound.is_bot()) {
            stop = true;
            unified_data->stop.test_and_set();
          }
        }
        __syncthreads();
        if(stop) {
          break;
        }

        // II. Propagate the current node.
        propagate(*unified_data, *grid_data, block_data, fp_engine, stop, has_changed, is_leaf_node);
        __syncthreads();

        // III. Branching

        if(!is_leaf_node) {
          // If we are at the root of the current subproblem, we create a snapshot for future backtracking.
          if(block_data.depth == 0) {
            block_data.store->copy_to(group, *block_data.root_store);
            if(threadIdx.x == 0) {
              block_data.snapshot_root_strategy = block_data.current_strategy;
              block_data.snapshot_next_unassigned_var = block_data.next_unassigned_var;
            }
          }
          __syncthreads();
          block_data.split(has_changed, grid_data->search_strategies);
          __syncthreads();
          // Split was not able to split a domain. It means that the search strategy is not complete due to unsplittable infinite domains.
          // We trigger backtracking, and set exhaustive to `false`.
          if(block_data.decisions[block_data.depth - 1].var.is_untyped()) {
            is_leaf_node = true;
            block_data.stats.exhaustive = false;
            if(threadIdx.x == 0 && config.verbose_solving >= 1) { printf("%% WARNING: infinite element detected during branching, search is not exhaustive\n");}
          }
          else if(threadIdx.x == 0) {
            // Apply the decision.
            // printf("split on %d (", block_data.decisions[block_data.depth-1].var.vid()); block_data.store->project(block_data.decisions[block_data.depth-1].var).print(); printf(")\n");
            block_data.store->embed(block_data.decisions[block_data.depth-1].var, block_data.decisions[block_data.depth-1].next());
            // printf("left decision: %d [", block_data.decisions[block_data.depth - 1].var.vid()); block_data.decisions[block_data.depth - 1].current().print(); printf("]\n");
          }
        }

        // IV. Backtracking

        if(is_leaf_node) {
          // Leaf node at root.
          if(block_data.depth == 0) {
            break;
          }
          if(threadIdx.x == 0) {
            block_data.depth = block_data.decisions[block_data.depth-1].ropes[block_data.decisions[block_data.depth-1].current_idx];
          }
          __syncthreads();
          // Check if there is no more node to visit.
          if(block_data.depth == -1) {
            break;
          }
          // Restore from root by copying the store and re-applying all decisions from root to block_data.depth-1.
#ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
          fp_engine.reset(iprop.num_deductions());
#endif
          block_data.root_store->copy_to(group, *block_data.store);
          // __syncthreads();
          // if(threadIdx.x == 0) {
          //   printf("%d: restoring store: ", block_data.depth); block_data.store->print(); printf("\n");
          // }
          // __syncthreads();
          if(threadIdx.x == 0) {
            has_changed = true;
          }
          __syncthreads();
          while(has_changed) {
            __syncthreads();
            if(threadIdx.x == 0) {
              has_changed = false;
            }
            __syncthreads();
            for(int i = threadIdx.x; i < block_data.depth - 1; i += blockDim.x) {
              if(block_data.store->embed(block_data.decisions[i].var, block_data.decisions[i].current())) {
                has_changed = true;
              }
            }
            __syncthreads();
          }
          if(threadIdx.x == 0) {
            block_data.store->embed(block_data.decisions[block_data.depth - 1].var, block_data.decisions[block_data.depth - 1].next());
            // printf("right decision: %d [", block_data.decisions[block_data.depth - 1].var.vid()); block_data.decisions[block_data.depth - 1].current().print(); printf("]\n");
            block_data.current_strategy = block_data.snapshot_root_strategy;
            block_data.next_unassigned_var = block_data.snapshot_next_unassigned_var;
          }
          // __syncthreads();
          // if(threadIdx.x == 0) {
          //   printf("%d: reapplied decisions: ", block_data.depth); block_data.store->print(); printf("\n");
          // }
          // __syncthreads();
        }
      }
      /** If we didn't stop solving because of an external interruption, we increase the number of subproblems solved. */
      if(threadIdx.x == 0 && block_data.stats.nodes < config.stop_after_n_nodes
        && !unified_data->stop.test())
      {
        block_data.stats.eps_solved_subproblems += 1;
      }
    }

    // G. Move to the next subproblem.

    /** We prepare the block to solve the next problem.
     * We update the subproblem index to the next subproblem to solve. */
    if(threadIdx.x == 0 && !stop) {
      /** To avoid that several blocks solve the same subproblem, we use an atomic post-increment. */
      block_data.subproblem_idx = grid_data->next_subproblem.atomic()++;
      /** The following commented code is completely valid and does not use atomic post-increment.
       * But honestly, we kinda need more performance so... let's avoid reexploring subproblems. */
      // subproblem_idx = grid_data->next_subproblem.value();
      // grid_data->next_subproblem.meet(ZLB<size_t, bt::local_memory>(subproblem_idx + size_t{1}));
    }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(block_data.stats.nodes < config.stop_after_n_nodes && !unified_data->stop.test()) {
      block_data.stats.num_blocks_done = 1;
    }
    block_data.stats.timers.update_timer(Timer::FIRST_BLOCK_IDLE, block_data.start_time);
    block_data.stats.cumulative_time_block = block_data.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE);
  }
  __syncthreads();
#ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
  fp_engine.destroy();
#endif
  block_data.deallocate_shared_data();
  __syncthreads();
}

template <class FPEngine>
__device__ INLINE void propagate(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data,
   FPEngine& fp_engine, bool& stop, bool& has_changed, bool& is_leaf_node)
{
  __shared__ int warp_iterations[CUDA_THREADS_PER_BLOCK/32];
  warp_iterations[threadIdx.x / 32] = 0;
  auto& config = unified_data.root.config;
  IProp& iprop = *block_data.iprop;
  auto group = cooperative_groups::this_thread_block();

  TIMEPOINT(SEARCH);
  if(threadIdx.x == 0) {
    is_leaf_node = false;
  }

  // II. Compute the fixpoint of the current node.
  int fp_iterations;
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
  int num_active = iprop.num_deductions();
#else
  int num_active = fp_engine.num_active();
#endif
  switch(config.fixpoint) {
    case FixpointKind::AC1: {
      fp_iterations = fp_engine.fixpoint(
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
        iprop.num_deductions(),
#endif
        [&](int i){ return iprop.deduce(i); },
        [&](){ return iprop.is_bot(); });
      if(threadIdx.x == 0) {
        block_data.stats.num_deductions += fp_iterations * num_active;
      }
      break;
    }
    case FixpointKind::WAC1: {
      if(num_active <= config.wac1_threshold) {
        fp_iterations = fp_engine.fixpoint(
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
        iprop.num_deductions(),
#endif
          [&](int i){ return iprop.deduce(i); },
          [&](){ return iprop.is_bot(); });
        if(threadIdx.x == 0) {
          block_data.stats.num_deductions += fp_iterations * num_active;
        }
      }
      else {
        fp_iterations = fp_engine.fixpoint(
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
          iprop.num_deductions(),
#endif
          [&](int i){ return warp_fixpoint<CUDA_THREADS_PER_BLOCK>(iprop, i, warp_iterations); },
          [&](){ return iprop.is_bot(); });
        if(threadIdx.x == 0) {
          for(int i = 0; i < CUDA_THREADS_PER_BLOCK/32; ++i) {
            block_data.stats.num_deductions += warp_iterations[i] * 32;
          }
        }
      }
      break;
    }
  }
  TIMEPOINT(FIXPOINT);

  // III. Analyze the result of propagation

  if(!iprop.is_bot()) {
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
    if(threadIdx.x == 0) {
      has_changed = false;
    }
    __syncthreads();
    for(int i = threadIdx.x; !has_changed && i < iprop.num_deductions(); i += blockDim.x) {
      if(!iprop.ask(i)) {
        has_changed = true;
      }
    }
    __syncthreads();
    num_active = has_changed ? 1 : 0;
#else
    fp_engine.select([&](int i) { return !iprop.ask(i); });
    num_active = fp_engine.num_active();
#endif
    TIMEPOINT(SELECT_FP_FUNCTIONS);
    /** Whenever we reach a solution node, we must have a bound better than the best bound of the local block.
     * Note that it doesn't mean the best bound of the block must be the best bound of the grid.
     * It is to prevent copying a store with a worst bound into `best_store`.
     */
    if(num_active == 0) {
      is_leaf_node = true;
      if(block_data.best_bound.value() > block_data.store->project(grid_data.obj_var).lb().value()) {
        if(threadIdx.x == 0) {
          block_data.best_bound.meet(Itv::UB(block_data.store->project(grid_data.obj_var).lb().value()));
          grid_data.appx_best_bound.meet(block_data.best_bound);
          block_data.stats.timers.update_timer(Timer::LATEST_BEST_OBJ_FOUND, block_data.start_time);
        }
        block_data.store->copy_to(group, *block_data.best_store);
        if(threadIdx.x == 0) {
          block_data.stats.solutions++;
          if(config.verbose_solving >= 2) {
            grid_data.print_lock.acquire();
            printf("%% objective="); block_data.best_bound.print(); printf("\n");
            grid_data.print_lock.release();
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

    // IV. Checking stopping conditions.

    if(block_data.stats.nodes >= config.stop_after_n_nodes
      || unified_data.stop.test())
    {
      block_data.stats.exhaustive = false;
      stop = true;
    }
  }
}

__global__ void reduce_blocks(UnifiedData* unified_data, GridData* grid_data) {
  auto& root = unified_data->root;
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    root.stats.meet(grid_data->blocks[i].stats);
    int64_t& grid_first_block_idle = root.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE);
    int64_t block_idle = grid_data->blocks[i].stats.timers.time_of(Timer::FIRST_BLOCK_IDLE);
    if(grid_first_block_idle > block_idle) {
      grid_first_block_idle = block_idle;
    }
  }
  int best_block_idx = 0;
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    auto& block = grid_data->blocks[i];
    if(block.stats.solutions > 0) {
      if(root.bab->is_satisfaction()) {
        block.best_store->extract(*root.best);
        break;
      }
      else {
        bool equal_bound = (grid_data->appx_best_bound == block.best_bound);
        bool is_better = grid_data->appx_best_bound.meet(block.best_bound);
        int64_t& grid_best_time = root.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND);
        int64_t block_best_time = block.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND);
        if(is_better || (equal_bound && block_best_time <= grid_best_time)) {
          grid_best_time = block_best_time;
          best_block_idx = i;
        }
      }
    }
  }
  // If we found a bound, we copy the best store into the unified data.
  if(!grid_data->appx_best_bound.is_top()) {
    grid_data->blocks[best_block_idx].best_store->copy_to(*root.best);
  }
}

__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>* grid_data) {
  grid_data->reset();
}

#endif // TURBO_IPC_ABSTRACT_DOMAIN
#endif // __CUDACC__

#if defined(TURBO_IPC_ABSTRACT_DOMAIN) || !defined(__CUDACC__)

void barebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {
#ifdef TURBO_IPC_ABSTRACT_DOMAIN
  std::cerr << "-arch barebones does not support IPC abstract domain." << std::endl;
#else
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
#endif
}

#endif

} // namespace barebones

#endif // TURBO_BAREBONES_DIVE_AND_SOLVE_HPP
