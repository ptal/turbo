// Copyright 2023 Pierre Talbot

#ifndef TURBO_GPU_DIVE_AND_SOLVE_HPP
#define TURBO_GPU_DIVE_AND_SOLVE_HPP

#include "common_solving.hpp"
#include "memory_gpu.hpp"
#include <thread>
#include <algorithm>

namespace bt = ::battery;

#ifdef __CUDACC__

#include <cuda/std/chrono>
#include <cuda/semaphore>

template <
  class Universe0, // Universe used locally to one thread.
  class Universe1, // Universe used in the scope of a block.
  class Universe2, // Universe used in the scope of a grid.
  class ConcurrentAllocator> // This allocator allocates memory that can be both accessed by the CPU and the GPU. See PR #18 for the reason (basically, non-Linux systems do not support concurrent managed memory (accessed from CPU and GPU) and must rely on pinned memory instead).
struct StateTypes
{
  using U0 = Universe0;
  using U1 = Universe1;
  using U2 = Universe2;
  using concurrent_allocator = ConcurrentAllocator;

  /** We first interpret the formula in an abstract domain with sequential concurrent memory, that we call `GridCP`. */
  using GridCP = AbstractDomains<U0,
    bt::statistics_allocator<ConcurrentAllocator>,
    bt::statistics_allocator<UniqueLightAlloc<ConcurrentAllocator, 0>>,
    bt::statistics_allocator<UniqueLightAlloc<ConcurrentAllocator, 1>>>;

  /** Then, once everything is initialized, we rely on a parallel abstract domain called `BlockCP`, usually using atomic shared and global memory. */
  using BlockCP = AbstractDomains<U1,
    bt::global_allocator,
    bt::pool_allocator,
    UniqueAlloc<bt::pool_allocator, 0>>;
};

using Itv0 = Interval<ZLB<bound_value_type, bt::local_memory>>;
using Itv1 = Interval<ZLB<bound_value_type, bt::atomic_memory_block>>;
using Itv2 = Interval<ZLB<bound_value_type, bt::atomic_memory_grid>>;
using AtomicBool = B<bt::atomic_memory_block>;
using FPEngine = FixpointSubsetGPU<BlockAsynchronousFixpointGPU<true>, bt::global_allocator, CUDA_THREADS_PER_BLOCK>;

// Version for non-Linux systems such as Windows where pinned memory must be used (see PR #19).
#ifdef NO_CONCURRENT_MANAGED_MEMORY
  using ItvSolverPinned = StateTypes<Itv0, Itv1, Itv2, bt::pinned_allocator>;
#else
  // using ItvSolver = StateTypes<Itv0, Itv1, Itv2, bt::managed_allocator>;
  using ItvSolver = StateTypes<Itv0, Itv0, Itv0, bt::managed_allocator>;
#endif

template <class S>
struct BlockData;

/** `GridData` represents the problem to be solved (`root`), the data of each block when running EPS (`blocks`), the index of the subproblem that needs to be solved next (`next_subproblem`), an estimation of the best bound found across subproblems in case of optimisation problem and other global information.
 */
template <class S>
struct GridData {
  using GridCP = typename S::GridCP;
  using BlockCP = typename S::BlockCP;
  using U2 = typename S::U2;

  GridCP root;
  // `blocks_root` is a copy of `root` but with the same allocators than the ones used in `blocks`.
  // This is helpful to share immutable data among blocks (for instance the propagators).
  bt::shared_ptr<BlockCP, bt::global_allocator> blocks_root;
  // Stop from the CPU, for instance because of a timeout.
  cuda::std::atomic_flag cpu_stop;
  // Boolean indicating that the blocks have been reduced, and the CPU can now print the statistics.
  volatile bool blocks_reduced;
  MemoryConfig mem_config;
  bt::vector<BlockData<S>, bt::global_allocator> blocks;
  // Stop from a block on the GPU, for instance because we found a solution.
  bt::shared_ptr<B<bt::atomic_memory_grid>, bt::global_allocator> gpu_stop;
  bt::shared_ptr<ZLB<size_t, bt::atomic_memory_grid>, bt::global_allocator> next_subproblem;
  bt::shared_ptr<U2, bt::global_allocator> best_bound;

  // All of what follows is only to support printing while the kernel is running.
  // In particular, we transfer the solution to the CPU where it is printed, because printing on the GPU can be very slow when the problem is large.
  bt::shared_ptr<cuda::binary_semaphore<cuda::thread_scope_device>, bt::global_allocator> print_lock;
  cuda::std::atomic_flag ready_to_produce;
  cuda::std::atomic_flag ready_to_consume;

  GridData(const GridCP& root, const MemoryConfig& mem_config)
    : root(root)
    , mem_config(mem_config)
    , cpu_stop(false)
    , blocks_reduced(false)
  {
    ready_to_consume.clear();
    ready_to_produce.clear();
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
  }

  template <class BlockBAB>
  __device__ void produce_solution(const BlockBAB& bab) {
    print_lock->acquire();
    if(!cpu_stop.test()) {
      bab.extract(*(root.bab));
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
      ready_to_consume.test_and_set(cuda::std::memory_order_seq_cst);
      ready_to_consume.notify_one();
      // Wait the CPU has consumed the solution before continuing.
      // This avoids a problem with "blocks_reduced" being `true` too fast.
      ready_to_produce.wait(false, cuda::std::memory_order_seq_cst);
      ready_to_produce.clear();
    }
    print_lock->release();
  }

  __host__ bool consume_solution() {
    ready_to_consume.wait(false, cuda::std::memory_order_seq_cst);
    ready_to_consume.clear();
    if(blocks_reduced) {
      root.print_final_solution();
      if(root.config.print_statistics) {
        root.print_mzn_statistics();
      }
      return true;
    }
    else {
      root.print_solution();
    }
    ready_to_produce.test_and_set(cuda::std::memory_order_seq_cst);
    ready_to_produce.notify_one();
    return false;
  }

  __device__ void allocate() {
    assert(threadIdx.x == 0 && blockIdx.x == 0);
    auto root_mem_config(mem_config);
    root_mem_config.mem_kind = MemoryKind::GLOBAL;
    blocks_root = bt::make_shared<BlockCP, bt::global_allocator>(
      typename BlockCP::tag_gpu_block_copy{},
      false, // Due to different allocators between BlockCP and GridCP, it won't be able to share data anyways.
      root,
      bt::global_allocator{},
      root_mem_config.make_prop_pool(bt::pool_allocator(nullptr,0)),
      root_mem_config.make_store_pool(bt::pool_allocator(nullptr,0)));
    blocks = bt::vector<BlockData<S>, bt::global_allocator>(root.stats.num_blocks);
    gpu_stop = bt::make_shared<B<bt::atomic_memory_grid>, bt::global_allocator>(false);
    print_lock = bt::make_shared<cuda::binary_semaphore<cuda::thread_scope_device>, bt::global_allocator>(1);
    next_subproblem = bt::make_shared<ZLB<size_t, bt::atomic_memory_grid>, bt::global_allocator>(0);
    best_bound = bt::make_shared<U2, bt::global_allocator>();
  }

  __device__ void deallocate() {
    assert(threadIdx.x == 0 && blockIdx.x == 0);
    blocks = bt::vector<BlockData<S>, bt::global_allocator>();
    blocks_root->deallocate();
    blocks_root.reset();
    gpu_stop.reset();
    print_lock.reset();
    next_subproblem.reset();
    best_bound.reset();
  }
};

/** `BlockData` contains all the structures required to solve a subproblem including the problem itself (`root`) and the fixpoint engine (`fp_engine`). */
template <class S>
struct BlockData {
  using GridCP = typename S::GridCP;
  using BlockCP = typename S::BlockCP;

  using snapshot_type = typename BlockCP::IST::snapshot_type<bt::global_allocator>;
  size_t subproblem_idx;
  bt::shared_ptr<FPEngine, bt::pool_allocator> fp_engine;
  bt::shared_ptr<AtomicBool, bt::pool_allocator> has_changed;
  bt::shared_ptr<AtomicBool, bt::pool_allocator> stop;
  bt::shared_ptr<BlockCP, bt::global_allocator> root;
  bt::shared_ptr<snapshot_type, bt::global_allocator> snapshot_root;

  __device__ BlockData():
    has_changed(nullptr, bt::pool_allocator(nullptr, 0)),
    stop(nullptr, bt::pool_allocator(nullptr, 0))
  {}

public:
  /** Initialize the block data.
  * Allocate the abstract domains in the best memory available depending on how large are the abstract domains. */
  __device__ void allocate(GridData<S>& grid_data, unsigned char* shared_mem) {
    auto block = cooperative_groups::this_thread_block();
    if(threadIdx.x == 0) {
      subproblem_idx = blockIdx.x;
      MemoryConfig& mem_config = grid_data.mem_config;
      bt::pool_allocator shared_mem_pool(mem_config.make_shared_pool(shared_mem));
      fp_engine = bt::allocate_shared<FPEngine, bt::pool_allocator>(shared_mem_pool);
      has_changed = bt::allocate_shared<AtomicBool, bt::pool_allocator>(shared_mem_pool, true);
      stop = bt::allocate_shared<AtomicBool, bt::pool_allocator>(shared_mem_pool, false);
      root = bt::make_shared<BlockCP, bt::global_allocator>(typename BlockCP::tag_gpu_block_copy{},
        (mem_config.mem_kind != MemoryKind::TCN_SHARED),
        *(grid_data.blocks_root),
        bt::global_allocator{},
        mem_config.make_prop_pool(shared_mem_pool),
        mem_config.make_store_pool(shared_mem_pool));
      snapshot_root = bt::make_shared<snapshot_type, bt::global_allocator>(root->search_tree->template snapshot<bt::global_allocator>());
    }
    block.sync();
    fp_engine->init(root->iprop->num_deductions());
    block.sync();
  }

  __device__ void deallocate_shared() {
    if(threadIdx.x == 0) {
      fp_engine.reset();
      has_changed.reset();
      stop.reset();
      root->deallocate();
      snapshot_root.reset();
    }
    __syncthreads();
  }

  __device__ void restore() {
    if(threadIdx.x == 0) {
      root->search_tree->restore(*snapshot_root);
    }
    __syncthreads();
  }
};

template <class S>
__global__ void initialize_grid_data(GridData<S>* grid_data) {
  grid_data->allocate();
  size_t num_subproblems = 1;
  num_subproblems <<= grid_data->root.config.subproblems_power;
  grid_data->next_subproblem->meet(ZLB<size_t, bt::local_memory>(grid_data->root.stats.num_blocks));
  grid_data->root.stats.eps_num_subproblems = num_subproblems;
}

template <class S>
__global__ void deallocate_grid_data(GridData<S>* grid_data) {
  grid_data->deallocate();
}

/** We update the bound found by the current block so it is visible to all other blocks.
 * Note that this operation might not always succeed, which is okay, the best bound is still preserved in `block_data` and then reduced at the end (in `reduce_blocks`).
 * The worst that can happen is that a best bound is found twice, which does not prevent the correctness of the algorithm.
 */
template <class S>
__device__ bool update_grid_best_bound(BlockData<S>& block_data, GridData<S>& grid_data) {
  using U0 = typename S::U0;
  assert(threadIdx.x == 0);
  if(block_data.root->bab->is_optimization()) {
    const auto& bab = block_data.root->bab;
    auto local_best = bab->optimum().project(bab->objective_var());
    // printf("[new bound] %d: [%d..%d] (current best: [%d..%d])\n", blockIdx.x, local_best.lb().value(), local_best.ub().value(), grid_data.best_bound->lb().value(), grid_data.best_bound->ub().value());
    if(bab->is_maximization()) {
      return grid_data.best_bound->meet_lb(dual_bound<typename U0::LB>(local_best.ub()));
    }
    else {
      return grid_data.best_bound->meet_ub(dual_bound<typename U0::UB>(local_best.lb()));
    }
  }
  return false;
}

/** This function updates the best bound of the current block according to the best bound found so far across blocks.
 * We directly update the store with the best bound.
 * This function should be called in each node, since the best bound is erased on backtracking (it is not included in the snapshot).
 */
template <class S>
__device__ void update_block_best_bound(BlockData<S>& block_data, GridData<S>& grid_data) {
  using U0 = typename S::U0;
  if(threadIdx.x == 0 && block_data.root->bab->is_optimization()) {
    const auto& bab = block_data.root->bab;
    VarEnv<bt::global_allocator> empty_env{};
    auto best_formula = bab->template deinterpret_best_bound<bt::global_allocator>(
      bab->is_maximization()
      ? U0(dual_bound<typename U0::UB>(grid_data.best_bound->lb()))
      : U0(dual_bound<typename U0::LB>(grid_data.best_bound->ub())));
    // printf("global best: "); grid_data.best_bound->ub().print(); printf("\n");
    // best_formula.print(); printf("\n");
    IDiagnostics diagnostics;
    interpret_and_tell(best_formula, empty_env, *block_data.root->store, diagnostics);
  }
}

/** Propagate a node of the search tree and process the leaf nodes (failed or solution).
 * Branching on unknown nodes is a task left to the caller.
 */
template <class S>
__device__ bool propagate(BlockData<S>& block_data, GridData<S>& grid_data) {
  using BlockCP = typename S::BlockCP;
  __shared__ int warp_iterations[CUDA_THREADS_PER_BLOCK/32];
  warp_iterations[threadIdx.x / 32] = 0;
  bool is_leaf_node = false;
  BlockCP& cp = *block_data.root;
  auto group = cooperative_groups::this_thread_block();
  auto& fp_engine = *block_data.fp_engine;
  auto& iprop = *cp.iprop;
  auto start = cp.stats.start_timer_device();
  int fp_iterations;
  switch(cp.config.fixpoint) {
    case FixpointKind::AC1: {
      fp_iterations = fp_engine.fixpoint(
        [&](int i){ return iprop.deduce(i); },
        [&](){ return iprop.is_bot(); });
      if(threadIdx.x == 0) {
        cp.stats.num_deductions += fp_iterations * fp_engine.num_active();
      }
      break;
    }
    case FixpointKind::WAC1: {
      if(fp_engine.num_active() <= cp.config.wac1_threshold) {
        fp_iterations = fp_engine.fixpoint(
          [&](int i){ return iprop.deduce(i); },
          [&](){ return iprop.is_bot(); });
        if(threadIdx.x == 0) {
          cp.stats.num_deductions += fp_iterations * fp_engine.num_active();
        }
      }
      else {
        fp_iterations = fp_engine.fixpoint(
          [&](int i){ return warp_fixpoint<CUDA_THREADS_PER_BLOCK>(iprop, i, warp_iterations); },
          [&](){ return iprop.is_bot(); });
        if(threadIdx.x == 0) {
          for(int i = 0; i < CUDA_THREADS_PER_BLOCK/32; ++i) {
            cp.stats.num_deductions += warp_iterations[i] * 32;
          }
        }
      }
      break;
    }
  }

  start = cp.stats.stop_timer(Timer::FIXPOINT, start);
  if(!iprop.is_bot()) {
    fp_engine.select([&](int i) { return !iprop.ask(i); });
    start = cp.stats.stop_timer(Timer::SELECT_FP_FUNCTIONS, start);
    if(fp_engine.num_active() == 0) {
      is_leaf_node = cp.store->template is_extractable<AtomicExtraction>(group);
    }
  }
  else {
    is_leaf_node = true;
  }
  if(threadIdx.x == 0) {
    cp.stats.fixpoint_iterations += fp_iterations;
    bool is_pruned = cp.on_node();
    if(iprop.is_bot()) {
      cp.on_failed_node();
    }
    else if(is_leaf_node) { // is_leaf_node is set to true above.
      if(cp.bab->is_satisfaction() || cp.bab->compare_bound(*cp.store, cp.bab->optimum())) {
        cp.bab->deduce();
        bool best_has_changed = update_grid_best_bound(block_data, grid_data);
        if(cp.bab->is_satisfaction() || (best_has_changed && cp.is_printing_intermediate_sol())) {
          grid_data.produce_solution(*cp.bab);
        }
        is_pruned |= cp.update_solution_stats();
      }
    }
    if(is_pruned) {
      grid_data.gpu_stop->join(true);
    }
    cp.stats.stop_timer(Timer::SEARCH, start);
  }
  if(is_leaf_node) {
    fp_engine.reset(cp.iprop->num_deductions());
  }
  return is_leaf_node;
}

/** The initial problem tackled during the dive must always be the same.
 * Hence, don't be tempted to add the objective during diving because it might lead to ignoring some subproblems since the splitting decisions will differ.
 */
template <class S>
__device__ size_t dive(BlockData<S>& block_data, GridData<S>& grid_data) {
  using BlockCP = typename S::BlockCP;
  BlockCP& cp = *block_data.root;
  auto& fp_engine = *block_data.fp_engine;
  auto& stop = *block_data.stop;
  // Note that we use `block_has_changed` to stop the "diving", not really to indicate something has changed or not (since we do not need this information for this algorithm).
  auto& stop_diving = *block_data.has_changed;
  stop.meet(false);
  stop_diving.meet(false);
  fp_engine.barrier();
  size_t remaining_depth = grid_data.root.config.subproblems_power;
  while(remaining_depth > 0 && !stop_diving && !stop) {
    bool is_leaf_node = propagate(block_data, grid_data);
    stop.join(grid_data.cpu_stop.test() || *(grid_data.gpu_stop));
    if(is_leaf_node) {
      if(threadIdx.x == 0) {
        stop_diving.join(true);
      }
    }
    else {
      remaining_depth--;
      if(threadIdx.x == 0) {
        auto start = cp.stats.start_timer_device();
        size_t branch_idx = (block_data.subproblem_idx & (size_t{1} << remaining_depth)) >> remaining_depth;
        auto branches = cp.split->split();
        assert(branches.size() == 2);
        cp.iprop->deduce(branches[branch_idx]);
        cp.stats.stop_timer(Timer::SEARCH, start);
      }
    }
    fp_engine.barrier();
  }
  return remaining_depth;
}

template <class S>
__device__ void solve_problem(BlockData<S>& block_data, GridData<S>& grid_data) {
  using BlockCP = typename S::BlockCP;
  BlockCP& cp = *block_data.root;
  auto& fp_engine = *block_data.fp_engine;
  auto& block_has_changed = *block_data.has_changed;
  auto& stop = *block_data.stop;
  block_has_changed.join(true);
  stop.meet(false);
  fp_engine.barrier();
  auto start = cp.stats.start_timer_device();
  // In the condition, we must only read variables that are local to this block.
  // Otherwise, two threads might read different values if it is changed in between by another block.
  while(block_has_changed && !stop) {
    update_block_best_bound(block_data, grid_data);
    cp.stats.stop_timer(Timer::SEARCH, start);
    propagate(block_data, grid_data);
    if(threadIdx.x == 0) {
      start = cp.stats.start_timer_device();
      stop.join(grid_data.cpu_stop.test() || *(grid_data.gpu_stop));
      // propagate induces a memory fence, therefore all threads are already past the "while" condition.
      // auto t = cp.stats.start_timer_device();
      block_has_changed.meet(cp.search_tree->deduce());
      // cp.stats.stop_timer(Timer::ST_DEDUCE, t);
      //  auto t = cp.stats.start_timer_device();
      //  auto s = cp.search_tree->split->split();
      //  t = cp.stats.stop_timer(Timer::SPLIT, t);
      //  auto P = cp.search_tree->push(std::move(s));
      //  t = cp.stats.stop_timer(Timer::PUSH, t);
      //  block_has_changed.meet(cp.search_tree->pop(std::move(P)));
      //  cp.stats.stop_timer(Timer::POP, t);
    }
    fp_engine.barrier();
  }
  cp.stats.stop_timer(Timer::SEARCH, start);
}

template <class S>
CUDA void reduce_blocks(GridData<S>* grid_data) {
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    if(grid_data->blocks[i].root) { // `nullptr` could happen if we try to terminate the program before all blocks are even created.
      grid_data->root.meet(*(grid_data->blocks[i].root));
    }
  }
}

template <class S>
__global__ void gpu_solve_kernel(GridData<S>* grid_data)
{
  if(threadIdx.x == 0 && blockIdx.x == 0 && grid_data->root.config.verbose_solving) {
    printf("%% GPU kernel started, starting solving...\n");
  }
  extern __shared__ unsigned char shared_mem[];
  size_t num_subproblems = grid_data->root.stats.eps_num_subproblems;
  BlockData<S>& block_data = grid_data->blocks[blockIdx.x];
  block_data.allocate(*grid_data, shared_mem);
  auto solve_start = block_data.root->stats.start_timer_device();
  while(block_data.subproblem_idx < num_subproblems && !*(block_data.stop)) {
    if(threadIdx.x == 0 && grid_data->root.config.verbose_solving >= 2) {
      grid_data->print_lock->acquire();
      printf("%% Block %d solves subproblem num %" PRIu64 "\n", blockIdx.x, block_data.subproblem_idx);
      grid_data->print_lock->release();
    }
    block_data.restore();
    __syncthreads();
    auto dive_start = block_data.root->stats.start_timer_device();
    size_t remaining_depth = dive(block_data, *grid_data);
    block_data.root->stats.stop_timer(Timer::DIVE, dive_start);
    if(remaining_depth == 0) {
      solve_problem(block_data, *grid_data);
      if(threadIdx.x == 0 && !*(block_data.stop)) {
        block_data.root->stats.eps_solved_subproblems += 1;
      }
    }
    else {
      if(threadIdx.x == 0 && !*(block_data.stop)) {
        size_t next_subproblem_idx = ((block_data.subproblem_idx >> remaining_depth) + size_t{1}) << remaining_depth;
        grid_data->next_subproblem->meet(ZLB<size_t, bt::local_memory>(next_subproblem_idx));
        // It is possible that several blocks skip similar subproblems. Hence, we only count the subproblems skipped by the block solving the left most subproblem.
        if((block_data.subproblem_idx & ((size_t{1} << remaining_depth) - size_t{1})) == size_t{0}) {
          block_data.root->stats.eps_skipped_subproblems += next_subproblem_idx - block_data.subproblem_idx;
        }
      }
    }
    // Load next problem.
    if(threadIdx.x == 0 && !*(block_data.stop)) {
      block_data.subproblem_idx = grid_data->next_subproblem->value();
      grid_data->next_subproblem->meet(ZLB<size_t, bt::local_memory>(block_data.subproblem_idx + size_t{1}));
    }
    __syncthreads();
  }
  __syncthreads();
  if(threadIdx.x == 0 && !*(block_data.stop)) {
    block_data.root->stats.num_blocks_done = 1;
  }
  if(threadIdx.x == 0) {
    grid_data->print_lock->acquire();
    if(!grid_data->blocks_reduced) {
      int n = 0;
      for(int i = 0; i < grid_data->blocks.size(); ++i) {
        if(grid_data->blocks[i].root) { // `nullptr` could happen if we try to terminate the program before all blocks are even created.
          n += grid_data->blocks[i].root->stats.num_blocks_done;
        }
      }
      if(block_data.stop->value() || n == grid_data->blocks.size()) {
        reduce_blocks(grid_data);
        grid_data->blocks_reduced = true;
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
        grid_data->ready_to_consume.test_and_set(cuda::std::memory_order_seq_cst);
        grid_data->ready_to_consume.notify_one();
      }
    }
    grid_data->print_lock->release();
  }
  // We must destroy all objects allocated in the shared memory, trying to destroy them anywhere else will lead to segfault.
  block_data.deallocate_shared();
}

template <class S, class U>
size_t sizeof_store(const CP<U>& root) {
  return gpu_sizeof<typename S::BlockCP::IStore>()
       + gpu_sizeof<typename S::BlockCP::IStore::universe_type>() * root.store->vars();
}

/** \returns the size of the shared memory and the kind of memory used. */
template <class S, class U>
MemoryConfig configure_memory(CP<U>& root) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  const auto& config = root.config;
  size_t shared_mem_capacity = deviceProp.sharedMemPerBlock;

  // Copy the root to know how large are the abstract domains.
  CP<U> root2(root);

  size_t store_alignment = 200; // The store does not create much alignment overhead since it is almost only a single array.

  MemoryConfig mem_config;
  // Need a bit of shared memory for the fixpoint engine.
  mem_config.shared_bytes = sizeof(FPEngine)+100;
  mem_config.store_bytes = sizeof_store<S>(root2) + store_alignment;
  // We add 20% extra memory due to the alignment of the shared memory which is not taken into account in the statistics.
  // From limited experiments, alignment overhead is usually around 10%.
  mem_config.prop_bytes = root2.prop_allocator.total_bytes_allocated();
  mem_config.prop_bytes += mem_config.prop_bytes / 5;
  if(config.only_global_memory || shared_mem_capacity < mem_config.shared_bytes + mem_config.store_bytes) {
    if(!config.only_global_memory && config.verbose_solving) {
      printf("%% The store of variables (%zuKB) cannot be stored in the shared memory of the GPU (%zuKB), therefore we use the global memory.\n",
      mem_config.store_bytes / 1000,
      shared_mem_capacity / 1000);
    }
    mem_config.mem_kind = MemoryKind::GLOBAL;
  }
  else if(shared_mem_capacity > mem_config.shared_bytes + mem_config.store_bytes + mem_config.prop_bytes) {
    if(config.verbose_solving) {
      printf("%% The store of variables and the propagators (%zuKB) are stored in the shared memory of the GPU (%zuKB).\n",
      (mem_config.shared_bytes + mem_config.store_bytes + mem_config.prop_bytes) / 1000,
      shared_mem_capacity / 1000);
    }
    mem_config.shared_bytes += mem_config.store_bytes + mem_config.prop_bytes;
    mem_config.mem_kind = MemoryKind::TCN_SHARED;
  }
  else {
    if(config.verbose_solving) {
      printf("%% The store of variables (%zuKB) is stored in the shared memory of the GPU (%zuKB).\n",
        mem_config.store_bytes / 1000,
        shared_mem_capacity / 1000);
    }
    mem_config.shared_bytes += mem_config.store_bytes;
    mem_config.mem_kind = MemoryKind::STORE_SHARED;
  }
  root.stats.print_memory_statistics(config.verbose_solving, "store_memory_real", root2.store_allocator.total_bytes_allocated());
  root.stats.print_memory_statistics(config.verbose_solving, "prop_memory_real", root2.prop_allocator.total_bytes_allocated());
  root.stats.print_memory_statistics(config.verbose_solving, "other_memory_real", root2.basic_allocator.total_bytes_allocated());
  return mem_config;
}

template <class S>
void consume_kernel_solutions(GridData<S>& grid_data) {
  while(!grid_data.consume_solution()) {}
}

template <class S, class U, class Timepoint>
void transfer_memory_and_run(CP<U>& root, MemoryConfig mem_config, const Timepoint& start) {
  using concurrent_allocator = typename S::concurrent_allocator;
  auto grid_data = bt::make_shared<GridData<S>, concurrent_allocator>(root, mem_config);
  initialize_grid_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
  if(grid_data->root.config.print_statistics) {
    mem_config.print_mzn_statistics(root.config, root.stats);
  }
  std::thread consumer_thread(consume_kernel_solutions<S>, std::ref(*grid_data));
  gpu_solve_kernel
    <<<static_cast<unsigned int>(grid_data->root.stats.num_blocks),
      CUDA_THREADS_PER_BLOCK,
      grid_data->mem_config.shared_bytes>>>
    (grid_data.get());
  bool interrupted = wait_solving_ends(grid_data->cpu_stop, grid_data->root, start);
  consumer_thread.join();
  CUDAEX(cudaDeviceSynchronize());
  deallocate_grid_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
}

// From https://stackoverflow.com/a/32531982/2231159
int threads_per_sm(cudaDeviceProp devProp) {
  switch (devProp.major){
    case 2: return (devProp.minor == 1) ? 48 : 32; // Fermi
    case 3: return 192; // Kepler
    case 5: return 128; // Maxwell
    case 6: return (devProp.minor == 0) ? 64 : 128; // Pascal
    case 7: return 64; // Volta and Turing
    case 8: return (devProp.minor == 0) ? 64 : 128; // Ampere
    case 9: return 128; // Hopper
    default: return 64;
  }
}

template <class S, class U>
void configure_blocks_threads(CP<U>& root, const MemoryConfig& mem_config) {
  int max_block_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_block_per_sm, (void*) gpu_solve_kernel<S>, CUDA_THREADS_PER_BLOCK, (int)mem_config.shared_bytes);
  if(root.config.verbose_solving) {
    printf("%% max_blocks_per_sm=%d\n", max_block_per_sm);
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  size_t total_global_mem = deviceProp.totalGlobalMem;
  size_t num_sm = deviceProp.multiProcessorCount;
  size_t num_threads_per_sm = threads_per_sm(deviceProp);

  auto& config = root.config;
  root.stats.num_blocks = (config.or_nodes == 0) ? max_block_per_sm : config.or_nodes;

  config.stack_kb = config.stack_kb == 0 ? 32 : config.stack_kb;

  // The stack allocated depends on the maximum number of threads per SM, not on the actual number of threads per block.
  size_t total_stack_size = num_sm * deviceProp.maxThreadsPerMultiProcessor * config.stack_kb * 1000;
  size_t remaining_global_mem = total_global_mem - total_stack_size;
  remaining_global_mem -= remaining_global_mem / 10; // We leave 10% of global memory free for CUDA allocations, not sure if it is useful though.

  // Basically the size of the store and propagator, and 100 bytes per variable.
  // +1 for the root node in GridCP.
  size_t heap_usage_estimation = (root.stats.num_blocks + 1) * (mem_config.prop_bytes + mem_config.store_bytes + 100 * root.store->vars());
  while(heap_usage_estimation > remaining_global_mem) {
    root.stats.num_blocks--;
  }

  // The stack always need to be modified for this algorithm due to recursive function calls.
  CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, config.stack_kb*1000));
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, remaining_global_mem/15));

  root.stats.print_memory_statistics(config.verbose_solving, "stack_memory", total_stack_size);
  root.stats.print_memory_statistics(config.verbose_solving, "heap_memory", remaining_global_mem);
  root.stats.print_memory_statistics(config.verbose_solving, "heap_usage_estimation", heap_usage_estimation);
  if(config.verbose_solving) {
    printf("%% num_blocks=%d\n", root.stats.num_blocks);
  }
}

template <class S, class U, class Timepoint>
void configure_and_run(CP<U>& root, const Timepoint& start) {
  MemoryConfig mem_config = configure_memory<S>(root);
  configure_blocks_threads<S>(root, mem_config);
  transfer_memory_and_run<S>(root, mem_config, start);
}

#endif // __CUDACC__

void gpu_dive_and_solve(Configuration<bt::standard_allocator>& config) {
#ifndef __CUDACC__
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
#else
  check_support_managed_memory();
  check_support_concurrent_managed_memory();
  auto start = std::chrono::steady_clock::now();
  CP<Itv> root(config);
  root.preprocess();
  if(root.iprop->is_bot()) {
    root.print_final_solution();
    return;
  }
  block_signal_ctrlc();
#ifdef NO_CONCURRENT_MANAGED_MEMORY
  configure_and_run<ItvSolverPinned>(root, start);
#else
  configure_and_run<ItvSolver>(root, start);
#endif
#endif
}

#endif // TURBO_GPU_DIVE_AND_SOLVE_HPP
