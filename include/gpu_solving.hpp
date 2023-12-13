// Copyright 2023 Pierre Talbot

#ifndef TURBO_GPU_SOLVING_HPP
#define TURBO_GPU_SOLVING_HPP

#include "common_solving.hpp"
#include <thread>
#include <algorithm>

namespace bt = ::battery;

#ifdef __CUDACC__

#include <cuda/semaphore>

using F = TFormula<bt::managed_allocator>;
using FormulaPtr = bt::shared_ptr<F, bt::managed_allocator>;

/** We first interpret the formula in an abstract domain with sequential managed memory, that we call `GridCP`. */
using Itv0 = Interval<ZInc<int, bt::local_memory>>;
using GridCP = AbstractDomains<Itv0,
  bt::statistics_allocator<bt::managed_allocator>,
  bt::statistics_allocator<UniqueLightAlloc<bt::managed_allocator, 0>>,
  bt::statistics_allocator<UniqueLightAlloc<bt::managed_allocator, 1>>>;

/** Then, once everything is initialized, we rely on a parallel abstract domain called `BlockCP`, using atomic shared and global memory. */
using Itv1 = Interval<ZInc<int, bt::atomic_memory_block>>;
using Itv2 = Interval<ZInc<int, bt::atomic_memory_grid>>;
using AtomicBInc = BInc<bt::atomic_memory_block>;
using FPEngine = BlockAsynchronousIterationGPU<bt::pool_allocator>;

using BlockCP = AbstractDomains<Itv1,
  bt::global_allocator,
  bt::pool_allocator,
  UniqueAlloc<bt::pool_allocator, 0>>;

/** Depending on the problem, we can store the abstract elements in different memories.
 * The "worst" is everything in global memory (GLOBAL) when the problem is too large for the shared memory.
 * The "best" is when both the store of variables and the propagators (STORE_PC_SHARED) can be stored in shared memory.
 * A third possibility is to store only the variables' domains in the shared memory (STORE_SHARED).
*/
enum class MemoryKind {
  GLOBAL,
  STORE_SHARED,
  STORE_PC_SHARED
};

/** The shared memory must be configured by hand before the kernel is launched.
 * This class encapsulates information about the size of each relevant abstract elements, and help creating the allocators accordingly.
*/
struct MemoryConfig {
  MemoryKind mem_kind;
  size_t shared_bytes;
  size_t store_bytes;
  size_t pc_bytes;

  CUDA bt::pool_allocator make_global_pool(size_t bytes) {
    void* mem_pool = bt::global_allocator{}.allocate(bytes);
    return bt::pool_allocator(static_cast<unsigned char*>(mem_pool), bytes);
  }

  CUDA bt::pool_allocator make_shared_pool(unsigned char* shared_mem) {
    return bt::pool_allocator(shared_mem, shared_bytes);
  }

  CUDA bt::pool_allocator make_pc_pool(bt::pool_allocator shared_mem) {
    if(mem_kind == MemoryKind::STORE_PC_SHARED) {
      return shared_mem;
    }
    else {
      return make_global_pool(pc_bytes);
    }
  }

  CUDA bt::pool_allocator make_store_pool(bt::pool_allocator shared_mem) {
    if(mem_kind == MemoryKind::STORE_PC_SHARED || mem_kind == MemoryKind::STORE_SHARED) {
      return shared_mem;
    }
    else {
      return make_global_pool(store_bytes);
    }
  }

  CUDA void print_mzn_statistics() const {
    printf("%%%%%%mzn-stat: memory_configuration=\"%s\"\n",
      mem_kind == MemoryKind::GLOBAL ? "global" : (
      mem_kind == MemoryKind::STORE_SHARED ? "store_shared" : "store_pc_shared"));
    printf("%%%%%%mzn-stat: shared_mem=%lu\n", shared_bytes);
    printf("%%%%%%mzn-stat: store_mem=%lu\n", store_bytes);
    printf("%%%%%%mzn-stat: propagator_mem=%lu\n", pc_bytes);
  }
};

struct BlockData;

/** `GridData` represents the problem to be solved (`root`), the data of each block when running EPS (`blocks`), the index of the subproblem that needs to be solved next (`next_subproblem`), an estimation of the best bound found across subproblems in case of optimisation problem and other global information.
 */
struct GridData {
  GridCP root;
  // Stop from the CPU, for instance because of a timeout.
  volatile bool cpu_stop;
  volatile bool stat_printed;
  MemoryConfig mem_config;
  bt::vector<BlockData, bt::global_allocator> blocks;
  // Stop from a block on the GPU, for instance because we found a solution.
  bt::shared_ptr<BInc<bt::atomic_memory_grid>, bt::global_allocator> gpu_stop;
  bt::shared_ptr<ZInc<size_t, bt::atomic_memory_grid>, bt::global_allocator> next_subproblem;
  bt::shared_ptr<Itv2, bt::global_allocator> best_bound;

  // All of what follows is only to support printing while the kernel is running.
  // In particular, we transfer the solution to the CPU where it is printed, because printing on the GPU can be very slow when the problem is large.
  bt::shared_ptr<cuda::binary_semaphore<cuda::thread_scope_device>, bt::global_allocator> print_lock;
  cuda::std::atomic_flag ready_to_produce;
  cuda::std::atomic_flag ready_to_consume;

  GridData(const GridCP& root, const MemoryConfig& mem_config)
    : root(root)
    , mem_config(mem_config)
    , cpu_stop(false)
    , stat_printed(false)
  {
    ready_to_consume.clear();
    ready_to_produce.test_and_set();
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
  }

  template <class BlockBAB>
  __device__ void produce_solution(const BlockBAB& bab) {
    print_lock->acquire();
    ready_to_produce.wait(false, cuda::std::memory_order_seq_cst);
    ready_to_produce.clear();
    bab.extract(*(root.bab));
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    ready_to_consume.test_and_set(cuda::std::memory_order_seq_cst);
    ready_to_consume.notify_one();
    print_lock->release();
  }

  __host__ void consume_solution() {
    ready_to_consume.wait(false, cuda::std::memory_order_seq_cst);
    ready_to_consume.clear();
    root.print_solution();
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    ready_to_produce.test_and_set(cuda::std::memory_order_seq_cst);
    ready_to_produce.notify_one();
  }

  __device__ void allocate() {
    assert(threadIdx.x == 0 && blockIdx.x == 0);
    blocks = bt::vector<BlockData, bt::global_allocator>(root.config.or_nodes);
    gpu_stop = bt::make_shared<BInc<bt::atomic_memory_grid>, bt::global_allocator>(false);
    print_lock = bt::make_shared<cuda::binary_semaphore<cuda::thread_scope_device>, bt::global_allocator>(1);
    next_subproblem = bt::make_shared<ZInc<size_t, bt::atomic_memory_grid>, bt::global_allocator>(0);
    best_bound = bt::make_shared<Itv2, bt::global_allocator>();
  }

  __device__ void deallocate() {
    assert(threadIdx.x == 0 && blockIdx.x == 0);
    blocks = bt::vector<BlockData, bt::global_allocator>();
    gpu_stop.reset();
    print_lock.reset();
    next_subproblem.reset();
    best_bound.reset();
  }
};

/** `BlockData` contains all the structures required to solve a subproblem including the problem itself (`root`) and the fixpoint engine (`fp_engine`). */
struct BlockData {
  using snapshot_type = typename BlockCP::IST::snapshot_type<bt::global_allocator>;
  size_t subproblem_idx;
  bt::shared_ptr<FPEngine, bt::global_allocator> fp_engine;
  bt::shared_ptr<AtomicBInc, bt::pool_allocator> has_changed;
  bt::shared_ptr<AtomicBInc, bt::pool_allocator> stop;
  bt::shared_ptr<BlockCP, bt::global_allocator> root;
  bt::shared_ptr<snapshot_type, bt::global_allocator> snapshot_root;

  __device__ BlockData():
    has_changed(nullptr, bt::pool_allocator(nullptr, 0)),
    stop(nullptr, bt::pool_allocator(nullptr, 0))
  {}

public:

  /** Initialize the block data.
  * Allocate the abstract domains in the best memory available depending on how large are the abstract domains. */
  __device__ void allocate(GridData& grid_data, unsigned char* shared_mem) {
    auto block = cooperative_groups::this_thread_block();
    if(threadIdx.x == 0) {
      subproblem_idx = blockIdx.x;
      MemoryConfig& mem_config = grid_data.mem_config;
      bt::pool_allocator shared_mem_pool(mem_config.make_shared_pool(shared_mem));
      fp_engine = bt::make_shared<FPEngine, bt::global_allocator>(block, shared_mem_pool);
      has_changed = bt::allocate_shared<AtomicBInc, bt::pool_allocator>(shared_mem_pool, true);
      stop = bt::allocate_shared<AtomicBInc, bt::pool_allocator>(shared_mem_pool, false);
      bt::pool_allocator pc_pool{mem_config.make_pc_pool(shared_mem_pool)};
      root = bt::make_shared<BlockCP, bt::global_allocator>(grid_data.root, bt::global_allocator{}, pc_pool, mem_config.make_store_pool(shared_mem_pool));
      snapshot_root = bt::make_shared<snapshot_type, bt::global_allocator>(root->search_tree->template snapshot<bt::global_allocator>());
    }
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
    cooperative_groups::this_thread_block().sync();
  }

  __device__ void restore() {
    if(threadIdx.x == 0) {
      root->search_tree->restore(*snapshot_root);
      root->eps_split->reset();
    }
    cooperative_groups::this_thread_block().sync();
  }
};

__global__ void initialize_grid_data(GridData* grid_data) {
  grid_data->allocate();
  size_t num_subproblems = 1;
  num_subproblems <<= grid_data->root.config.subproblems_power;
  grid_data->next_subproblem->tell(ZInc<size_t, bt::local_memory>(grid_data->root.config.or_nodes));
  grid_data->root.stats.eps_num_subproblems = num_subproblems;
}

__global__ void deallocate_grid_data(GridData* grid_data) {
  grid_data->deallocate();
}

/** We update the bound found by the current block so it is visible to all other blocks.
 * Note that this operation might not always succeed, which is okay, the best bound is still preserved in `block_data` and then reduced at the end (in `reduce_blocks`).
 * The worst that can happen is that a best bound is found twice, which does not prevent the correctness of the algorithm.
 */
__device__ void update_grid_best_bound(BlockData& block_data, GridData& grid_data, local::BInc& best_has_changed) {
  if(threadIdx.x == 0 && block_data.root->bab->is_optimization()) {
    const auto& bab = block_data.root->bab;
    auto local_best = bab->optimum().project(bab->objective_var());
    // printf("[new bound] %d: [%d..%d] (current best: [%d..%d])\n", blockIdx.x, local_best.lb().value(), local_best.ub().value(), grid_data.best_bound->lb().value(), grid_data.best_bound->ub().value());
    if(bab->is_maximization()) {
      grid_data.best_bound->tell_lb(dual<typename Itv0::LB>(local_best.ub()), best_has_changed);
    }
    else {
      grid_data.best_bound->tell_ub(dual<typename Itv0::UB>(local_best.lb()), best_has_changed);
    }
  }
}

/** This function updates the best bound of the current block according to the best bound found so far across blocks.
 * We directly update the store with the best bound.
 * This function should be called in each node, since the best bound is erased on backtracking (it is not included in the snapshot).
 */
__device__ void update_block_best_bound(BlockData& block_data, GridData& grid_data) {
  if(threadIdx.x == 0 && block_data.root->bab->is_optimization()) {
    const auto& bab = block_data.root->bab;
    VarEnv<bt::global_allocator> empty_env{};
    auto best_formula = bab->template deinterpret_best_bound<bt::global_allocator>(
      bab->is_maximization()
      ? Itv0(dual<typename Itv0::UB>(grid_data.best_bound->lb()))
      : Itv0(dual<typename Itv0::LB>(grid_data.best_bound->ub())));
    // printf("global best: "); grid_data.best_bound->ub().print(); printf("\n");
    // best_formula.print(); printf("\n");
    IDiagnostics diagnostics;
    interpret_and_tell(best_formula, empty_env, *block_data.root->store, diagnostics);
  }
}

/** Propagate a node of the search tree and process the leaf nodes (failed or solution).
 * Branching on unknown nodes is a task left to the caller.
 */
__device__ bool propagate(BlockData& block_data, GridData& grid_data, local::BInc& thread_has_changed) {
  bool is_leaf_node = false;
  BlockCP& cp = *block_data.root;
  auto& fp_engine = *block_data.fp_engine;
  size_t iterations = fp_engine.fixpoint(*cp.ipc, thread_has_changed);
  if(threadIdx.x == 0) {
    cp.stats.fixpoint_iterations += iterations;
    cp.on_node();
    if(cp.ipc->is_top()) {
      is_leaf_node = true;
      cp.on_failed_node();
    }
    else if(cp.search_tree->template is_extractable<AtomicExtraction>()) {
      is_leaf_node = true;
      if(cp.bab->is_satisfaction() || cp.bab->compare_bound(*cp.store, cp.bab->optimum())) {
        cp.bab->refine(thread_has_changed);
        bool do_not_stop = cp.update_solution_stats();
        if(!do_not_stop) {
          grid_data.gpu_stop->tell_top();
        }
        local::BInc best_has_changed;
        update_grid_best_bound(block_data, grid_data, best_has_changed);
        if(best_has_changed && cp.is_printing_intermediate_sol()) {
          grid_data.produce_solution(*cp.bab);
        }
      }
    }
  }
  return is_leaf_node;
}

/** The initial problem tackled during the dive must always be the same.
 * Hence, don't be tempted to add the objective during diving because it might lead to ignoring some subproblems since the splitting decisions will differ.
 */
__device__ size_t dive(BlockData& block_data, GridData& grid_data) {
  BlockCP& cp = *block_data.root;
  auto& fp_engine = *block_data.fp_engine;
  auto& stop = *block_data.stop;
  // Note that we use `block_has_changed` to stop the "diving", not really to indicate something has changed or not (since we do not need this information for this algorithm).
  auto& stop_diving = *block_data.has_changed;
  stop.dtell_bot();
  stop_diving.dtell_bot();
  fp_engine.barrier();
  size_t remaining_depth = grid_data.root.config.subproblems_power;
  while(remaining_depth > 0 && !stop_diving && !stop) {
    remaining_depth--;
    local::BInc thread_has_changed;
    bool is_leaf_node = propagate(block_data, grid_data, thread_has_changed);
    if(threadIdx.x == 0) {
      if(is_leaf_node) {
        stop_diving.tell_top();
      }
      else {
        size_t branch_idx = (block_data.subproblem_idx & (1 << remaining_depth)) >> remaining_depth;
        auto branches = cp.eps_split->split();
        assert(branches.size() == 2);
        cp.ipc->tell(branches[branch_idx]);
      }
      stop.tell(local::BInc(grid_data.cpu_stop || *(grid_data.gpu_stop)));
    }
    fp_engine.barrier();
  }
  return remaining_depth;
}

__device__ void solve_problem(BlockData& block_data, GridData& grid_data) {
  BlockCP& cp = *block_data.root;
  auto& fp_engine = *block_data.fp_engine;
  auto& block_has_changed = *block_data.has_changed;
  auto& stop = *block_data.stop;
  block_has_changed.tell_top();
  stop.dtell_bot();
  fp_engine.barrier();
  // In the condition, we must only read variables that are local to this block.
  // Otherwise, two threads might read different values if it is changed in between by another block.
  while(block_has_changed && !stop) {
    // For correctness we need this local variable, we cannot use `block_has_changed` (because it might still need to be read by other threads to enter this loop).
    local::BInc thread_has_changed;
    update_block_best_bound(block_data, grid_data);
    propagate(block_data, grid_data, thread_has_changed);
    if(threadIdx.x == 0) {
      stop.tell(local::BInc(grid_data.cpu_stop || *(grid_data.gpu_stop)));
      cp.search_tree->refine(thread_has_changed);
    }
    block_has_changed.dtell_bot();
    fp_engine.barrier();
    block_has_changed.tell(thread_has_changed);
    fp_engine.barrier();
  }
}

CUDA void reduce_blocks(GridData* grid_data) {
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    if(grid_data->blocks[i].root) { // `nullptr` could happen if we try to terminate the program before all blocks are even created.
      grid_data->root.join(*(grid_data->blocks[i].root));
    }
  }
  grid_data->root.print_final_solution();
  if(grid_data->root.config.print_statistics) {
    grid_data->root.print_mzn_statistics();
  }
}

__global__ void gpu_solve_kernel(GridData* grid_data)
{
  if(threadIdx.x == 0 && blockIdx.x == 0 && grid_data->root.config.verbose_solving) {
    printf("%% GPU kernel started, starting solving...\n");
  }
  extern __shared__ unsigned char shared_mem[];
  size_t num_subproblems = grid_data->root.stats.eps_num_subproblems;
  BlockData& block_data = grid_data->blocks[blockIdx.x];
  block_data.allocate(*grid_data, shared_mem);
  while(block_data.subproblem_idx < num_subproblems && !*(block_data.stop)) {
    if(threadIdx.x == 0 && grid_data->root.config.verbose_solving) {
      grid_data->print_lock->acquire();
      printf("%% Block %d solves subproblem num %lu\n", blockIdx.x, block_data.subproblem_idx);
      grid_data->print_lock->release();
    }
    block_data.restore();
    cooperative_groups::this_thread_block().sync();
    size_t remaining_depth = dive(block_data, *grid_data);
    if(remaining_depth == 0) {
      solve_problem(block_data, *grid_data);
      if(threadIdx.x == 0 && !*(block_data.stop)) {
        block_data.root->stats.eps_solved_subproblems += 1;
      }
    }
    else {
      if(threadIdx.x == 0 && !*(block_data.stop)) {
        size_t next_subproblem_idx = ((block_data.subproblem_idx >> remaining_depth) + 1) << remaining_depth;
        grid_data->next_subproblem->tell(ZInc<size_t, bt::local_memory>(next_subproblem_idx));
        // It is possible that several blocks skip similar subproblems. Hence, we only count the subproblems skipped by the block solving the left most subproblem.
        if((block_data.subproblem_idx & ((1 << remaining_depth)-1)) == 0) {
          block_data.root->stats.eps_skipped_subproblems += next_subproblem_idx - block_data.subproblem_idx;
        }
      }
    }
    // Load next problem.
    if(threadIdx.x == 0 && !*(block_data.stop)) {
      block_data.subproblem_idx = grid_data->next_subproblem->value();
      grid_data->next_subproblem->tell(ZInc<size_t, bt::local_memory>(block_data.subproblem_idx + 1));
    }
    cooperative_groups::this_thread_block().sync();
  }
  cooperative_groups::this_thread_block().sync();
  if(threadIdx.x == 0 && !*(block_data.stop)) {
    block_data.root->stats.num_blocks_done = 1;
  }
  if(threadIdx.x == 0) {
    grid_data->print_lock->acquire();
    if(!grid_data->stat_printed) {
      int n = 0;
      for(int i = 0; i < grid_data->blocks.size(); ++i) {
        if(grid_data->blocks[i].root) { // `nullptr` could happen if we try to terminate the program before all blocks are even cretaed.
          n += grid_data->blocks[i].root->stats.num_blocks_done;
        }
      }
      if(block_data.stop->value() || n == grid_data->blocks.size()) {
        reduce_blocks(grid_data);
        grid_data->stat_printed = true;
      }
    }
    grid_data->print_lock->release();
  }
  // We must destroy all objects allocated in the shared memory, trying to destroy them anywhere else will lead to segfault.
  block_data.deallocate_shared();
}

template <class T> __global__ void gpu_sizeof_kernel(size_t* size) { *size = sizeof(T); }
template <class T>
size_t gpu_sizeof() {
  auto s = bt::make_unique<size_t, bt::managed_allocator>();
  gpu_sizeof_kernel<T><<<1, 1>>>(s.get());
  CUDAEX(cudaDeviceSynchronize());
  return *s;
}

size_t sizeof_store(const CP& root) {
  return gpu_sizeof<typename BlockCP::IStore>()
       + gpu_sizeof<typename BlockCP::IStore::universe_type>() * root.store->vars();
}

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

/** \returns the size of the shared memory and the kind of memory used. */
MemoryConfig configure_memory(CP& root) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  const auto& config = root.config;
  size_t shared_mem_capacity = deviceProp.sharedMemPerBlock;

  // Copy the root to know how large are the abstract domains.
  CP root2(root);

  size_t store_alignment = 200; // The store does not create much alignment overhead since it is almost only a single array.

  MemoryConfig mem_config;
  // Need a bit of shared memory for the fixpoint engine.
  mem_config.shared_bytes = 100;
  mem_config.store_bytes = sizeof_store(root2) + store_alignment;
  // We add 20% extra memory due to the alignment of the shared memory which is not taken into account in the statistics.
  // From limited experiments, alignment overhead is usually around 10%.
  mem_config.pc_bytes = root2.prop_allocator.total_bytes_allocated();
  mem_config.pc_bytes += mem_config.pc_bytes / 5;
  if(shared_mem_capacity < mem_config.shared_bytes + mem_config.store_bytes) {
    if(config.verbose_solving) {
      printf("%% The store of variables (%zuKB) cannot be stored in the shared memory of the GPU (%zuKB), therefore we use the global memory.\n",
      mem_config.store_bytes / 1000,
      shared_mem_capacity / 1000);
    }
    mem_config.mem_kind = MemoryKind::GLOBAL;
  }
  else if(shared_mem_capacity > mem_config.shared_bytes + mem_config.store_bytes + mem_config.pc_bytes) {
    if(config.verbose_solving) {
      printf("%% The store of variables and the propagators (%zuKB) are stored in the shared memory of the GPU (%zuKB).\n",
      (mem_config.shared_bytes + mem_config.store_bytes + mem_config.pc_bytes) / 1000,
      shared_mem_capacity / 1000);
    }
    mem_config.shared_bytes += mem_config.store_bytes + mem_config.pc_bytes;
    mem_config.mem_kind = MemoryKind::STORE_PC_SHARED;
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
  if(config.verbose_solving) {
    print_memory_statistics("store_memory_real", root2.store_allocator.total_bytes_allocated());
    print_memory_statistics("pc_memory_real", root2.prop_allocator.total_bytes_allocated());
    print_memory_statistics("other_memory_real", root2.basic_allocator.total_bytes_allocated());
  }
  return mem_config;
}

/** Wait the solving ends because of a timeout, CTRL-C or because the kernel finished. */
template<class Timepoint>
bool wait_solving_ends(GridData& grid_data, const Timepoint& start) {
  cudaEvent_t event;
  cudaEventCreateWithFlags(&event,cudaEventDisableTiming);
  cudaEventRecord(event);
  while(!must_quit() && check_timeout(grid_data.root, start) && cudaEventQuery(event) == cudaErrorNotReady) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if(cudaEventQuery(event) == cudaErrorNotReady) {
    grid_data.cpu_stop = true;
    grid_data.root.stats.exhaustive = false;
    return true;
  }
  else {
    cudaError error = cudaDeviceSynchronize();
    if(error == cudaErrorIllegalAddress) {
      printf("%% ERROR: CUDA kernel failed due to an illegal memory access. This might be due to a stack overflow because it is too small. Try increasing the stack size with the options -stack. If it does not work, please report it as a bug.\n");
      exit(EXIT_FAILURE);
    }
    CUDAEX(error);
    return false;
  }
}

void consume_kernel_solutions(GridData& grid_data) {
  while(!grid_data.cpu_stop) {
    grid_data.consume_solution();
  }
}

template <class Timepoint>
void transfer_memory_and_run(CP& root, MemoryConfig mem_config, const Timepoint& start) {
  auto grid_data = bt::make_shared<GridData, bt::managed_allocator>(std::move(root), mem_config);
  initialize_grid_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
  if(grid_data->root.config.print_statistics) {
    mem_config.print_mzn_statistics();
  }
  std::thread consumer_thread(consume_kernel_solutions, std::ref(*grid_data));
  gpu_solve_kernel
    <<<grid_data->root.config.or_nodes,
      grid_data->root.config.and_nodes,
      grid_data->mem_config.shared_bytes>>>
    (grid_data.get());
  bool interrupted = wait_solving_ends(*grid_data, start);

  // We wait to let the kernel prints the statistics.
  if(grid_data->root.config.kernel_shutdown_timeout_ms != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(grid_data->root.config.kernel_shutdown_timeout_ms));
    exit(EXIT_SUCCESS);
  }

  CUDAEX(cudaDeviceSynchronize());
  if(!interrupted) {
    deallocate_grid_data<<<1,1>>>(grid_data.get());
    CUDAEX(cudaDeviceSynchronize());
  }
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

void configure_blocks_threads(CP& root, const MemoryConfig& mem_config) {
  int hint_num_blocks;
  int hint_num_threads;
  CUDAE(cudaOccupancyMaxPotentialBlockSize(&hint_num_blocks, &hint_num_threads, (void*) gpu_solve_kernel, (int)mem_config.shared_bytes));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  size_t total_global_mem = deviceProp.totalGlobalMem;
  size_t num_sm = deviceProp.multiProcessorCount;
  size_t num_threads_per_sm = threads_per_sm(deviceProp);

  auto& config = root.config;
  config.or_nodes = (config.or_nodes == 0) ? hint_num_blocks : config.or_nodes;
  config.and_nodes = (config.and_nodes == 0) ? hint_num_threads : config.and_nodes;

  if(config.and_nodes > deviceProp.maxThreadsPerBlock) {
    if(config.verbose_solving) {
      printf("%% WARNING: -and %lu too high for this GPU, we use the maximum %d instead.", config.and_nodes, deviceProp.maxThreadsPerBlock);
    }
    config.and_nodes = deviceProp.maxThreadsPerBlock;
  }

  // The stack allocated depends on the maximum number of threads per SM, not on the actual number of threads per block.
  size_t total_stack_size = num_sm * deviceProp.maxThreadsPerMultiProcessor * config.stack_kb * 1000;
  size_t remaining_global_mem = total_global_mem - total_stack_size;
  remaining_global_mem -= remaining_global_mem / 10; // We leave 10% of global memory free for CUDA allocations, not sure if it is useful though.

  // Basically the size of the store and propagator, and 100 bytes per variable.
  // +1 for the root node in GridCP.
  size_t heap_usage_estimation = (config.or_nodes + 1) * (mem_config.pc_bytes + mem_config.store_bytes + 100 * root.store->vars());
  while(heap_usage_estimation > remaining_global_mem) {
    config.or_nodes--;
  }

  CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, config.stack_kb*1000));
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, remaining_global_mem));

  if(config.verbose_solving) {
    print_memory_statistics("stack_memory", total_stack_size);
    print_memory_statistics("heap_memory", remaining_global_mem);
    print_memory_statistics("heap_usage_estimation", heap_usage_estimation);
    printf("%% and_nodes=%lu\n", config.and_nodes);
    printf("%% or_nodes=%lu\n", config.or_nodes);
  }
}

template <class Timepoint>
void configure_and_run(CP& root, const Timepoint& start) {
  MemoryConfig mem_config = configure_memory(root);
  configure_blocks_threads(root, mem_config);
  transfer_memory_and_run(root, mem_config, start);
}

#endif // __CUDACC__

void gpu_solve(Configuration<bt::standard_allocator>& config) {
#ifndef __CUDACC__
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
#else
  auto start = std::chrono::high_resolution_clock::now();
  if(config.kernel_shutdown_timeout_ms != 0) {
    if(config.timeout_ms != 0) {
      if(config.kernel_shutdown_timeout_ms > config.timeout_ms) {
        std::cerr << "The kernel shutdown timeout must be smaller than the timeout." << std::endl;
        exit(EXIT_FAILURE);
      }
      else {
        config.timeout_ms -= config.kernel_shutdown_timeout_ms;
      }
    }
  }
  CP root(config);
  root.preprocess();
  block_signal_ctrlc();
  configure_and_run(root, start);
#endif
}

#endif // TURBO_GPU_SOLVING_HPP
