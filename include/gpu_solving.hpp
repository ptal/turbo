// Copyright 2023 Pierre Talbot

#ifndef TURBO_GPU_SOLVING_HPP
#define TURBO_GPU_SOLVING_HPP

#include "common_solving.hpp"
#include <thread>
#include <algorithm>

#ifdef __NVCC__

#include <cuda/semaphore>

using F = TFormula<managed_allocator>;
using FormulaPtr = battery::shared_ptr<F, managed_allocator>;

/** We first interpret the formula in an abstract domain with sequential managed memory, that we call `GridCP`. */
using Itv0 = Interval<ZInc<int, battery::local_memory>>;
using GridCP = AbstractDomains<Itv0,
  statistics_allocator<managed_allocator>,
  statistics_allocator<UniqueLightAlloc<managed_allocator, 0>>,
  statistics_allocator<UniqueLightAlloc<managed_allocator, 1>>>;

/** Then, once everything is initialized, we rely on a parallel abstract domain called `A1`, using atomic global memory. */
using Itv1 = Interval<ZInc<int, atomic_memory_block>>;
using AtomicBInc = BInc<atomic_memory_block>;
using FPEngine = BlockAsynchronousIterationGPU<pool_allocator>;

using BlockCP = AbstractDomains<Itv1,
  global_allocator,
  pool_allocator,
  UniqueAlloc<pool_allocator, 0>>;

enum class MemoryKind {
  GLOBAL,
  STORE_SHARED,
  STORE_PC_SHARED
};

struct MemoryConfig {
  MemoryKind mem_kind;
  size_t shared_bytes;
  size_t store_bytes;
  size_t pc_bytes;

  CUDA pool_allocator make_global_pool(size_t bytes) {
    void* mem_pool = global_allocator{}.allocate(bytes);
    return pool_allocator(static_cast<unsigned char*>(mem_pool), bytes);
  }

  CUDA pool_allocator make_shared_pool(unsigned char* shared_mem) {
    return pool_allocator(shared_mem, shared_bytes);
  }

  CUDA pool_allocator make_pc_pool(pool_allocator shared_mem) {
    if(mem_kind == MemoryKind::STORE_PC_SHARED) {
      return shared_mem;
    }
    else {
      return make_global_pool(pc_bytes);
    }
  }

  CUDA pool_allocator make_store_pool(pool_allocator shared_mem) {
    if(mem_kind == MemoryKind::STORE_PC_SHARED || mem_kind == MemoryKind::STORE_SHARED) {
      return shared_mem;
    }
    else {
      return make_global_pool(store_bytes);
    }
  }
};

struct BlockData;

struct GridData {
  GridCP root;
  // Stop from the CPU, for instance because of a timeout.
  bool cpu_stop;
  MemoryConfig mem_config;
  vector<BlockData, global_allocator> blocks;
  // Stop from a block on the GPU, for instance because we found a solution.
  shared_ptr<BInc<atomic_memory_grid>, global_allocator> gpu_stop;
  shared_ptr<cuda::binary_semaphore<cuda::thread_scope_device>, global_allocator> print_lock;
  shared_ptr<ZInc<size_t, atomic_memory_grid>, global_allocator> next_subproblem;

  GridData(const GridCP& root, const MemoryConfig& mem_config)
    : root(root)
    , mem_config(mem_config)
    , cpu_stop(false)
  {}

  __device__ void allocate() {
    assert(threadIdx.x == 0 && blockIdx.x == 0);
    blocks = vector<BlockData, global_allocator>(root.config.or_nodes);
    gpu_stop = make_shared<BInc<atomic_memory_grid>, global_allocator>(false);
    print_lock = make_shared<cuda::binary_semaphore<cuda::thread_scope_device>, global_allocator>(1);
    next_subproblem = make_shared<ZInc<size_t, atomic_memory_grid>, global_allocator>(0);
  }

  __device__ void deallocate() {
    assert(threadIdx.x == 0 && blockIdx.x == 0);
    blocks = vector<BlockData, global_allocator>();
    gpu_stop.reset();
    print_lock.reset();
    next_subproblem.reset();
  }
};

struct BlockData {
  using snapshot_type = typename BlockCP::IST::snapshot_type<global_allocator>;
  size_t subproblem_idx;
  shared_ptr<FPEngine, global_allocator> fp_engine;
  shared_ptr<AtomicBInc, pool_allocator> has_changed;
  shared_ptr<AtomicBInc, pool_allocator> stop;
  shared_ptr<BlockCP, global_allocator> root;
  shared_ptr<snapshot_type, global_allocator> snapshot_root;

  __device__ BlockData():
    has_changed(nullptr, pool_allocator(nullptr, 0)),
    stop(nullptr, pool_allocator(nullptr, 0))
  {}

public:

  /** Initialize the block data.
  * Allocate the abstract domains in the best memory available depending on how large are the abstract domains. */
  __device__ void allocate(GridData& grid_data, unsigned char* shared_mem) {
    auto block = cooperative_groups::this_thread_block();
    if(threadIdx.x == 0) {
      subproblem_idx = blockIdx.x;
      MemoryConfig& mem_config = grid_data.mem_config;
      pool_allocator shared_mem_pool(mem_config.make_shared_pool(shared_mem));
      fp_engine = make_shared<FPEngine, global_allocator>(block, shared_mem_pool);
      has_changed = allocate_shared<AtomicBInc, pool_allocator>(shared_mem_pool, true);
      stop = allocate_shared<AtomicBInc, pool_allocator>(shared_mem_pool, false);
      pool_allocator pc_pool{mem_config.make_pc_pool(shared_mem_pool)};
      root = make_shared<BlockCP, global_allocator>(grid_data.root, global_allocator{}, pc_pool, mem_config.make_store_pool(shared_mem_pool));
      snapshot_root = make_shared<snapshot_type, global_allocator>(root->search_tree->template snapshot<global_allocator>());
    }
    block.sync();
  }

  __device__ void deallocate() {
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
    }
    cooperative_groups::this_thread_block().sync();
  }
};

__global__ void initialize_grid_data(GridData* grid_data) {
  grid_data->allocate();
}

__global__ void deallocate_grid_data(GridData* grid_data) {
  grid_data->deallocate();
}

/** We update the bound found by the current block so it is visible to all other blocks.
 * Note that this operation might not always succeed, which is okay, the best bound is still preserved in `block_data` at then reduced at the end (in `reduce_blocks`).
 * The worst that can happen is that a best bound is found twice, which does not prevent the correctness of the algorithm.
 */
__device__ void update_grid_best_bound(BlockData& block_data, GridData& grid_data) {
  if(threadIdx.x == 0 && !grid_data.root.bab->objective_var().is_untyped()) {
    VarEnv<global_allocator> empty_env{};
    auto best_formula = block_data.root->bab->template deinterpret_best_bound<global_allocator>();
    auto best_tell = grid_data.root.store->interpret_tell_in(best_formula, empty_env);
    grid_data.root.store->tell(best_tell.value());
  }
}

__device__ void update_block_best_bound(BlockData& block_data, GridData& grid_data) {
  if(threadIdx.x == 0 && !grid_data.root.bab->objective_var().is_untyped()) {
    VarEnv<global_allocator> empty_env{};
    auto objvar = grid_data.root.bab->objective_var();
    block_data.root->store->tell(objvar, grid_data.root.store->project(objvar));
    auto best_formula = block_data.root->bab->template deinterpret_best_bound<global_allocator>();
    auto best_tell = block_data.root->store->interpret_tell_in(best_formula, empty_env);
    block_data.root->store->tell(best_tell.value());
  }
}

__device__ size_t dive(BlockData& block_data, GridData& grid_data) {
  BlockCP& a = *block_data.root;
  auto& fp_engine = *block_data.fp_engine;
  auto& stop = *block_data.stop;
  // Note that we use `block_has_changed` to stop the "diving", not really to indicate something has changed or not (since we do not need this information for this algorithm).
  auto& stop_diving = *block_data.has_changed;
  stop.dtell_bot();
  stop_diving.dtell_bot();
  fp_engine.barrier();
  size_t depth = grid_data.root.config.subproblems_power + 1;
  while(depth > 0 && !stop_diving && !stop) {
    depth--;
    local::BInc thread_has_changed;
    fp_engine.fixpoint(*a.ipc, thread_has_changed);
    if(threadIdx.x == 0) {
      a.on_node();
      if(a.ipc->is_top()) {
        a.on_failed_node();
        stop_diving.tell_top();
      }
      else if(a.bab->template refine<AtomicExtraction>(thread_has_changed)) {
        grid_data.print_lock->acquire();
        bool do_not_stop = a.on_solution_node();
        grid_data.print_lock->release();
        if(!do_not_stop) {
          grid_data.gpu_stop->tell_top();
        }
        stop_diving.tell_top();
        update_grid_best_bound(block_data, grid_data);
          }
      else {
        size_t branch_idx = (block_data.subproblem_idx & (1 << depth)) >> depth;
        auto branches = a.split->split();
        assert(branches.size() == 2);
        a.ipc->tell(branches[branch_idx]);
      }
      stop.tell(local::BInc(grid_data.cpu_stop || *(grid_data.gpu_stop)));
    }
    fp_engine.barrier();
  }
  return depth;
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
    // For correctness we need this local variable, we cannot use `block_has_changed`.
    local::BInc thread_has_changed;
    fp_engine.fixpoint(*cp.ipc, thread_has_changed);
    block_has_changed.dtell_bot();
    fp_engine.barrier();
    if(threadIdx.x == 0) {
      cp.on_node();
      if(cp.ipc->is_top()) {
        cp.on_failed_node();
        update_block_best_bound(block_data, grid_data);
      }
      else if(cp.bab->template refine<AtomicExtraction>(thread_has_changed)) {
        grid_data.print_lock->acquire();
        bool do_not_stop = cp.on_solution_node();
        grid_data.print_lock->release();
        if(!do_not_stop) {
          grid_data.gpu_stop->tell_top();
        }
        update_grid_best_bound(block_data, grid_data);
          }
      stop.tell(local::BInc(grid_data.cpu_stop || *(grid_data.gpu_stop)));
      if(!stop) {
        cp.search_tree->refine(thread_has_changed);
      }
    }
    block_has_changed.tell(thread_has_changed);
    fp_engine.barrier();
  }
  fp_engine.barrier();
}

__global__ void gpu_solve_kernel(GridData* grid_data)
{
  extern __shared__ unsigned char shared_mem[];
  BlockData& block_data = grid_data->blocks[blockIdx.x];
  block_data.allocate(*grid_data, shared_mem);
  size_t num_subproblems = 1;
  num_subproblems <<= grid_data->root.config.subproblems_power;
  if(threadIdx.x == 0 && blockIdx.x == 0) {
    grid_data->next_subproblem->tell(ZInc<size_t, local_memory>(gridDim.x));
    grid_data->root.stats.eps_num_subproblems = num_subproblems;
  }
  while(block_data.subproblem_idx < num_subproblems && !*(block_data.stop)) {
    // if(threadIdx.x == 0 && grid_data->root.config.verbose_solving) {
    //   printf("%% Block %d solves subproblem num %lu\n", blockIdx.x, block_data.subproblem_idx);
    // }
    block_data.restore();
    update_block_best_bound(block_data, *grid_data);
    cooperative_groups::this_thread_block().sync();
    size_t remaining_depth = dive(block_data, *grid_data);
    if(remaining_depth == 0) {
      solve_problem(block_data, *grid_data);
      if(!*(block_data.stop)) {
        block_data.root->stats.eps_solved_subproblems += 1;
      }
    }
    else {
      if(threadIdx.x == 0 && !*(block_data.stop)) {
        size_t next_subproblem_idx = ((block_data.subproblem_idx >> remaining_depth) + 1) << remaining_depth;
        block_data.root->stats.eps_skipped_subproblems += next_subproblem_idx - block_data.subproblem_idx;
        grid_data->next_subproblem->tell(ZInc<size_t, local_memory>(next_subproblem_idx));
      }
    }
    update_grid_best_bound(block_data, *grid_data);
    // Load next problem.
    if(threadIdx.x == 0 && !*(block_data.stop)) {
      block_data.subproblem_idx = grid_data->next_subproblem->value();
      grid_data->next_subproblem->tell(ZInc<size_t, local_memory>(block_data.subproblem_idx + 1));
    }
    cooperative_groups::this_thread_block().sync();
  }
  // We must destroy all objects allocated in the shared memory, trying to destroy them anywhere else will lead to segfault.
  block_data.deallocate();
}

__global__ void reduce_blocks(GridData* grid_data) {
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    grid_data->root.join(*(grid_data->blocks[i].root));
  }
}

template <class T> __global__ void gpu_sizeof_kernel(size_t* size) { *size = sizeof(T); }
template <class T>
size_t gpu_sizeof() {
  auto s = make_unique<size_t, managed_allocator>();
  gpu_sizeof_kernel<T><<<1, 1>>>(s.get());
  CUDAEX(cudaDeviceSynchronize());
  return *s;
}

size_t sizeof_store(const CP& root) {
  return gpu_sizeof<typename BlockCP::IStore>()
       + gpu_sizeof<typename BlockCP::IStore::universe_type>() * root.store->vars();
}

template <class Alloc>
void increase_memory_limits(const Configuration<Alloc>& config, const MemoryConfig& mem_config) {
  CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, config.stack_kb*1000));
  if(config.verbose_solving) {
    size_t max_stack_size;
    cudaDeviceGetLimit(&max_stack_size, cudaLimitStackSize);
    printf("%% GPU_max_stack_size=%zu (%zuKB)\n", max_stack_size, max_stack_size/1000);
  }
  if(config.heap_mb == 0) {
    // 4 * (size of store + size of propagators).
    size_t per_block_mem = 4 * (mem_config.store_bytes + mem_config.pc_bytes);
    CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, per_block_mem * config.or_nodes));
  }
  else {
    CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t(config.heap_mb) * 1000 * 1000));
  }
  if(config.verbose_solving) {
    size_t max_heap_size;
    cudaDeviceGetLimit(&max_heap_size, cudaLimitMallocHeapSize);
    printf("%% GPU_max_heap_size=%zu (%zuMB)\n", max_heap_size, max_heap_size/1000/1000);
  }
}

// Inspired by https://stackoverflow.com/questions/39513830/launch-cuda-kernel-with-a-timeout/39514902
inline void guard_timeout(int timeout_ms, bool* is_timeout) {
  if(timeout_ms == 0) {
    return;
  }
  int progressed = 0;
  while (!(*is_timeout)) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    progressed += 1000;
    if (progressed >= timeout_ms) {
      *is_timeout = true;
    }
  }
}

/** \returns the size of the shared memory and the kind of memory used. */
MemoryConfig configure_memory(CP& root) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  const auto& config = root.config;
  size_t shared_mem_capacity = deviceProp.sharedMemPerBlock;

  // Copy the root to know how large are the abstract domains.
  CP root2(root);
  root2.print_allocation_statistics();

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
      printf("%% The store of variables (%zu, %zuKB) cannot be stored in the shared memory of the GPU (%zuKB), therefore we use the global memory.\n",
      mem_config.store_bytes,
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
      printf("%% The store of variables (%zu, %zuKB) is stored in the shared memory of the GPU (%zuKB).\n",
        mem_config.store_bytes,
        mem_config.store_bytes / 1000,
        shared_mem_capacity / 1000);
    }
    mem_config.shared_bytes += mem_config.store_bytes;
    mem_config.mem_kind = MemoryKind::STORE_SHARED;
  }
  return mem_config;
}

template <class Timepoint>
void transfer_memory_and_run(CP& root, MemoryConfig mem_config, const Timepoint& start) {
  auto grid_data = make_shared<GridData, managed_allocator>(std::move(root), mem_config);
  initialize_grid_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
  std::thread timeout_thread(guard_timeout, grid_data->root.config.timeout_ms, &grid_data->cpu_stop);
  gpu_solve_kernel
    <<<grid_data->root.config.or_nodes,
       grid_data->root.config.and_nodes,
       grid_data->mem_config.shared_bytes>>>
    (grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
  grid_data->cpu_stop = true;
  check_timeout(grid_data->root, start);
  timeout_thread.join();
  reduce_blocks<<<1, 1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
  grid_data->root.on_finish();
  deallocate_grid_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
}

template <class Timepoint>
void configure_and_run(CP& root, const Timepoint& start) {
  MemoryConfig mem_config = configure_memory(root);
  increase_memory_limits(root.config, mem_config);
  transfer_memory_and_run(root, mem_config, start);
}

#endif // __NVCC__

void gpu_solve(const Configuration<standard_allocator>& config) {
#ifndef __NVCC__
  std::cout << "You must use the NVCC compiler to compile Turbo on GPU." << std::endl;
#else
  auto start = std::chrono::high_resolution_clock::now();
  CP root(config);
  root.prepare_solver();
  configure_and_run(root, start);
#endif
}

#endif // TURBO_GPU_SOLVING_HPP
