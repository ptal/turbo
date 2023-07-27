// Copyright 2023 Pierre Talbot

#ifndef TURBO_GPU_SOLVING_HPP
#define TURBO_GPU_SOLVING_HPP

#include "common_solving.hpp"
#include <thread>

#ifdef __NVCC__

#include <cuda/semaphore>

using F = TFormula<managed_allocator>;
using FormulaPtr = battery::shared_ptr<F, managed_allocator>;

/** We first interpret the formula in an abstract domain with sequential managed memory, that we call `A0`. */
using Itv0 = Interval<local::ZInc>;
using A0 = AbstractDomains<Itv0,
  statistics_allocator<managed_allocator>,
  statistics_allocator<UniqueLightAlloc<managed_allocator, 0>>,
  statistics_allocator<UniqueLightAlloc<managed_allocator, 1>>>;

/** Then, once everything is initialized, we rely on a parallel abstract domain called `A1`, using atomic global memory. */
using Itv1 = Interval<ZInc<int, atomic_memory_block>>;
using Itv2 = Interval<ZInc<int, atomic_memory_grid>>;
using AtomicBInc = BInc<atomic_memory_block>;
using FPEngine = BlockAsynchronousIterationGPU<pool_allocator>;

using A1 = AbstractDomains<Itv1,
  global_allocator,
  UniqueLightAlloc<global_allocator, 0>,
  UniqueLightAlloc<global_allocator, 1>>;

using A2 = AbstractDomains<Itv1,
  global_allocator,
  UniqueLightAlloc<global_allocator, 0>,
  pool_allocator>;

using A3 = AbstractDomains<Itv1,
  global_allocator,
  UniqueAlloc<pool_allocator, 0>,
  pool_allocator>;

template <class A>
struct GridData;

template <class A>
struct BlockData {
  using snapshot_type = typename A::IST::snapshot_type<global_allocator>;
  shared_ptr<FPEngine, global_allocator> fp_engine;
  shared_ptr<AtomicBInc, pool_allocator> has_changed;
  shared_ptr<AtomicBInc, pool_allocator> stop;
  shared_ptr<A, global_allocator> abstract_doms;
  shared_ptr<snapshot_type, global_allocator> snapshot_root;
  BlockData() = default;

  /** Initialize the block data.
  * Allocate the abstract domains in the best memory available depending on how large are the abstract domains. */
  __device__ void allocate(GridData<A>& grid_data, unsigned char* shared_mem, size_t shared_mem_capacity) {
    auto block = cooperative_groups::this_thread_block();
    if(threadIdx.x == 0) {
      pool_allocator shared_mem_pool{static_cast<unsigned char*>(shared_mem), shared_mem_capacity};
      fp_engine = make_shared<FPEngine, global_allocator>(block, shared_mem_pool);
      has_changed = allocate_shared<AtomicBInc, pool_allocator>(shared_mem_pool, true);
      stop = allocate_shared<AtomicBInc, pool_allocator>(shared_mem_pool, false);
      if constexpr(std::is_same_v<A, A1>) {
        abstract_doms = make_shared<A1, global_allocator>(grid_data.root);
      }
      else if constexpr(std::is_same_v<A, A2>) {
        abstract_doms = make_shared<A2, global_allocator>(grid_data.root, typename A::basic_allocator_type{}, typename A::prop_allocator_type{}, shared_mem_pool);
      }
      else if constexpr(std::is_same_v<A, A3>) {
        abstract_doms = make_shared<A3, global_allocator>(grid_data.root, typename A::basic_allocator_type{}, typename A::prop_allocator_type{shared_mem_pool}, shared_mem_pool);
      }
      else {
        static_assert(std::is_same_v<A, A1>, "Unknown abstract domains.");
      }
      snapshot_root = make_shared<snapshot_type, global_allocator>(abstract_doms->search_tree->template snapshot<global_allocator>());
    }
    block.sync();
  }

  __device__ void deallocate() {
    if(threadIdx.x == 0) {
      fp_engine.reset();
      has_changed.reset();
      stop.reset();
      abstract_doms->deallocate();
      snapshot_root.reset();
    }
    cooperative_groups::this_thread_block().sync();
  }

  __device__ void restore() {
    if(threadIdx.x == 0) {
      abstract_doms->search_tree->restore(*snapshot_root);
    }
    cooperative_groups::this_thread_block().sync();
  }
};

template <class A>
struct GridData {
  A0 root;
  // Stop from the CPU, for instance because of a timeout.
  bool cpu_stop;
  vector<BlockData<A>, global_allocator> blocks;
  // Stop from a block on the GPU, for instance because we found a solution.
  shared_ptr<BInc<atomic_memory_grid>, global_allocator> gpu_stop;
  shared_ptr<cuda::binary_semaphore<cuda::thread_scope_device>, global_allocator> print_lock;

  GridData(A0&& root)
    : root(std::move(root))
    , cpu_stop(false)
  {}

  __device__ void allocate() {
    assert(threadIdx.x == 0 && blockIdx.x == 0);
    blocks = vector<BlockData<A>, global_allocator>(root.config.or_nodes);
    gpu_stop = make_shared<BInc<atomic_memory_grid>, global_allocator>(false);
    print_lock = make_shared<cuda::binary_semaphore<cuda::thread_scope_device>, global_allocator>(1);
  }

  __device__ void deallocate() {
    assert(threadIdx.x == 0 && blockIdx.x == 0);
    blocks = vector<BlockData<A>, global_allocator>();
    gpu_stop.reset();
    print_lock.reset();
  }
};

template <class A>
__global__ void initialize_grid_data(GridData<A>* grid_data) {
  grid_data->allocate();
}

template <class A>
__global__ void deallocate_grid_data(GridData<A>* grid_data) {
  grid_data->deallocate();
}

template <class A>
__device__ void solve_problem(BlockData<A>& block_data, GridData<A>& grid_data) {
  A& a = *block_data.abstract_doms;
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
    fp_engine.fixpoint(*a.ipc, thread_has_changed);
    block_has_changed.dtell_bot();
    fp_engine.barrier();
    if(threadIdx.x == 0) {
      a.on_node();
      if(a.ipc->is_top()) {
        a.on_failed_node();
      }
      else if(a.bab->template refine<AtomicExtraction>(thread_has_changed)) {
        grid_data.print_lock->acquire();
        bool do_not_stop = a.on_solution_node();
        grid_data.print_lock->release();
        if(!do_not_stop) {
          grid_data.gpu_stop->tell_top();
        }
      }
      stop.tell(local::BInc(grid_data.cpu_stop || *(grid_data.gpu_stop)));
      if(!stop) {
        a.search_tree->refine(thread_has_changed);
      }
    }
    block_has_changed.tell(thread_has_changed);
    fp_engine.barrier();
  }
  fp_engine.barrier();
}

/** We update the bound found by the current block so it is visible to all other blocks.
 * Note that this operation might not always succeed, which is okay, the best bound is still preserved in `block_data` at then reduced at the end (in `reduce_blocks`).
 * The worst that can happen is that a best bound is found twice, which does not prevent the correctness of the algorithm.
 */
template <class A>
__device__ void update_best_bound(BlockData<A>& block_data, GridData<A>& grid_data) {
  if(threadIdx.x == 0) {
    local::BInc has_changed;
    auto objvar = grid_data.root.bab->objective_var();
    const auto& new_bound = block_data.abstract_doms->best->project(objvar);
    grid_data.root.bab->tell(new_bound, has_changed);
  }
  cooperative_groups::this_thread_block().sync();
}

template <class A>
__global__ void gpu_solve_kernel(GridData<A>* grid_data, size_t shared_mem_capacity)
{
  extern __shared__ unsigned char shared_mem[];
  BlockData<A>& block_data = grid_data->blocks[blockIdx.x];
  block_data.allocate(*grid_data, shared_mem, shared_mem_capacity);
  for(int i = 0; i < 5; ++i) {
    block_data.restore();
    solve_problem(block_data, *grid_data);
    update_best_bound(block_data, *grid_data);
  }
  // We must destroy all objects allocated in the shared memory, trying to destroy them anywhere else will lead to segfault.
  block_data.deallocate();
}

template <class A>
__global__ void reduce_blocks(GridData<A>* grid_data) {
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    grid_data->root.join(*(grid_data->blocks[i].abstract_doms));
  }
}

template <class Alloc>
void increase_memory_limits(const Configuration<Alloc>& config, const A0& root) {
  size_t max_stack_size = 1024;
  CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, max_stack_size*100));
  if(config.verbose_solving) {
    cudaDeviceGetLimit(&max_stack_size, cudaLimitStackSize);
    printf("%% GPU_max_stack_size=%zu (%zuKB)\n", max_stack_size, max_stack_size/1000);
  }
  size_t max_heap_size;
  cudaDeviceGetLimit(&max_heap_size, cudaLimitMallocHeapSize);
  size_t estimated_max_heap_size = root.store_allocator.total_bytes_allocated() + (root.prop_allocator.total_bytes_allocated()/10) * config.or_nodes;
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, estimated_max_heap_size));
  if(config.verbose_solving) {
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

template <class Alloc1, class Alloc2>
CUDA void print_allocation_statistics(const char* alloc_name, const statistics_allocator<Alloc1, Alloc2>& alloc) {
  printf("%% %s.total_bytes_allocated=%zu (%zuKB); %zu allocations; %zu deallocations\n",
    alloc_name,
    alloc.total_bytes_allocated(),
    alloc.total_bytes_allocated() / 1000,
    alloc.num_allocations(),
    alloc.num_deallocations());
}

CUDA void print_allocation_statistics(A0& a) {
  print_allocation_statistics("basic_allocator", a.basic_allocator);
  print_allocation_statistics("prop_allocator", a.prop_allocator);
  print_allocation_statistics("store_allocator", a.store_allocator);
}

/** \returns the size of the shared memory and the kind of memory used.
 * 1 for `A1`, 2 for `A2` and 3 for `A3`. */
CUDA tuple<size_t, size_t> configure_memory(A0& root) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  const auto& config = root.config;
  size_t shared_mem_capacity = deviceProp.sharedMemPerBlock;
  increase_memory_limits(config, root);

  // Need a bit of shared memory for the fixpoint engine.
  size_t fixpoint_shared_mem = 100;

  // We add 33% extra memory due to the alignment of the shared memory which is not taken into account in the statistics.
  size_t store_bytes = root.store_allocator.total_bytes_allocated();
  store_bytes += store_bytes / 3;
  size_t prop_bytes = root.prop_allocator.total_bytes_allocated();
  prop_bytes += prop_bytes / 3;
  size_t both_bytes = store_bytes + prop_bytes;

  if(shared_mem_capacity < store_bytes) {
    if(config.verbose_solving) {
      printf("%% The store of variables (%zuKB) cannot be stored in the shared memory of the GPU (%zuKB), therefore we use the global memory.\n",
      store_bytes / 1000,
      shared_mem_capacity / 1000);
    }
    return make_tuple(fixpoint_shared_mem, 1);
  }
  else if(shared_mem_capacity > both_bytes) {
    if(config.verbose_solving) {
      printf("%% The store of variables and the propagators (%zuKB) are stored in the shared memory of the GPU (%zuKB).\n",
      both_bytes / 1000,
      shared_mem_capacity / 1000);
    }
    return make_tuple(both_bytes, 3);
  }
  else {
    if(config.verbose_solving) {
      printf("%% The store of variables (%zuKB) is stored in the shared memory of the GPU (%zuKB).\n",
        store_bytes / 1000,
        shared_mem_capacity / 1000);
    }
    return make_tuple(store_bytes, 2);
  }
}

template <class A, class Timepoint>
void transfer_memory_and_run(A0& root, size_t shared_mem_capacity, const Timepoint& start) {
  auto grid_data = make_shared<GridData<A>, managed_allocator>(std::move(root));
  initialize_grid_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
  std::thread timeout_thread(guard_timeout, grid_data->root.config.timeout_ms, &grid_data->cpu_stop);
  gpu_solve_kernel
    <<<grid_data->root.config.or_nodes,
       grid_data->root.config.and_nodes,
       shared_mem_capacity>>>
    (grid_data.get(), shared_mem_capacity);
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
void configure_and_run(A0& root, const Timepoint& start) {
  tuple<size_t, size_t> memory_config = configure_memory(root);
  size_t shared_mem_capacity = get<0>(memory_config);
  switch(get<1>(memory_config)) {
    case 1:
      transfer_memory_and_run<A1>(root, shared_mem_capacity, start);
      break;
    case 2:
      transfer_memory_and_run<A2>(root, shared_mem_capacity, start);
      break;
    case 3:
      transfer_memory_and_run<A3>(root, shared_mem_capacity, start);
      break;
    default:
      printf("[bug] unknown memory configuration\n");
      assert(false);
  }
}

#endif // __NVCC__

void gpu_solve(const Configuration<standard_allocator>& config) {
#ifndef __NVCC__
  std::cout << "You must use the NVCC compiler to compile Turbo on GPU." << std::endl;
#else
  auto start = std::chrono::high_resolution_clock::now();

  A0::basic_allocator_type basic_allocator{managed_allocator{}};
  A0::prop_allocator_type prop_allocator{UniqueLightAlloc<managed_allocator, 0>{}};
  A0::store_allocator_type store_allocator{UniqueLightAlloc<managed_allocator, 1>{}};
  auto root = make_shared<A0, managed_allocator>(config, basic_allocator, prop_allocator, store_allocator);
  // I. Parse the FlatZinc model.
  battery::shared_ptr<TFormula<A0::basic_allocator_type>, A0::basic_allocator_type> f =
    parse_flatzinc<A0::basic_allocator_type>(root->config.problem_path.data(), root->fzn_output);
  if(!f) {
    std::cerr << "Could not parse FlatZinc model." << std::endl;
    exit(EXIT_FAILURE);
  }
  if(config.verbose_solving) {
    printf("%% FlatZinc parsed\n");
  }

  if(config.print_ast) {
    printf("%% Parsed AST:\n");
    f->print(true);
    printf("\n");
  }

  // I. Create the abstract domains.
  root->allocate(num_quantified_vars(*f));

  // II. Interpret the formula in the abstract domain.
  root->typing(*f);
  if(config.print_ast) {
    printf("%% Typed AST:\n");
    f->print(true);
    printf("\n");
  }
  if(root->interpret(*f)) {
    if(config.print_ast) {
      printf("%% Interpreted AST:\n");
      root->ipc->deinterpret(root->env).print(true);
      printf("\n");
    }
    auto interpretation_time = std::chrono::high_resolution_clock::now();
    root->stats.interpretation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(interpretation_time - start).count();
    if(config.verbose_solving) {
      printf("%% Formula has been loaded, solving begins...\n");
      print_allocation_statistics(*root);
    }
    configure_and_run(*root, interpretation_time);
  }
#endif
}

#endif // TURBO_GPU_SOLVING_HPP
