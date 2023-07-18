// Copyright 2023 Pierre Talbot

#ifndef TURBO_GPU_SOLVING_HPP
#define TURBO_GPU_SOLVING_HPP

#include "common_solving.hpp"
#include <thread>

#ifdef __NVCC__

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
using AtomicBInc = BInc<atomic_memory_block>;
using FPEngine = BlockAsynchronousIterationGPU<global_allocator>;

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

/** We have one abstract element `A1` per GPU block. */
template <class A>
struct BlockData {
  shared_ptr<FPEngine, global_allocator> fp_engine;
  shared_ptr<AtomicBInc, global_allocator> has_changed;
  shared_ptr<A, global_allocator> abstract_doms;
  BlockData() = default;
};

/** The interpretation must be done on the device because some abstract domains use runtime polymorphism and thus rely on vtable.
 * Since vtable cannot migrate between host and device, we must initialize those objects on the device. */
template <class F, class A>
__global__ void initialize_abstract_domains(F* f, A* a, bool* failed)
{
  assert(threadIdx.x == 0 && blockDim.x == 1);
  // I. Create the abstract domains.
  a->allocate(num_quantified_vars(*f));
  // II. Interpret the formula in the abstract domain.
  if(!a->interpret(*f)) {
    *failed = true;
  }
}

/** The members of `A` cannot be deleted on the host size since they were allocated in the global memory in the kernel `initialize_abstract_domains`. */
template <class A>
__global__ void deallocate_abstract_domains(A* a)
{
  a->on_finish();
  a->deallocate();
}

template <class A>
__global__ void gpu_solve_kernel(A0* a0, bool* is_timeout, size_t shared_mem_capacity)
{
  extern __shared__ unsigned char shared_mem[];
  __shared__ BlockData<A>* block_data;
  __shared__ bool stop;
  if(blockIdx.x == 0) {
    if(threadIdx.x == 0) {
      stop = false;
      block_data = new BlockData<A>();
      auto block = cooperative_groups::this_thread_block();
      block_data->fp_engine = make_shared<FPEngine, global_allocator>(block);
      block_data->has_changed = make_shared<AtomicBInc, global_allocator>(true);
      if constexpr(std::is_same_v<A, A1>) {
        block_data->abstract_doms = make_shared<A1, global_allocator>(*a0);
      }
      else if constexpr(std::is_same_v<A, A2>) {
        pool_allocator shared_mem_pool{static_cast<unsigned char*>(shared_mem), shared_mem_capacity};
        block_data->abstract_doms = make_shared<A2, global_allocator>(*a0, typename A::basic_allocator_type{}, typename A::prop_allocator_type{}, shared_mem_pool);
      }
      else if constexpr(std::is_same_v<A, A3>) {
        pool_allocator shared_mem_pool{static_cast<unsigned char*>(shared_mem), shared_mem_capacity};
        block_data->abstract_doms = make_shared<A3, global_allocator>(*a0, typename A::basic_allocator_type{}, typename A::prop_allocator_type{shared_mem_pool}, shared_mem_pool);
      }
      else {
        static_assert(std::is_same_v<A, A1>, "Unknown abstract domains.");
      }
      printf("%%GPU_block_size=%d\n", blockDim.x);
    }
    __syncthreads();
    A& a = *block_data->abstract_doms;
    while(*(block_data->has_changed) && !(*is_timeout) && !stop) {
      local::BInc has_changed;
      block_data->fp_engine->fixpoint(*a.ipc, has_changed);
      block_data->has_changed->dtell_bot();
      block_data->fp_engine->barrier();
      if(threadIdx.x == 0) {
        a.on_node();
        if(a.ipc->is_top()) {
          a.on_failed_node();
        }
        else if(a.bab->refine(has_changed)) {
          if(!a.on_solution_node()) {
            stop = true;
          }
        }
        if(!stop) {
          a.search_tree->refine(has_changed);
        }
      }
      block_data->has_changed->tell(has_changed);
      block_data->fp_engine->barrier();
    }
    block_data->fp_engine->barrier();
    if(threadIdx.x == 0) {
      a0->join(*block_data->abstract_doms);
      delete block_data;
    }
  }
}

void increase_memory_limits_interpretation(const Config& config) {
  size_t max_stack_size = 1024;
  CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, max_stack_size*100));
  cudaDeviceGetLimit(&max_stack_size, cudaLimitStackSize);
  if(config.verbose_solving) {
    printf("%%GPU_max_stack_size=%zu (%zuKB)\n", max_stack_size, max_stack_size/1000);
  }
  size_t max_heap_size;
  cudaDeviceGetLimit(&max_heap_size, cudaLimitMallocHeapSize);
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_heap_size*10));
  cudaDeviceGetLimit(&max_heap_size, cudaLimitMallocHeapSize);
  if(config.verbose_solving) {
    printf("%%GPU_max_heap_size=%zu (%zuMB)\n", max_heap_size, max_heap_size/1000/1000);
  }
}

void increase_memory_limits_solving(const Config& config) {
  size_t max_stack_size = 1024;
  CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, max_stack_size*100));
  cudaDeviceGetLimit(&max_stack_size, cudaLimitStackSize);
  if(config.verbose_solving) {
    printf("%%GPU_max_stack_size=%zu (%zuKB)\n", max_stack_size, max_stack_size/1000);
  }
}

// Inspired by https://stackoverflow.com/questions/39513830/launch-cuda-kernel-with-a-timeout/39514902
// Timeout expected in seconds.
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
  printf("%%%s.total_bytes_allocated=%zu (%zuKB); %zu allocations; %zu deallocations\n",
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

void run_gpu_kernel(shared_ptr<A0, managed_allocator> a0, shared_ptr<bool, managed_allocator> is_timeout) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  const Config& config = a0->config;
  size_t shared_mem_capacity = deviceProp.sharedMemPerBlock;
  // increase_memory_limits_solving(config);
  increase_memory_limits_interpretation(config);

  // We add 33% extra memory due to the alignment of the shared memory which is not taken into account in the statistics.
  size_t store_bytes = a0->store_allocator.total_bytes_allocated();
  store_bytes += store_bytes / 3;
  size_t prop_bytes = a0->prop_allocator.total_bytes_allocated();
  prop_bytes += prop_bytes / 3;
  size_t both_bytes = store_bytes + prop_bytes;

  if(shared_mem_capacity < store_bytes) {
    if(config.verbose_solving) {
      printf("%%The store of variables (%zuKB) cannot be stored in the shared memory of the GPU (%zuKB), therefore we use the global memory.\n",
      store_bytes / 1000,
      shared_mem_capacity / 1000);
    }
    gpu_solve_kernel<A1><<<a0->config.or_nodes, a0->config.and_nodes>>>(a0.get(), is_timeout.get(), 0);
  }
  else if(shared_mem_capacity > both_bytes) {
    if(config.verbose_solving) {
      printf("%%The store of variables and the propagators (%zuKB) are stored in the shared memory of the GPU (%zuKB).\n",
      both_bytes / 1000,
      shared_mem_capacity / 1000);
    }
    gpu_solve_kernel<A3><<<a0->config.or_nodes, a0->config.and_nodes, both_bytes>>>(a0.get(), is_timeout.get(), both_bytes);
  }
  else {
    if(config.verbose_solving) {
      printf("%%The store of variables (%zuKB) is stored in the shared memory of the GPU (%zuKB).\n",
        store_bytes / 1000,
        shared_mem_capacity / 1000);
    }
    gpu_solve_kernel<A2><<<a0->config.or_nodes, a0->config.and_nodes, store_bytes>>>(a0.get(), is_timeout.get(), store_bytes);
  }
  CUDAEX(cudaDeviceSynchronize());
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
  auto a0 = make_shared<A0, managed_allocator>(config, basic_allocator, prop_allocator, store_allocator);
  // I. Parse the FlatZinc model.
  battery::shared_ptr<TFormula<A0::basic_allocator_type>, A0::basic_allocator_type> f =
    parse_flatzinc<A0::basic_allocator_type>(a0->config.problem_path.data(), a0->fzn_output);
  if(!f) {
    std::cerr << "Could not parse FlatZinc model." << std::endl;
    exit(EXIT_FAILURE);
  }
  if(config.verbose_solving) {
    printf("%%FlatZinc parsed\n");
  }

  if(config.print_ast) {
    printf("Parsed AST:\n");
    f->print(true);
    printf("\n");
  }

  auto failure = make_shared<bool, managed_allocator>(false);
  auto is_timeout = make_shared<bool, managed_allocator>(false);
  // increase_memory_limits_interpretation(config);
  // initialize_abstract_domains<<<1,1>>>(f.get(), a0.get(), failure.get());
  // CUDAEX(cudaDeviceSynchronize());

  // I. Create the abstract domains.
  a0->allocate(num_quantified_vars(*f));

  // II. Interpret the formula in the abstract domain.
  a0->typing(*f);
  if(config.print_ast) {
    printf("Typed AST:\n");
    f->print(true);
    printf("\n");
  }
  if(!a0->interpret(*f)) {
    *failure = true;
  }
  if(!(*failure)) {
    if(config.print_ast) {
      printf("Interpreted AST:\n");
      a.ipc->deinterpret(a.env).print(true);
      printf("\n");
    }
    auto interpretation_time = std::chrono::high_resolution_clock::now();
    a0->stats.interpretation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(interpretation_time - start).count();
    if(config.verbose_solving) {
      printf("%%Formula has been loaded, solving begins...\n");
      print_allocation_statistics(*a0);
    }
    std::thread timeout_thread(guard_timeout, a0->config.timeout_ms, is_timeout.get());
    run_gpu_kernel(a0, is_timeout);
    *is_timeout = true;
    check_timeout(*a0, interpretation_time);
    timeout_thread.join();
  }
  // deallocate_abstract_domains<<<1,1>>>(a0.get());
  // CUDAEX(cudaDeviceSynchronize());
  a0->on_finish();
  a0->deallocate();
#endif
}

#endif // TURBO_GPU_SOLVING_HPP
