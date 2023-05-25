// Copyright 2023 Pierre Talbot

#ifndef GPU_SOLVING_HPP
#define GPU_SOLVING_HPP

#include "common_solving.hpp"
#include <thread>

#ifdef __NVCC__

using F = TFormula<managed_allocator>;
using FormulaPtr = battery::shared_ptr<F, managed_allocator>;

/** We first interpret the formula in an abstract domain with sequential managed memory, that we call `A0`. */
using Itv0 = Interval<local::ZInc>;
using A0 = AbstractDomains<Itv0,
  managed_allocator,
  UniqueLightAlloc<managed_allocator, 0>,
  UniqueLightAlloc<managed_allocator, 1>>;

/** Then, once everything is initialized, we rely on a parallel abstract domain called `A1`, using atomic global memory. */
using Itv1 = Interval<ZInc<int, atomic_memory_block<global_allocator>>>;
using AtomicBInc = BInc<atomic_memory_block<global_allocator>>;
using FPEngine = BlockAsynchronousIterationGPU<global_allocator>;
// using A1 = AbstractDomains<Itv1, global_allocator, pool_allocator>;
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
struct BlockData {
  shared_ptr<FPEngine, global_allocator> fp_engine;
  shared_ptr<AtomicBInc, global_allocator> has_changed;
  shared_ptr<A1, global_allocator> a1;
  BlockData() = default;
};

/** The interpretation must be done on the device because some abstract domains use runtime polymorphism and thus rely on vtable.
 * Since vtable cannot migrate between host and device, we must initialize those objects on the device. */
__global__ void initialize_abstract_domains(F* f, A0* a0, bool* failed)
{
  assert(threadIdx.x == 0 && blockDim.x == 1);
  // I. Create the abstract domains.
  a0->allocate(num_quantified_vars(*f));
  // II. Interpret the formula in the abstract domain.
  if(!a0->interpret(*f)) {
    *failed = true;
  }
}

/** The members of `A` cannot be deleted on the host size since they were allocated in the global memory in the kernel `initialize_abstract_domains`. */
__global__ void deallocate_abstract_domains(A0* a0)
{
  a0->on_finish();
  a0->deallocate();
}

__global__ void gpu_solve_kernel(A0* a0, bool* is_timeout)
{
  __shared__ BlockData* block_data;
  __shared__ bool stop;
  if(blockIdx.x == 0) {
    if(threadIdx.x == 0) {
      stop = false;
      block_data = new BlockData();
      auto block = cooperative_groups::this_thread_block();
      block_data->fp_engine = make_shared<FPEngine, global_allocator>(block);
      block_data->has_changed = make_shared<AtomicBInc, global_allocator>(true);
      // pool_allocator shared_mem_pool{};
      block_data->a1 = make_shared<A1, global_allocator>(*a0);
      printf("%%GPU_block_size=%d\n", blockDim.x);
    }
    __syncthreads();
    A1& a = *block_data->a1;
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
      a0->join(*block_data->a1);
      delete block_data;
    }
  }
}

void increase_memory_limits() {
  size_t max_stack_size;
  cudaDeviceGetLimit(&max_stack_size, cudaLimitStackSize);
  cudaDeviceSetLimit(cudaLimitStackSize, max_stack_size*50);
  cudaDeviceGetLimit(&max_stack_size, cudaLimitStackSize);
  std::cout << "%GPU_max_stack_size=" << max_stack_size << std::endl;
  size_t max_heap_size;
  cudaDeviceGetLimit(&max_heap_size, cudaLimitMallocHeapSize);
  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_heap_size*10);
  std::cout << "%GPU_max_heap_size=" << max_heap_size << std::endl;
}

#endif

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

void gpu_solve(const Configuration<standard_allocator>& config) {
  #ifndef __NVCC__
    std::cout << "You must use the NVCC compiler to compile Turbo on GPU." << std::endl;
  #else
  auto start = std::chrono::high_resolution_clock::now();

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  size_t shared_mem_capacity = deviceProp.sharedMemPerBlock;
  auto a0 = make_shared<A0, managed_allocator>(config);
  // I. Parse the FlatZinc model.
  FormulaPtr f = parse_flatzinc<managed_allocator>(a0->config.problem_path.data(), a0->fzn_output);
  if(!f) {
    std::cerr << "Could not parse FlatZinc model." << std::endl;
    exit(EXIT_FAILURE);
  }

  printf("%%FlatZinc parsed\n");

  increase_memory_limits();

  // printf("  Total amount of constant memory:               %zu bytes\n",
  //       deviceProp.totalConstMem);
  // printf("  Total amount of shared memory per block:       %zu bytes\n",
  //         );
  // printf("  Total shared memory per multiprocessor:        %zu bytes\n",
  //         deviceProp.sharedMemPerMultiprocessor);
  // printf("  Total amount of global memory:                 %.0f MBytes "
  //     "(%llu bytes)\n",
  //     static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
  //     (unsigned long long)deviceProp.totalGlobalMem);
  // printf("  Maximum number of threads per multiprocessor:  %d\n",
  //          deviceProp.maxThreadsPerMultiProcessor);
  // printf("  Maximum number of threads per block:           %d\n",
  //          deviceProp.maxThreadsPerBlock);
  // printf("  Number of SMs: %lu\n", deviceProp.multiProcessorCount);

  auto failure = make_shared<bool, managed_allocator>(false);
  auto is_timeout = make_shared<bool, managed_allocator>(false);
  initialize_abstract_domains<<<1,1>>>(f.get(), a0.get(), failure.get());
  CUDIE(cudaDeviceSynchronize());
  if(!(*failure)) {
    auto interpretation_time = std::chrono::high_resolution_clock::now();
    a0->stats.interpretation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(interpretation_time - start).count();
    printf("%%Formula has been loaded, solving begins...\n");
    std::thread timeout_thread(guard_timeout, a0->config.timeout_ms, is_timeout.get());
    gpu_solve_kernel<<<a0->config.or_nodes, a0->config.and_nodes>>>(a0.get(), is_timeout.get());
    CUDIE(cudaDeviceSynchronize());
    *is_timeout = true;
    check_timeout(*a0, interpretation_time);
    timeout_thread.join();
  }
  deallocate_abstract_domains<<<1,1>>>(a0.get());
  CUDIE(cudaDeviceSynchronize());
#endif
}

#endif
