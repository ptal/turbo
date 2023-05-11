// Copyright 2023 Pierre Talbot

#ifndef GPU_SOLVING_HPP
#define GPU_SOLVING_HPP

#include "common_solving.hpp"
#include <thread>

#ifdef __NVCC__

using F = TFormula<ManagedAllocator>;
using FormulaPtr = battery::shared_ptr<F, battery::ManagedAllocator>;

using Itv1 = Interval<local::ZInc>;
using Itv2 = Interval<ZInc<int, AtomicMemoryBlock<GlobalAllocator>>>;
using AtomicBInc = BInc<AtomicMemoryBlock<GlobalAllocator>>;
using FPEngine = AsynchronousIterationGPU<GlobalAllocator>;
using A0 = AbstractDomains<Itv1, ManagedAllocator>;
using A1 = AbstractDomains<Itv2, GlobalAllocator>;

struct BlockData {
  shared_ptr<FPEngine, GlobalAllocator> fp_engine;
  shared_ptr<AtomicBInc, GlobalAllocator> has_changed;
  shared_ptr<A1, GlobalAllocator> a1;
  BlockData() = default;
};

/** The interpretation must be done on the device because some abstract domains use runtime polymorphism and thus rely on vtable.
 * Since vtable cannot migrate between host and device, we must initialize those objects on the device. */
__global__ void initialize_abstract_domains(FormulaPtr f,
    shared_ptr<A0, ManagedAllocator> a0,
    shared_ptr<bool, ManagedAllocator> failed)
{
  assert(threadIdx.x == 0 && blockDim.x == 1);
  // I. Create the abstract domains.
  a0->allocate(num_quantified_vars(*f));
  // II. Interpret the formula in the abstract domain.
  if(!a0->interpret(*f)) {
    *failed = true;
  }
}

/** The members of `A` cannot be deleted on the host size since they were allocated on the global memory in the kernel `initialize_abstract_domains`. */
__global__ void deallocate_abstract_domains(shared_ptr<A0, ManagedAllocator> a0)
{
  a0->deallocate();
}

__global__ void gpu_solve_kernel(
  shared_ptr<A0, ManagedAllocator> a0,
  shared_ptr<BlockData, ManagedAllocator> block_data,
  shared_ptr<bool, ManagedAllocator> is_timeout)
{
  if(blockIdx.x == 0) {
    if(threadIdx.x == 0) {
      block_data->fp_engine = make_shared<FPEngine, GlobalAllocator>();
      block_data->has_changed = make_shared<AtomicBInc, GlobalAllocator>(true);
      block_data->a1 = make_shared<A1, GlobalAllocator>(a0);
      printf("%%GPU_block_size=%d\n", blockDim.x);
    }
    __syncthreads();
    A2& a = *block_data->a1;
    while(*(block_data->has_changed) && !(a.bab->is_top()) && !(*is_timeout)) {
      local::BInc has_changed;
      block_data->fp_engine->fixpoint(*a.ipc, has_changed);
      block_data->has_changed->dtell_bot();
      block_data->fp_engine->barrier();
      if(threadIdx.x == 0) {
        a.split->reset();
        GaussSeidelIteration{}.iterate(*a.split, has_changed);
        a.on_node();
        if(a.ipc->is_top()) {
          a.on_failed_node();
        }
        else if(a.bab->refine(a.env, has_changed)) {
          if(!a.on_solution_node()) {
            break;
          }
        }
        a.search_tree->refine(a.env, has_changed);
      }
      block_data->has_changed->tell(has_changed);
      block_data->fp_engine->barrier();
    }
    if(threadIdx.x == 0) {
      block_data->fp_engine = nullptr;
      block_data->has_changed = nullptr;
      *a2 = nullptr;
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
inline void guard_timeout(int timeout_ms, bool& is_timeout) {
  if(timeout_ms == 0) {
    return;
  }
  int progressed = 0;
  while (!is_timeout) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    progressed += 1000;
    if (progressed >= timeout_ms) {
      is_timeout = true;
    }
  }
}

void gpu_solve(const Configuration<StandardAllocator>& config) {
  #ifndef __NVCC__
    std::cout << "You must use the NVCC compiler to compile Turbo on GPU." << std::endl;
  #else
  auto start = std::chrono::high_resolution_clock::now();

  auto a0 = make_shared<A0, ManagedAllocator>(std::move(A0(config)));
  auto a2 = make_shared<shared_ptr<A2, GlobalAllocator>, ManagedAllocator>();
  // I. Parse the FlatZinc model.
  FormulaPtr f = parse_flatzinc<ManagedAllocator>(a->config.problem_path.data(), a->fzn_output);
  if(!f) {
    std::cerr << "Could not parse FlatZinc model." << std::endl;
    exit(EXIT_FAILURE);
  }

  printf("%%FlatZinc parsed\n");

  increase_memory_limits();

  auto failure = make_shared<bool, ManagedAllocator>(false);
  auto is_timeout = make_shared<bool, ManagedAllocator>(false);
  auto block_data = make_shared<BlockData, ManagedAllocator>();
  initialize_abstract_domains<<<1,1>>>(f, a, failure);
  CUDIE(cudaDeviceSynchronize());
  if(!(*failure)) {
    auto interpretation_time = std::chrono::high_resolution_clock::now();
    a2->stats.interpretation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(interpretation_time - start).count();
    printf("%%Formula has been loaded, solving begins...\n");
    std::thread timeout_thread(guard_timeout, a2->config.timeout_ms, std::ref(*is_timeout));
    gpu_solve_kernel<<<a2->config.or_nodes, a2->config.and_nodes>>>(a0, a2, block_data, is_timeout);
    CUDIE(cudaDeviceSynchronize());
    *is_timeout = true;
    printf("%%Problem solved.\n");
    check_timeout(*a, interpretation_time);
    timeout_thread.join();
  }
  a2->stats.print_mzn_final_separator();
  if(a2->config.print_statistics) {
    a2->stats.print_mzn_statistics();
  }
  deallocate_abstract_domains<<<1,1>>>(a);
  CUDIE(cudaDeviceSynchronize());
#endif
}

#endif
