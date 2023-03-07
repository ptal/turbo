// Copyright 2023 Pierre Talbot

#ifndef GPU_SOLVING_HPP
#define GPU_SOLVING_HPP

#include "common_solving.hpp"

#ifdef __NVCC__

using F = TFormula<ManagedAllocator>;
using FormulaPtr = shared_ptr<F, ManagedAllocator>;

using Itv1 = Interval<local::ZInc>;
using Itv2 = Interval<ZInc<int, AtomicMemoryBlock<GlobalAllocator>>>;
using A1 = AbstractDomains<Itv1, GlobalAllocator, ManagedAllocator>;
using A2 = AbstractDomains<Itv2, GlobalAllocator, ManagedAllocator>;

__global__ void initialize_abstract_domains(FormulaPtr f,
    shared_ptr<A1, ManagedAllocator> a,
    shared_ptr<bool, ManagedAllocator> failed)
{
  assert(threadIdx.x == 0 && blockDim.x == 1);
  int num_vars = num_quantified_vars(*f);

  // I. Create the abstract domains.
  a->store = make_shared<A1::IStore, GlobalAllocator>(a->sty, num_vars);
  a->ipc = make_shared<A1::IPC, GlobalAllocator>(A1::IPC(a->pty, a->store));
  a->split = make_shared<A1::ISplitInputLB, GlobalAllocator>(A1::ISplitInputLB(a->split_ty, a->ipc, a->ipc));
  a->search_tree = make_shared<A1::IST, GlobalAllocator>(A1::IST(a->tty, a->ipc, a->split));
  a->bab = make_shared<A1::IBAB, GlobalAllocator>(A1::IBAB(a->bab_ty, a->search_tree));
  printf("%%Abstract domain initialized.\n");
  // II. Interpret the formula in the abstract domain.
  auto r = a->bab->interpret_in(*f, a->env);
  if(!r.has_value()) {
    r.print_diagnostics();
    *failed = true;
    return;
  }
  local::BInc has_changed;
  a->bab->tell(std::move(r.value()), has_changed);
  a->stats.variables = a->store->vars();
  a->stats.constraints = a->ipc->num_refinements();
}

/** The members of `A` cannot be deleted on the host size since they were allocated on the global memory in the kernel `initialize_abstract_domains`. */
__global__ void deallocate_abstract_domains(shared_ptr<A1, ManagedAllocator> a)
{
  a->store = nullptr;
  a->ipc = nullptr;
  a->split = nullptr;
  a->search_tree = nullptr;
  a->bab = nullptr;
  a->env = VarEnv<GlobalAllocator>{}; // this is to release the memory used by `VarEnv`.
}

__global__ void gpu_solve_kernel(shared_ptr<A1, ManagedAllocator> a)
{
  if(threadIdx.x == 0 && blockIdx.x == 0) {
    AbstractDeps<GlobalAllocator> deps;
    local::BInc has_changed = true;
    while(has_changed) { // && check_timeout(config, stats, start)) {
      has_changed = false;
      GaussSeidelIteration::fixpoint(*a->ipc, has_changed);
      a->split->reset();
      GaussSeidelIteration::iterate(*a->split, has_changed);
      if(a->bab->refine(a->env, has_changed)) {
        a->fzn_output.print_solution(a->env, a->bab->optimum());
        a->stats.print_mzn_separator();
        a->stats.solutions++;
        if(a->config.stop_after_n_solutions != 0 &&
           a->stats.solutions >= a->config.stop_after_n_solutions)
        {
          a->stats.exhaustive = false;
          break;
        }
      }
      a->search_tree->refine(a->env, has_changed);
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

void gpu_solve(const Configuration<StandardAllocator>& config) {
  #ifndef __NVCC__
    std::cout << "You must use the NVCC compiler to compile Turbo on GPU." << std::endl;
  #else
  auto start = std::chrono::high_resolution_clock::now();

  auto a = make_shared<A1, ManagedAllocator>(std::move(A1(config)));
  // I. Parse the FlatZinc model.
  FormulaPtr f = parse_flatzinc<ManagedAllocator>(a->config.problem_path.data(), a->fzn_output);
  if(!f) {
    std::cerr << "Could not parse FlatZinc model." << std::endl;
    exit(EXIT_FAILURE);
  }

  printf("%%FlatZinc parsed\n");

  increase_memory_limits();

  auto failure = make_shared<bool, ManagedAllocator>(false);
  initialize_abstract_domains<<<1,1>>>(f, a, failure);
  CUDIE(cudaDeviceSynchronize());
  if(!(*failure)) {
    auto now = std::chrono::high_resolution_clock::now();
    a->stats.interpretation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    printf("%%Formula has been loaded, solving begins...\n");
    gpu_solve_kernel<<<a->config.or_nodes, a->config.and_nodes>>>(a);
    CUDIE(cudaDeviceSynchronize());
    printf("%%Problem solved.\n");
  }
  deallocate_abstract_domains<<<1,1>>>(a);
  CUDIE(cudaDeviceSynchronize());
  a->stats.print_mzn_final_separator();
  if(a->config.print_statistics) {
    a->stats.print_mzn_statistics();
  }
#endif
}

#endif
