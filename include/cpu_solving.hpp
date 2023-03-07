// Copyright 2023 Pierre Talbot

#ifndef CPU_SOLVING_HPP
#define CPU_SOLVING_HPP

#include "common_solving.hpp"

using Itv = Interval<local::ZInc>;
using A = AbstractDomains<Itv, StandardAllocator>;

template <class Timepoint>
bool check_timeout(A& a, const Timepoint& start) {
  if(a.config.timeout_ms == 0) {
    return true;
  }
  auto now = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
  if(elapsed >= a.config.timeout_ms) {
    a.stats.exhaustive = false;
    return false;
  }
  return true;
}

void cpu_solve(const Configuration<StandardAllocator>& config) {
  auto start = std::chrono::high_resolution_clock::now();

  A a(config);

  // I. Parse the FlatZinc model.
  using FormulaPtr = battery::shared_ptr<TFormula<StandardAllocator>, StandardAllocator>;
  FormulaPtr f = parse_flatzinc<StandardAllocator>(config.problem_path.data(), a.fzn_output);
  if(!f) {
    std::cerr << "Could not parse FlatZinc model." << std::endl;
    exit(EXIT_FAILURE);
  }

  printf("%%FlatZinc parsed\n");

  // II. Create the abstract domain.
  int num_vars = num_quantified_vars(*f);
  a.store = make_shared<A::IStore, StandardAllocator>(a.sty, num_vars);
  a.ipc = make_shared<A::IPC, StandardAllocator>(A::IPC(a.pty, a.store));
  a.split = make_shared<A::ISplitInputLB, StandardAllocator>(A::ISplitInputLB(a.split_ty, a.ipc, a.ipc));
  a.search_tree = make_shared<A::IST, StandardAllocator>(A::IST(a.tty, a.ipc, a.split));
  a.bab = make_shared<A::IBAB, StandardAllocator>(A::IBAB(a.bab_ty, a.search_tree));

  // III. Interpret the formula in the abstract domain.
  auto r = a.bab->interpret_in(*f, a.env);
  if(!r.has_value()) {
    r.print_diagnostics();
    exit(EXIT_FAILURE);
  }
  local::BInc has_changed;
  a.bab->tell(std::move(r.value()), has_changed);
  a.stats.variables = a.store->vars();
  a.stats.constraints = a.ipc->num_refinements();

  auto now = std::chrono::high_resolution_clock::now();
  a.stats.interpretation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();

  printf("%%Formula has been loaded, solving begins...\n");

  // IV. Solve the problem.
  AbstractDeps<StandardAllocator> deps;
  has_changed = true;
  while(has_changed && check_timeout(a, start)) {
    has_changed = false;
    GaussSeidelIteration::fixpoint(*a.ipc, has_changed);
    a.split->reset();
    GaussSeidelIteration::iterate(*a.split, has_changed);
    if(a.bab->refine(a.env, has_changed)) {
      a.fzn_output.print_solution(a.env, a.bab->optimum());
      a.stats.print_mzn_separator();
      a.stats.solutions++;
      if(config.stop_after_n_solutions != 0 &&
         a.stats.solutions >= config.stop_after_n_solutions)
      {
        a.stats.exhaustive = false;
        break;
      }
    }
    a.search_tree->refine(a.env, has_changed);
  }
  a.stats.print_mzn_final_separator();
  now = std::chrono::high_resolution_clock::now();
  a.stats.duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
  if(a.config.print_statistics) {
    a.stats.print_mzn_statistics();
  }
}

#endif
