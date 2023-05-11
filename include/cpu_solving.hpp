// Copyright 2023 Pierre Talbot

#ifndef CPU_SOLVING_HPP
#define CPU_SOLVING_HPP

#include "common_solving.hpp"

using Itv = Interval<local::ZInc>;
using A = AbstractDomains<Itv, StandardAllocator>;

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
  a.allocate(num_quantified_vars(*f));

  // III. Interpret the formula in the abstract domain.
  if(!a.interpret(*f)) {
    exit(EXIT_FAILURE);
  }

  auto interpretation_time = std::chrono::high_resolution_clock::now();
  a.stats.interpretation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(interpretation_time - start).count();

  printf("%%Formula has been loaded, solving begins...\n");

  // IV. Solve the problem.
  local::BInc has_changed = true;
  GaussSeidelIteration fp_engine;
  while(check_timeout(a, interpretation_time) && !a.bab->is_top() && has_changed) {
    has_changed = false;
    fp_engine.fixpoint(*a.ipc, has_changed);
    a.split->reset();
    fp_engine.iterate(*a.split, has_changed);
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
  a.stats.print_mzn_final_separator();
  if(a.config.print_statistics) {
    a.stats.print_mzn_statistics();
  }
}

#endif
