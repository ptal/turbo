// Copyright 2023 Pierre Talbot

#ifndef TURBO_CPU_SOLVING_HPP
#define TURBO_CPU_SOLVING_HPP

#include "common_solving.hpp"

using Itv = Interval<local::ZInc>;
using A = AbstractDomains<Itv,
  standard_allocator,
  UniqueLightAlloc<standard_allocator, 0>,
  UniqueLightAlloc<standard_allocator, 1>>;

void cpu_solve(const Configuration<standard_allocator>& config) {
  auto start = std::chrono::high_resolution_clock::now();

  A a(config);

  // I. Parse the FlatZinc model.
  using FormulaPtr = battery::shared_ptr<TFormula<standard_allocator>, standard_allocator>;
  FormulaPtr f = parse_flatzinc<standard_allocator>(config.problem_path.data(), a.fzn_output);
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

  // II. Create the abstract domain.
  a.allocate(num_quantified_vars(*f));

  // III. Interpret the formula in the abstract domain.
  a.typing(*f);
  if(config.print_ast) {
    printf("Typed AST:\n");
    f->print(true);
    printf("\n");
  }
  if(!a.interpret(*f)) {
    exit(EXIT_FAILURE);
  }

  if(config.print_ast) {
    printf("Interpreted AST:\n");
    a.ipc->deinterpret(a.env).print(true);
    printf("\n");
  }

  auto interpretation_time = std::chrono::high_resolution_clock::now();
  a.stats.interpretation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(interpretation_time - start).count();

  if(config.verbose_solving) {
    printf("%%Formula has been loaded, solving begins...\n");
  }

  // IV. Solve the problem.
  local::BInc has_changed = true;
  GaussSeidelIteration fp_engine;
  while(check_timeout(a, interpretation_time) && has_changed) {
    has_changed = false;
    fp_engine.fixpoint(*a.ipc, has_changed);
    a.on_node();
    if(a.ipc->is_top()) {
      a.on_failed_node();
    }
    else if(a.bab->refine(has_changed)) {
      if(!a.on_solution_node()) {
        break;
      }
    }
    a.search_tree->refine(has_changed);
  }
  a.on_finish();
}

#endif
