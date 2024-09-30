// Copyright 2023 Pierre Talbot

#ifndef TURBO_CPU_SOLVING_HPP
#define TURBO_CPU_SOLVING_HPP

#include "common_solving.hpp"

void cpu_solve(const Configuration<battery::standard_allocator>& config) {
  auto start = std::chrono::high_resolution_clock::now();

  CP<Itv> cp(config);
  cp.preprocess();

  GaussSeidelIteration fp_engine;
  local::B has_changed = true;
  block_signal_ctrlc();
  while(!must_quit() && check_timeout(cp, start) && has_changed) {
    has_changed = false;
    local::B has_changed2 = false;
    cp.stats.fixpoint_iterations += fp_engine.fixpoint(*cp.ipc, has_changed2);
    bool must_prune = cp.on_node();
    if(cp.ipc->is_bot()) {
      cp.on_failed_node();
    }
    else if(cp.search_tree->template is_extractable<AtomicExtraction>()) {
      has_changed |= cp.bab->deduce();
      must_prune |= cp.on_solution_node();
    }
    has_changed |= cp.search_tree->deduce();
    if(must_prune) { break; }
  }
  cp.print_final_solution();
  cp.print_mzn_statistics();
}

#endif
