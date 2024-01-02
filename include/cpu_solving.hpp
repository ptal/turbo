// Copyright 2023 Pierre Talbot

#ifndef TURBO_CPU_SOLVING_HPP
#define TURBO_CPU_SOLVING_HPP

#include "common_solving.hpp"

void cpu_solve(const Configuration<battery::standard_allocator>& config) {
  auto start = std::chrono::high_resolution_clock::now();

  CP cp(config);
  cp.preprocess();

  GaussSeidelIteration fp_engine;
  local::BInc has_changed = true;
  block_signal_ctrlc();
  while(has_changed) {
    if(must_quit() || !check_timeout(cp, start)) {
      cp.stats.exhaustive = false;
      break;
    }
    has_changed = false;
    cp.stats.fixpoint_iterations += fp_engine.fixpoint(*cp.ipc, has_changed);
    cp.on_node();
    if(cp.ipc->is_top()) {
      cp.on_failed_node();
    }
    else if(cp.search_tree->template is_extractable<AtomicExtraction>()) {
      cp.bab->refine(has_changed);
      if(!cp.on_solution_node()) {
        break;
      }
    }
    cp.search_tree->refine(has_changed);
  }
  cp.print_final_solution();
  cp.print_mzn_statistics();
}

#endif
