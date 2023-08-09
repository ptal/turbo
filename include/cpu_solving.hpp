// Copyright 2023 Pierre Talbot

#ifndef TURBO_CPU_SOLVING_HPP
#define TURBO_CPU_SOLVING_HPP

#include "common_solving.hpp"

void cpu_solve(const Configuration<standard_allocator>& config) {
  auto start = std::chrono::high_resolution_clock::now();
  CP cp(config);
  cp.prepare_solver();
  local::BInc has_changed = true;
  GaussSeidelIteration fp_engine;
  block_signal_ctrlc();
  while(!must_quit() && check_timeout(cp, start) && has_changed) {
    has_changed = false;
    fp_engine.fixpoint(*cp.ipc, has_changed);
    cp.on_node();
    if(cp.ipc->is_top()) {
      cp.on_failed_node();
    }
    else if(cp.bab->template refine<AtomicExtraction>(has_changed)) {
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
