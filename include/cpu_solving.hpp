// Copyright 2023 Pierre Talbot

#ifndef TURBO_CPU_SOLVING_HPP
#define TURBO_CPU_SOLVING_HPP

#include "common_solving.hpp"

void cpu_solve(const Configuration<battery::standard_allocator>& config) {
  auto start = std::chrono::high_resolution_clock::now();

  CP<Itv> cp(config);
  cp.preprocess();

  FixpointSubsetCPU<GaussSeidelIteration> fp_engine(cp.ipc->num_deductions());
  local::B has_changed = true;
  block_signal_ctrlc();
  while(!must_quit() && check_timeout(cp, start) && has_changed) {
    has_changed = false;
    cp.stats.fixpoint_iterations += fp_engine.fixpoint([&](size_t i) { return cp.ipc->deduce(i); });
    bool must_prune = cp.on_node();
    if(cp.ipc->is_bot()) {
      cp.on_failed_node();
      fp_engine.reset();
    }
    else {
      fp_engine.select([&](size_t i) { return !cp.ipc->ask(i); });
      if(fp_engine.num_active() == 0 && cp.store->template is_extractable<AtomicExtraction>()) {
        has_changed |= cp.bab->deduce();
        must_prune |= cp.on_solution_node();
        fp_engine.reset();
      }
    }
    has_changed |= cp.search_tree->deduce();
    if(must_prune) { break; }
  }
  cp.print_final_solution();
  cp.print_mzn_statistics();
}

#endif
