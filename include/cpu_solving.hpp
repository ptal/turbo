// Copyright 2023 Pierre Talbot

#ifndef TURBO_CPU_SOLVING_HPP
#define TURBO_CPU_SOLVING_HPP

#include "common_solving.hpp"

void cpu_solve(const Configuration<battery::standard_allocator>& config) {
  auto start = std::chrono::steady_clock::now();

  CP<Itv> cp(config);
  cp.config.or_nodes = 1;
  cp.preprocess();
  if(cp.iprop->is_bot()) {
    cp.print_final_solution();
    cp.print_mzn_statistics();
    return;
  }

  FixpointSubsetCPU<GaussSeidelIteration> fp_engine(cp.iprop->num_deductions());
  local::B has_changed = true;
  block_signal_ctrlc();
  while(!must_quit(cp) && check_timeout(cp, start) && has_changed) {
    has_changed = false;
    auto start2 = cp.stats.start_timer_host();
    cp.stats.fixpoint_iterations += fp_engine.fixpoint([&](int i) { return cp.iprop->deduce(i); });
    start2 = cp.stats.stop_timer(Timer::FIXPOINT, start2);
    bool must_prune = cp.on_node();
    if(cp.iprop->is_bot()) {
      cp.on_failed_node();
      fp_engine.reset();
    }
    else {
      fp_engine.select([&](int i) { return !cp.iprop->ask(i); });
      cp.stats.stop_timer(Timer::SELECT_FP_FUNCTIONS, start2);
      if(fp_engine.num_active() == 0 && cp.search_tree->template is_extractable<AtomicExtraction>()) {
        has_changed |= cp.bab->deduce();
        must_prune |= cp.on_solution_node();
        fp_engine.reset();
      }
    }
    has_changed |= cp.search_tree->deduce();
    cp.stats.stop_timer(Timer::SEARCH, start2);
    if(must_prune) { break; }
  }
  cp.print_final_solution();
  cp.print_mzn_statistics();
}

#endif
