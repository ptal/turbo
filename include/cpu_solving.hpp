// Copyright 2023 Pierre Talbot

#ifndef TURBO_CPU_SOLVING_HPP
#define TURBO_CPU_SOLVING_HPP

#include "common_solving.hpp"

void cpu_solve(const Configuration<battery::standard_allocator>& config) {
  auto start = std::chrono::steady_clock::now();

#ifdef WITH_NNV 
  CP<FItv> cp(config);
#else
  CP<Itv> cp(config);
#endif
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
#ifdef WITH_NNV
    cp.stats.fixpoint_iterations += fp_engine.fixpoint([&](int i) { return cp.iprop->fdeduce(i); });
#else 
    cp.stats.fixpoint_iterations += fp_engine.fixpoint([&](int i) { return cp.iprop->deduce(i); });
#endif
    start2 = cp.stats.stop_timer(Timer::FIXPOINT, start2);
    bool must_prune = cp.on_node();
    if(cp.iprop->is_bot()) {
      cp.on_failed_node();
      fp_engine.reset();
    }
    else {
#ifdef WITH_NNV
      fp_engine.select([&](int i) { return !cp.iprop->fask(i); });
      cp.stats.stop_timer(Timer::SELECT_FP_FUNCTIONS, start2);
      printf("fp_engine.num_active() = %d\n", fp_engine.num_active());
      if(fp_engine.num_active() == 0 && cp.search_tree->template is_fextractable<AtomicExtraction>()) {
        has_changed |= cp.bab->fdeduce();
        must_prune |= cp.on_solution_node();
        break;
      }
      else if(cp.search_tree->is_unknown(config.epsilon)) {
        cp.on_unknown_node();
        fp_engine.reset();
      }
#else
      fp_engine.select([&](int i) { return !cp.iprop->ask(i); });
      cp.stats.stop_timer(Timer::SELECT_FP_FUNCTIONS, start2);
      if(fp_engine.num_active() == 0 && cp.search_tree->template is_extractable<AtomicExtraction>()) {
        has_changed |= cp.bab->deduce();
        must_prune |= cp.on_solution_node();
        fp_engine.reset();
      }
#endif
      
    }
#ifdef WITH_NNV 
    has_changed |= cp.search_tree->fdeduce(cp.config.epsilon);  // add branching strategies
#else
    has_changed |= cp.search_tree->deduce();
#endif
    cp.stats.stop_timer(Timer::SEARCH, start2);
    if(must_prune) { break; }
  }
  cp.print_final_solution();
  cp.print_mzn_statistics();
}

#endif
