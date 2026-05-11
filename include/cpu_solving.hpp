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
    cp.stats.fixpoint_iterations += fp_engine.fixpoint(
      [&](int i) { return cp.iprop->fdeduce(i, config.epsilon); },
      [&]() { return cp.iprop->is_bot(); }
    );
#else 
    cp.stats.fixpoint_iterations += fp_engine.fixpoint(
      [&](int i) { return cp.iprop->deduce(i); }, 
      [&]() { return cp.iprop->is_bot(); });
#endif
    start2 = cp.stats.stop_timer(Timer::FIXPOINT, start2);
    bool must_prune = cp.on_node();
    if(cp.iprop->is_bot()) {
      cp.on_failed_node();
      fp_engine.reset();
    }
    else {
#ifdef WITH_NNV
      // for(int i = 0; i < cp.iprop->num_deductions(); ++i) {
      //   if(!cp.iprop->is_fsolution(i)) {
      //     std::cout << "It does not have a solution" << std::endl;
      //     has_changed = true;
      //     cp.on_failed_node();
      //     fp_engine.reset();
      //     break;
      //   }
      // }
      cp.stats.stop_timer(Timer::SELECT_FP_FUNCTIONS, start2);
      if(!has_changed) {
        if(cp.search_tree->template is_extractable<AtomicExtraction>(AtomicExtraction(), config.epsilon)) {
          has_changed |= cp.bab->deduce();
          must_prune |= cp.on_solution_node();
          fp_engine.reset();
        }
      }
      else if(cp.search_tree->is_unknown(config.epsilon)) {
        // FIXME: check how to identify unknown nodes.
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
    has_changed |= cp.search_tree->fdeduce(cp.env, cp.config.epsilon);  // add branching strategies
#else
    has_changed |= cp.search_tree->deduce();
#endif
    cp.stats.stop_timer(Timer::SEARCH, start2);
    if(must_prune) { break; }
  }
  cp.print_final_solution();
  cp.print_mzn_statistics();

  if (cp.stats.solutions > 0) printf("sat\n");
  else if (cp.stats.unknowns > 0) printf("unknown\n");
  else if (check_timeout(cp, start)) printf("timeout\n");
  else printf("unsat\n");
}

#endif
