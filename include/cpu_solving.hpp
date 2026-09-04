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
      [&](int i) { return cp.iprop->fdeduce(i, cp.config.epsilon); }
    );
#else 
    cp.stats.fixpoint_iterations += fp_engine.fixpoint(
      [&](int i) { return cp.iprop->deduce(i); }, 
      [&]() { return cp.iprop->is_bot(); });
#endif
    start2 = cp.stats.stop_timer(Timer::FIXPOINT, start2);
    bool must_prune = cp.on_node(); 
    if(cp.iprop->is_bot()) {
#ifdef WITH_NNV
      /** `is_unknown(env, epsilon)` cannot detect the "UASS batch" case: once all remaining
       * variables have been collapsed to their midpoint (see `fvar_map_fold_left` in
       * split_strategy.hpp), they are exactly assigned (`lb == ub`), which `is_unknown` treats
       * as a genuine full assignment rather than an under-approximation of a still-nonempty
       * epsilon-small box. `is_uass()` is the correct signal here — it reflects whether *this*
       * box was produced by that under-approximating collapse (in which case a failure here
       * must be reported `unknown`, not `failed`: we only tested one arbitrary representative
       * point of the box, not the whole box, so we haven't proven infeasibility). */
      if(cp.search_tree->is_uass()) {
        cp.on_unknown_node();
      }
      else {
        cp.on_failed_node();
      }
      fp_engine.reset();
#else 
      cp.on_failed_node();
      fp_engine.reset();
#endif
    }
    else {
#ifdef WITH_NNV
      cp.stats.stop_timer(Timer::SELECT_FP_FUNCTIONS, start2);
      // if(cp.search_tree->template is_extractable<AtomicExtraction>(AtomicExtraction(), config.epsilon)) {
      if(cp.search_tree->is_solution(cp.env)) {
        has_changed |= cp.bab->deduce();
        must_prune |= cp.on_solution_node();
        fp_engine.reset();
        break;
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
  /** `must_quit` (an external SIGINT/SIGTERM, see `block_signal_ctrlc`) can end the loop just as
   * abruptly as a genuine timeout, but for a completely different reason: the search may not have
   * covered anywhere close to the whole space yet. The verdict below already treats "not really
   * exhausted" correctly for the timeout case (`!check_timeout(...)` -> `timeout`); it must treat
   * an external signal the same way, otherwise an interrupted (Ctrl+C'd, or killed by a wrapper
   * script/scheduler) run with `solutions == 0 && unknowns == 0` silently falls through to the
   * final `unsat`, unsoundly claiming a proof of infeasibility that was never actually completed. */
  bool interrupted = must_quit(cp);
  cp.print_final_solution();
  cp.print_mzn_statistics();

  if (cp.stats.solutions > 0) printf("sat\n");
  else if (cp.stats.unknowns > 0 && check_timeout(cp, start) && !interrupted) printf("unknown\n");
  else if (!check_timeout(cp, start) || interrupted) printf("timeout\n");
  else printf("unsat\n");
}

#endif
