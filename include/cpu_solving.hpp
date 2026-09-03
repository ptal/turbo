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
      // A leaf is accepted once the *decision* (branch) variables are pinned to
      // an exact point (`is_solution`) or can no longer be split further given
      // `epsilon` (`is_unknown`): under epsilon-based splitting, boxes almost
      // never collapse to an exact point, so requiring `lb == ub` alone never
      // fires in practice. We deliberately do NOT require every internal/derived
      // variable (e.g. hidden-layer activations) to also be epsilon-tight: their
      // width only reflects unavoidable interval-arithmetic rounding once the
      // inputs are pinned, not unresolved splitting, and demanding it collapse
      // below `epsilon` too is an over-strict criterion the search can never
      // satisfy for deep/chained networks.
      if(cp.search_tree->is_solution(cp.env) || cp.search_tree->is_unknown(cp.env, cp.config.epsilon)) {
        /** `is_solution`/`is_unknown` are NECESSARY but NOT SUFFICIENT evidence of
         * a real solution (Algorithm 1, Definition `def-verify`): `cp.iprop` was
         * told BOTH the network's equations AND the postcondition, so a
         * sound-but-not-exact forward enclosure of an auxiliary/output
         * variable can have been met against the goal region and collapsed
         * to a non-bottom box that does not correspond to any real solution
         * (Example `ex-phantom`). We certify via the meet-free forward
         * inclusion test (`cp.verify()`) before ever declaring `sat`. */
        if(cp.verify()) {
          has_changed |= cp.bab->deduce();
          must_prune |= cp.on_solution_node();
        }
        else {
          /** Spurious fixed point (candidate failed `verify`), or no
           * verification oracle was available at all: drop this candidate
           * rather than declare a possibly-unsound `sat`. Treated as an
           * unresolved leaf, not a failure: we have not proven this box
           * infeasible, only that its one examined point isn't a genuine
           * solution. */
          cp.on_unknown_node();
        }
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
    /** `fdeduce`'s return value reflects whether committing to the new branch's tell changed
     * anything in the abstract domain. For a "UASS" single-child branch (see split_strategy.hpp),
     * the tell (`x == mid`) is a no-op: the batch collapse already directly `embed`-ed every
     * remaining variable — including `x` — to that same midpoint as a side effect, *before* the
     * branch was even created. So the tell changes nothing, `has_changed` stays false, and the
     * loop exits here — without ever having called `on_node()`/checked `is_bot()`/`is_solution()`
     * on the node that was just pushed. That silently skips examining the one candidate point
     * meant to represent the entire remaining epsilon-small box, which can turn a genuine
     * "unknown" (or even a missed solution) into an unsound "unsat". Comparing `depth()` before
     * and after catches this: a new node was pushed onto the search-tree stack regardless of
     * whether its tell was a no-op, so the loop must keep going to examine it. */
    int depth_before_split = cp.search_tree->depth();
    has_changed |= cp.search_tree->fdeduce(cp.env, cp.config.epsilon);  // add branching strategies
    has_changed |= (cp.search_tree->depth() != depth_before_split);
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
