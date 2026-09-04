// Copyright 2025 Pierre Talbot

#ifndef TURBO_INTERPRETATION_HPP
#define TURBO_INTERPRETATION_HPP

#include "search_strategy.hpp"

#include "battery/vector.hpp"
#include "lala/logic/logic.hpp"
#include "lala/interpretation.hpp"

/**
 * This file provides the interpretation of a constraint network in the abstract elements used by
 * the solver. A constraint network is a conjunction of three kinds of top-level predicates:
 *   - `minimize(x)` / `maximize(x)`: the objective, interpreted in an `Objective`.
 *   - `search(VariableOrder, ValueOrder, x1, ..., xn)`: interpreted in a `StrategyType`.
 *   - anything else: a constraint, interpreted in the propagators abstract domain.
 *
 * Interpretation is kept outside of the abstract domains on purpose: a domain is a lattice and
 * does not need to know about logical formulas, and there is generally more than one way to
 * interpret a formula in a product of domains.
 */

/** The diagnostics macros (`RETURN_INTERPRETATION_ERROR`, ...) name `IDiagnostics` unqualified.
 * We only import that one name: a `using namespace lala` at this scope would make the `Sig`
 * enumerators (`LT`, `GT`, `EQ`, `IN`, ...) ambiguous with the XCSP3 parser's own constants. */
using lala::IDiagnostics;

/** The result of interpreting a constraint network: the constraints to be told to the
 * propagators domain `IProp`, the objective and the search strategies. */
template <class IProp, class Alloc>
struct interpreted_cn {
  using allocator_type = Alloc;

  typename IProp::template tell_type<Alloc> constraints;
  Objective objective;
  SearchStrategies<Alloc> strategies;

  CUDA interpreted_cn(const Alloc& alloc = Alloc{})
   : constraints(alloc), strategies(alloc)
  {}

  interpreted_cn(const interpreted_cn&) = default;
  interpreted_cn(interpreted_cn&&) = default;
  interpreted_cn& operator=(const interpreted_cn&) = default;
  interpreted_cn& operator=(interpreted_cn&&) = default;

  CUDA allocator_type get_allocator() const {
    return strategies.get_allocator();
  }
};

/** Interpret `minimize(x)` or `maximize(x)` in `objective`.
 * An objective already fixed to a constant is ignored (warning), as it makes the problem a
 * satisfaction problem. */
template <bool diagnose = false, class F, class Env>
CUDA NI bool interpret_objective(const F& f, Env& env, Objective& objective, IDiagnostics& diagnostics) {
  const char* name = "Objective";
  assert(f.is(F::Seq) && (f.sig() == lala::MINIMIZE || f.sig() == lala::MAXIMIZE));
  if(f.seq(0).is_variable()) {
    if(!objective.is_satisfaction()) {
      RETURN_INTERPRETATION_ERROR("Multi-objective optimization is not supported.");
    }
    lala::AVar x;
    if(!env.template interpret<diagnose>(f.seq(0), x, diagnostics)) {
      return false;
    }
    objective = Objective(x, f.sig() == lala::MINIMIZE);
    return true;
  }
  // If the objective variable is already fixed to a constant, we ignore this predicate.
  // If there is only one objective, it becomes a satisfaction problem.
  else if(lala::num_vars(f.seq(0)) == 0) {
    RETURN_INTERPRETATION_WARNING("This objective is already fixed to a constant, thus it is ignored.");
  }
  else {
    RETURN_INTERPRETATION_ERROR("Optimization predicates expect a variable to optimize (not an expression). Instead, you can create a new variable with the expression to optimize.");
  }
}

/** Interpret a predicate of the form `search(VariableOrder, ValueOrder, x_1, x_2, ..., x_n)` in `strat`. */
template <bool diagnose = false, class F, class Env, class Alloc>
CUDA NI bool interpret_strategy(const F& f, Env& env, StrategyType<Alloc>& strat, IDiagnostics& diagnostics) {
  const char* name = "SearchStrategy";
  if(!(f.is(F::ESeq)
    && f.eseq().size() >= 2
    && f.esig() == "search"
    && f.eseq()[0].is(F::ESeq) && f.eseq()[0].eseq().size() == 0
    && f.eseq()[1].is(F::ESeq) && f.eseq()[1].eseq().size() == 0))
  {
    RETURN_INTERPRETATION_ERROR("A search strategy must be a predicate of the form `search(input_order, indomain_min, x1, ..., xN)`.");
  }
  const auto& var_order_str = f.eseq()[0].esig();
  const auto& val_order_str = f.eseq()[1].esig();
  if(var_order_str == "input_order") { strat.var_order = VariableOrder::INPUT_ORDER; }
  else if(var_order_str == "first_fail") { strat.var_order = VariableOrder::FIRST_FAIL; }
  else if(var_order_str == "anti_first_fail") { strat.var_order = VariableOrder::ANTI_FIRST_FAIL; }
  else if(var_order_str == "smallest") { strat.var_order = VariableOrder::SMALLEST; }
  else if(var_order_str == "largest") { strat.var_order = VariableOrder::LARGEST; }
  else if(var_order_str == "random") { strat.var_order = VariableOrder::RANDOM; }
  else {
    RETURN_INTERPRETATION_ERROR("This variable order strategy is unsupported.");
  }
  if(val_order_str == "indomain_min") { strat.val_order = ValueOrder::MIN; }
  else if(val_order_str == "indomain_max") { strat.val_order = ValueOrder::MAX; }
  else if(val_order_str == "indomain_median") {
    printf("WARNING: indomain_median is not supported since we use interval domain. It is replaced by SPLIT");
    strat.val_order = ValueOrder::SPLIT;
  }
  else if(val_order_str == "indomain_split") { strat.val_order = ValueOrder::SPLIT; }
  else if(val_order_str == "indomain_reverse_split") { strat.val_order = ValueOrder::REVERSE_SPLIT; }
  else {
    RETURN_INTERPRETATION_ERROR("This value order strategy is unsupported.");
  }
  for(int i = 2; i < f.eseq().size(); ++i) {
    if(f.eseq(i).is(F::LV)) {
      strat.vars.push_back(lala::AVar{});
      if(!env.template interpret<diagnose>(f.eseq(i), strat.vars.back(), diagnostics)) {
        return false;
      }
    }
    else if(f.eseq(i).is(F::V)) {
      strat.vars.push_back(f.eseq(i).v());
    }
    else if(lala::num_vars(f.eseq(i)) > 0) {
      RETURN_INTERPRETATION_ERROR("The predicate `search` only supports variables or constants, but an expression containing a variable was passed to it.");
    }
    // Ignore constant expressions.
    else {}
  }
  return true;
}

/** Route each conjunct of the constraint network to the element interpreting it.
 * `true` is interpreted exactly since all the elements involved preserve the top element. */
template <bool diagnose = false, class IProp, class F, class Env, class Alloc>
CUDA NI bool interpret_cn_in(const IProp& iprop, const F& f, Env& env,
  interpreted_cn<IProp, Alloc>& intermediate, IDiagnostics& diagnostics)
{
  if(f.is_true()) {
    return true;
  }
  else if(f.is(F::Seq) && f.sig() == lala::AND) {
    for(int i = 0; i < f.seq().size(); ++i) {
      if(!interpret_cn_in<diagnose>(iprop, f.seq(i), env, intermediate, diagnostics)) {
        return false;
      }
    }
    return true;
  }
  else if(f.is(F::Seq) && (f.sig() == lala::MINIMIZE || f.sig() == lala::MAXIMIZE)) {
    return interpret_objective<diagnose>(f, env, intermediate.objective, diagnostics);
  }
  else if(f.is(F::ESeq) && f.esig() == "search") {
    StrategyType<Alloc> strat(intermediate.get_allocator());
    if(!interpret_strategy<diagnose>(f, env, strat, diagnostics)) {
      return false;
    }
    intermediate.strategies.push_back(std::move(strat));
    return true;
  }
  // Any other formula is a constraint, interpreted in the propagators domain (and, for the
  // formulas it cannot represent, in its underlying store of variables).
  return lala::ginterpret_in<lala::IKind::TELL, diagnose>(iprop, f, env, intermediate.constraints, diagnostics);
}

/** Interpret the constraint network `f` and, on success, deduce the constraints in `iprop` and
 * store the objective and the search strategies in `objective` and `strategies`.
 * On failure, `env`, `iprop`, `objective` and `strategies` are left unchanged.
 * `TellAlloc` is the allocator of the intermediate representation, which only lives for the
 * duration of the interpretation. */
template <bool diagnose = false, class TellAlloc = battery::standard_allocator,
  class IProp, class F, class Env, class Alloc>
CUDA NI bool interpret_and_tell_cn(IProp& iprop, const F& f, Env& env,
  Objective& objective, SearchStrategies<Alloc>& strategies, IDiagnostics& diagnostics,
  TellAlloc tell_alloc = TellAlloc{})
{
  auto snap = env.snapshot();
  interpreted_cn<IProp, TellAlloc> intermediate(tell_alloc);
  intermediate.objective = objective;
  if(!interpret_cn_in<diagnose>(iprop, f, env, intermediate, diagnostics)) {
    env.restore(snap);
    return false;
  }
  iprop.deduce(intermediate.constraints);
  objective = intermediate.objective;
  for(int i = 0; i < intermediate.strategies.size(); ++i) {
    strategies.push_back(StrategyType<Alloc>(intermediate.strategies[i], strategies.get_allocator()));
  }
  return true;
}

#endif
