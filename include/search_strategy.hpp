// Copyright 2025 Pierre Talbot

#ifndef TURBO_SEARCH_STRATEGY_HPP
#define TURBO_SEARCH_STRATEGY_HPP

#include <optional>
#include <algorithm>

#include "battery/vector.hpp"
#include "lala/logic/ast.hpp"

/** The vocabulary of search strategies: a variable order, a value order and the set of
 * variables they apply to. These are plain descriptions of a search strategy; the algorithm
 * consuming them lives in the solving architecture (see `barebones_dive_and_solve.hpp`).
 */

enum class VariableOrder {
  INPUT_ORDER,
  FIRST_FAIL,
  ANTI_FIRST_FAIL,
  SMALLEST,
  LARGEST,
  RANDOM
  // unsupported:
  // OCCURRENCE,
  // MOST_CONSTRAINED,
  // MAX_REGRET,
  // DOM_W_DEG,
};

inline const char* string_of_variable_order(VariableOrder order) {
  switch(order) {
    case VariableOrder::INPUT_ORDER: return "input_order";
    case VariableOrder::FIRST_FAIL: return "first_fail";
    case VariableOrder::ANTI_FIRST_FAIL: return "anti_first_fail";
    case VariableOrder::SMALLEST: return "smallest";
    case VariableOrder::LARGEST: return "largest";
    case VariableOrder::RANDOM: return "random";
    default: return "unknown";
  }
}

template <class StringType>
std::optional<VariableOrder> variable_order_of_string(const StringType& str) {
  if(str == "input_order") {
    return VariableOrder::INPUT_ORDER;
  }
  else if(str == "first_fail") {
    return VariableOrder::FIRST_FAIL;
  }
  else if(str == "anti_first_fail") {
    return VariableOrder::ANTI_FIRST_FAIL;
  }
  else if(str == "smallest") {
    return VariableOrder::SMALLEST;
  }
  else if(str == "largest") {
    return VariableOrder::LARGEST;
  }
  else if(str == "random") {
    return VariableOrder::RANDOM;
  }
  else {
    return std::nullopt;
  }
}

enum class ValueOrder {
  MIN,
  MAX,
  MEDIAN,
  SPLIT,
  REVERSE_SPLIT,
  // unsupported:
  // INTERVAL,
  // RANDOM,
  // MIDDLE,
};

inline const char* string_of_value_order(ValueOrder order) {
  switch(order) {
    case ValueOrder::MIN: return "min";
    case ValueOrder::MAX: return "max";
    case ValueOrder::MEDIAN: return "median";
    case ValueOrder::SPLIT: return "split";
    case ValueOrder::REVERSE_SPLIT: return "reverse_split";
    default: return "unknown";
  }
}

template <class StringType>
std::optional<ValueOrder> value_order_of_string(const StringType& str) {
  if(str == "min") {
    return ValueOrder::MIN;
  }
  else if(str == "max") {
    return ValueOrder::MAX;
  }
  else if(str == "median") {
    return ValueOrder::MEDIAN;
  }
  else if(str == "split") {
    return ValueOrder::SPLIT;
  }
  else if(str == "reverse_split") {
    return ValueOrder::REVERSE_SPLIT;
  }
  else {
    return std::nullopt;
  }
}

/** A split strategy consists of a variable order and value order on a subset of the variables. */
template <class Allocator>
struct StrategyType {
  using allocator_type = Allocator;

  VariableOrder var_order;
  ValueOrder val_order;
  // An empty vector of variables means we should split on the underlying store directly.
  battery::vector<lala::AVar, Allocator> vars;

  CUDA StrategyType(const Allocator& alloc = Allocator{})
   : var_order(VariableOrder::INPUT_ORDER), val_order(ValueOrder::MIN), vars(alloc)
  {}

  StrategyType(const StrategyType<Allocator>&) = default;
  StrategyType(StrategyType<Allocator>&&) = default;
  StrategyType& operator=(StrategyType<Allocator>&&) = default;
  StrategyType& operator=(const StrategyType<Allocator>&) = default;

  CUDA StrategyType(VariableOrder var_order, ValueOrder val_order, battery::vector<lala::AVar, Allocator>&& vars)
   : var_order(var_order), val_order(val_order), vars(std::move(vars))
  {}

  CUDA allocator_type get_allocator() const {
    return vars.get_allocator();
  }

  template <class StrategyType2>
  CUDA StrategyType(const StrategyType2& other, const Allocator& alloc = Allocator{})
  : var_order(other.var_order), val_order(other.val_order), vars(other.vars, alloc) {}

  template <class Alloc2>
  friend class StrategyType;
};

template <class Allocator>
using SearchStrategies = battery::vector<StrategyType<Allocator>, Allocator>;

/** Insert a strategy in front of all the others, to be used for the embarrassingly parallel
 * search decomposition (EPS) of the problem into subproblems. */
template <class Allocator>
CUDA void push_eps_strategy(SearchStrategies<Allocator>& strategies, VariableOrder var_order, ValueOrder val_order) {
  strategies.push_back(StrategyType<Allocator>(strategies.get_allocator()));
  for(int i = strategies.size() - 1; i > 0; --i) {
    strategies[i] = strategies[i-1];
  }
  battery::vector<lala::AVar, Allocator> vars(strategies.get_allocator());
  strategies[0] = StrategyType<Allocator>(var_order, val_order, std::move(vars));
}

/** Materialize and shuffle the variables of every `RANDOM` strategy.
 * A strategy with an empty set of variables ranges over all the `num_vars` variables of the store. */
template <class Allocator, class URBG>
void shuffle_random_strategies(SearchStrategies<Allocator>& strategies, lala::AType var_aty, int num_vars, URBG& g) {
  for(int i = 0; i < strategies.size(); ++i) {
    if(strategies[i].var_order == VariableOrder::RANDOM) {
      if(strategies[i].vars.empty()) {
        strategies[i].vars.reserve(num_vars);
        for(int j = 0; j < num_vars; ++j) {
          strategies[i].vars.push_back(lala::AVar{var_aty, j});
        }
      }
      std::shuffle(strategies[i].vars.data(), strategies[i].vars.data() + strategies[i].vars.size(), g);
    }
  }
}

/** The objective of the constraint network: the variable to optimize and the direction of the
 * optimization. An untyped variable means the problem is a satisfaction problem. */
struct Objective {
  lala::AVar x;
  // `true` for minimization, `false` for maximization.
  bool mode;

  CUDA Objective(): mode(true) {}
  Objective(const Objective&) = default;
  Objective(Objective&&) = default;
  Objective& operator=(const Objective&) = default;
  Objective& operator=(Objective&&) = default;
  CUDA Objective(lala::AVar x, bool mode): x(x), mode(mode) {}

  CUDA bool is_satisfaction() const { return x.is_untyped(); }
  CUDA bool is_optimization() const { return !is_satisfaction(); }
  CUDA bool is_minimization() const { return is_optimization() && mode; }
  CUDA bool is_maximization() const { return is_optimization() && !mode; }
  CUDA lala::AVar objective_var() const { return x; }
};

#endif
