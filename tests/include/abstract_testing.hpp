// Copyright 2021 Pierre Talbot

#ifndef TURBO_ABSTRACT_TESTING_HPP
#define TURBO_ABSTRACT_TESTING_HPP

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "lala/logic/logic.hpp"
#include "lala/logic/ternarize.hpp"
#include "lala/lb.hpp"
#include "lala/ub.hpp"
#include "lala/flatzinc_parser.hpp"

#include "interpretation.hpp"

/** Helpers to test the interpretation of logical formulas in the abstract universes and abstract
 * domains. They live in Turbo because interpretation does: an abstract universe or domain is a
 * lattice and has no `interpret` member anymore. The purely lattice-theoretic helpers
 * (`bot_top_test`, `join_meet_generic_test`, ...) stay in lala-core, next to the lattices. */

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;

static LVar<standard_allocator> var_x = "x";
static LVar<standard_allocator> var_y = "y";

inline VarEnv<standard_allocator> env_with(const char* fzn) {
  VarEnv<standard_allocator> env;
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  IDiagnostics diagnostics;
  if(f->is(F::Seq) && f->sig() == AND) {
    for(int i = 0; i < f->seq().size(); ++i) {
      AVar avar;
      EXPECT_TRUE(interpret_in(f->seq(i), env, avar, diagnostics));
    }
  }
  else {
    AVar avar;
    EXPECT_TRUE(interpret_in(*f, env, avar, diagnostics));
  }
  return std::move(env);
}

/** Initialize an environment with a single integer variable named `x` in the abstract domain typed `0`. */
inline VarEnv<standard_allocator> env_with_x() {
  return env_with("var int: x :: abstract(0);");
}

template<IKind kind, class L>
void interpret_must_error(const char* fzn, VarEnv<standard_allocator> env = VarEnv<standard_allocator>{}) {
  static_assert(kind == IKind::TELL || lattice_properties<L>::is_abstract_universe);
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  IDiagnostics diagnostics;
  L value = make_top<L>(env);
  bool res;
  if constexpr(lattice_properties<L>::is_abstract_universe) {
    res = ginterpret_in<kind, true>(*f, env, value, diagnostics);
  }
  else {
    if constexpr(kind == IKind::TELL) {
      typename L::template tell_type<standard_allocator> tell;
      res = top_level_ginterpret_in<kind, true>(value, *f, env, tell, diagnostics);
    }
    else {
      typename L::template ask_type<standard_allocator> ask;
      res = top_level_ginterpret_in<kind, true>(value, *f, env, ask, diagnostics);
    }
  }
  if(res) {
    EXPECT_TRUE(false) << "The formula should not be interpretable: ";
    value.print();
    printf("\n");
  }
}

template<class L>
void both_interpret_must_error(const char* fzn, VarEnv<standard_allocator> env = VarEnv<standard_allocator>{}) {
  interpret_must_error<IKind::TELL, L>(fzn, env);
  interpret_must_error<IKind::ASK, L>(fzn, env);
}

template <IKind kind, bool ternarize_formula = false, class L>
void interpret_must_succeed(const char* fzn, L& value, VarEnv<standard_allocator>& env, bool has_warning = false) {
  static_assert(kind == IKind::TELL || lattice_properties<L>::is_abstract_universe);
  using F = TFormula<standard_allocator>;
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  if(ternarize_formula) {
    *f = ternarize(*f, env);
    f->print(); printf("\n");
  }
  *f = normalize(*f);
  IDiagnostics diagnostics;
  bool res;
  if constexpr(lattice_properties<L>::is_abstract_universe) {
    res = ginterpret_in<kind, true>(*f, env, value, diagnostics);
  }
  else {
    if constexpr(kind == IKind::TELL) {
      res = interpret_and_tell<true>(*f, env, value, diagnostics);
    }
    else {
      typename L::template ask_type<standard_allocator> ask;
      res = top_level_ginterpret_in<kind, true>(value, *f, env, ask, diagnostics);
    }
  }
  if(!res) {
    diagnostics.print();
    EXPECT_TRUE(false) << "The formula should be interpretable: " << fzn;
  }
  if(diagnostics.has_warning() && !has_warning) {
    diagnostics.print();
    EXPECT_TRUE(false) << "The formula generates a warning but should not: " << fzn;
  }
  EXPECT_EQ(diagnostics.has_warning(), has_warning);
}

template <class L, bool ternarize_formula = false, class Typing>
L create_and_interpret_and_type_and_tell(const char* fzn, VarEnv<standard_allocator>& env, Typing&& typing, bool has_warning = false) {
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  if(ternarize_formula) {
    *f = ternarize(*f, env);
    f->print(); printf("\n");
  }
  *f = normalize(*f);
  printf("normalized:\n"); f->print(); printf("\n");
  typing(*f);
  IDiagnostics diagnostics;
  auto value = create_and_interpret_and_tell<L, true>(*f, env, diagnostics);
  if(diagnostics.is_fatal()) {
    diagnostics.print();
  }
  EXPECT_FALSE(diagnostics.is_fatal());
  EXPECT_EQ(diagnostics.has_warning(), has_warning);
  EXPECT_TRUE(value.has_value());
  return std::move(value.value());
}

template <class L, bool ternarize_formula = false>
L create_and_interpret_and_tell(const char* fzn, VarEnv<standard_allocator>& env, bool has_warning = false) {
  return create_and_interpret_and_type_and_tell<L, ternarize_formula>(fzn, env, [](const F&){}, has_warning);
}

template <class L, bool ternarize_formula = false>
L create_and_interpret_and_tell(const char* fzn, bool has_warning = false) {
  VarEnv<standard_allocator> env;
  return create_and_interpret_and_tell<L, ternarize_formula>(fzn, env, has_warning);
}

template <IKind kind, class L>
void expect_interpret_equal_to(const char* fzn, const L& expect, VarEnv<standard_allocator> env = VarEnv<standard_allocator>{}, bool has_warning = false) {
  L value{L::top()};
  interpret_must_succeed<kind>(fzn, value, env, has_warning);
  EXPECT_EQ(value, expect);
}

/** When we expect an exact interpretation. */
template <class L>
void expect_both_interpret_equal_to(const char* fzn, const L& expect, const VarEnv<standard_allocator>& env = VarEnv<standard_allocator>{}, bool has_warning = false) {
  expect_interpret_equal_to<IKind::TELL>(fzn, expect, env, has_warning);
  expect_interpret_equal_to<IKind::ASK>(fzn, expect, env, has_warning);
}

template <class L, bool ternarize_formula = false>
bool interpret_and_ask(const char* fzn, L& value, VarEnv<standard_allocator>& env, bool has_warning = false) {
  auto f = parse_flatzinc_str<standard_allocator>(fzn);
  EXPECT_TRUE(f);
  if(ternarize_formula) {
    *f = ternarize(*f, env);
    f->print(); printf("\n");
  }
  *f = normalize(*f);
  printf("normalized:\n"); f->print(); printf("\n");
  IDiagnostics diagnostics;
  typename L::template ask_type<standard_allocator> ask;
  if(!ginterpret_in<IKind::ASK, true>(value, *f, env, ask, diagnostics)) {
    diagnostics.print();
    EXPECT_TRUE(false) << "The formula should be (ask-)interpretable: " << fzn;
  }
  EXPECT_EQ(diagnostics.has_warning(), has_warning);
  return value.ask(ask);
}

/** `true` must be interpreted by the top element and `false` by the bottom element.
 * This is the interpretation half of lala-core's `bot_top_test`. */
template <class A>
void bot_top_interpret_test() {
  A bot = A::bot();
  A top = A::top();
  expect_interpret_equal_to<IKind::TELL, A>("constraint true;", top);
  expect_interpret_equal_to<IKind::TELL, A>("constraint false;", bot);
  if constexpr(lattice_properties<A>::is_abstract_universe) {
    expect_interpret_equal_to<IKind::ASK, A>("constraint true;", top);
    expect_interpret_equal_to<IKind::ASK, A>("constraint false;", bot);
  }
}

template <class A>
void generic_abs_test() {
  A a;
  auto env = env_with_x();
  interpret_must_succeed<IKind::TELL>("constraint int_ge(x, 0);", a, env);
  A r{};
  r.abs(A::top());
  EXPECT_EQ(r, a);
}

/** Check that \f$ \llbracket . \rrbracket = \llbracket . \rrbracket \circ \rrbacket . \llbracket \circ \llbracket . \rrbracket \f$ */
template <class L, bool ternarize_formula = false>
void check_interpret_idempotence(const char* fzn) {
  using F = TFormula<standard_allocator>;
  VarEnv<standard_allocator> env1, env2;
  L value1 = create_and_interpret_and_tell<L, ternarize_formula>(fzn, env1);
  F f1 = deinterpret_in(value1, env1);
  f1.print(true);
  printf("\n");
  L value2 = make_top<L>(env2);
  IDiagnostics diagnostics;
  EXPECT_TRUE(interpret_and_tell(f1, env2, value2, diagnostics));
  EXPECT_EQ(value1, value2);
  F f2 = deinterpret_in(value2, env2);
  f2.print(true);
  printf("\n");
  EXPECT_EQ(f1, f2);
}

#endif
