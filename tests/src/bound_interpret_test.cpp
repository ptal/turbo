// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"
#include "lala/lb.hpp"
#include "lala/ub.hpp"
#include "abstract_testing.hpp"

using namespace lala;
using namespace battery;

using zlb = LB<int>;
using zub = UB<int>;

/** A bound only accepts integer variables; an interval is needed to represent a Boolean domain. */
template<class B>
void interpret_types() {
  expect_interpret_equal_to<IKind::TELL>("var int: x;", B::top());
  both_interpret_must_error<B>("var real: x;");
  both_interpret_must_error<B>("var bool: x;");
}

TEST(BoundTest, InterpretTypes) {
  interpret_types<zlb>();
  interpret_types<zub>();
}

TEST(BoundTest, LBInterpretation) {
  expect_both_interpret_equal_to("constraint true;", zlb::top());
  expect_both_interpret_equal_to("constraint false;", zlb::bot());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_ge(x, 0);", zlb(0), env);
  expect_both_interpret_equal_to("constraint int_ge(x, -10);", zlb(-10), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 10);", zlb(10), env);

  expect_both_interpret_equal_to("constraint int_gt(x, 0);", zlb(1), env);
  expect_both_interpret_equal_to("constraint int_gt(x, -10);", zlb(-9), env);
  expect_both_interpret_equal_to("constraint int_gt(x, 10);", zlb(11), env);

  // Equality is exact in the tell language, and `x != k` under-approximates in the ask language.
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", zlb(0), env);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 1);", zlb(2), env);

  // A lower bound cannot represent an upper bound.
  interpret_must_error<IKind::TELL, zlb>("constraint int_le(x, 10);", env);
}

TEST(BoundTest, UBInterpretation) {
  expect_both_interpret_equal_to("constraint true;", zub::top());
  expect_both_interpret_equal_to("constraint false;", zub::bot());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_le(x, 0);", zub(0), env);
  expect_both_interpret_equal_to("constraint int_le(x, -10);", zub(-10), env);
  expect_both_interpret_equal_to("constraint int_le(x, 10);", zub(10), env);

  expect_both_interpret_equal_to("constraint int_lt(x, 0);", zub(-1), env);
  expect_both_interpret_equal_to("constraint int_lt(x, -10);", zub(-11), env);
  expect_both_interpret_equal_to("constraint int_lt(x, 10);", zub(9), env);

  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", zub(0), env);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 1);", zub(0), env);

  interpret_must_error<IKind::TELL, zub>("constraint int_ge(x, 10);", env);
}

TEST(BoundTest, SetMembership) {
  VarEnv<standard_allocator> env = env_with_x();
  // `x in S` keeps the bound of the union of the elements of `S`.
  expect_interpret_equal_to<IKind::TELL>("constraint set_in(x, 1..10);", zlb(1), env);
  expect_interpret_equal_to<IKind::TELL>("constraint set_in(x, 1..10);", zub(10), env);
}

TEST(BoundTest, ConjunctionDisjunction) {
  expect_both_interpret_equal_to("constraint true; constraint false;", zlb::bot());
  expect_both_interpret_equal_to("constraint false; constraint true;", zlb::bot());

  VarEnv<standard_allocator> env = env_with_x();
  // A conjunction is the meet of the bounds, i.e. the largest lower bound.
  expect_both_interpret_equal_to("constraint int_ge(x, 0); constraint int_ge(x, -2); constraint int_ge(x, 2);", zlb(2), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 2); constraint int_ge(x, -2); constraint int_ge(x, 0);", zlb(2), env);

  // A bound preserves joins, so a disjunction is interpreted exactly: the smallest lower bound.
  expect_both_interpret_equal_to("constraint bool_or(int_ge(x, 0), bool_or(int_ge(x, -2), int_ge(x, 2)), true);", zlb(-2), env);
}
