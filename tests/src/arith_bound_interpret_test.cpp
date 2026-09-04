// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"
#include "lala/universes/arith_bound.hpp"
#include "lala/universes/flat_universe.hpp"
#include "abstract_testing.hpp"

using namespace lala;
using namespace battery;


template<class Z, class F>
void interpret_integer_type() {
  std::cout << "Z ";
  expect_interpret_equal_to<IKind::TELL>("var int: x;", Z::top());
  std::cout << "F ";
  expect_interpret_equal_to<IKind::TELL>("var int: x;", F::top(), VarEnv<standard_allocator>{}, true);
}

TEST(ArithBoundTest, InterpretIntegerType) {
  interpret_integer_type<local::ZLB, local::FLB>();
  interpret_integer_type<local::ZUB, local::FUB>();
}

template<class Z, class F>
void interpret_real_type() {
  std::cout << "Z ";
  both_interpret_must_error<Z>("var real: x;");
  std::cout << "F ";
  expect_interpret_equal_to<IKind::TELL>("var real: x;", F::top());
}

TEST(ArithBoundTest, InterpretRealType) {
  interpret_real_type<local::ZLB, local::FLB>();
  interpret_real_type<local::ZUB, local::FUB>();
}

template<class Z, class F>
void interpret_bool_type() {
  std::cout << "Z ";
  both_interpret_must_error<Z>("var bool: x;");
  std::cout << "F ";
  both_interpret_must_error<F>("var bool: x;");
}

TEST(ArithBoundTest, InterpretBoolType) {
  interpret_bool_type<local::ZLB, local::FLB>();
  interpret_bool_type<local::ZUB, local::FUB>();
}

TEST(ArithBoundTest, ZLBInterpretation) {
  using zlb = local::ZLB;
  expect_both_interpret_equal_to("constraint true;", zlb::top());
  expect_both_interpret_equal_to("constraint false;", zlb::bot());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_ge(x, 0);", zlb(0), env);
  expect_both_interpret_equal_to("constraint int_ge(x, -10);", zlb(-10), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 10);", zlb(10), env);

  expect_both_interpret_equal_to("constraint int_gt(x, 0);", zlb(1), env);
  expect_both_interpret_equal_to("constraint int_gt(x, -10);", zlb(-9), env);
  expect_both_interpret_equal_to("constraint int_gt(x, 10);", zlb(11), env);

  interpret_must_error<IKind::ASK, zlb>("constraint int_eq(x, 0);", env);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", zlb(0), env);

  interpret_must_error<IKind::TELL, zlb>("constraint int_ne(x, 1);", env);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 1);", zlb(2), env);

  both_interpret_must_error<zlb>("constraint int_le(x, 10);", env);
  both_interpret_must_error<zlb>("constraint int_lt(x, 10);", env);
}

TEST(ArithBoundTest, ZUBInterpretation) {
  using zub = local::ZUB;
  expect_both_interpret_equal_to("constraint true;", zub::top());
  expect_both_interpret_equal_to("constraint false;", zub::bot());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_le(x, 0);", zub(0), env);
  expect_both_interpret_equal_to("constraint int_le(x, -10);", zub(-10), env);
  expect_both_interpret_equal_to("constraint int_le(x, 10);", zub(10), env);

  expect_both_interpret_equal_to("constraint int_lt(x, 0);", zub(-1), env);
  expect_both_interpret_equal_to("constraint int_lt(x, -10);", zub(-11), env);
  expect_both_interpret_equal_to("constraint int_lt(x, 10);", zub(9), env);

  interpret_must_error<IKind::ASK, zub>("constraint int_eq(x, 0);", env);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", zub(0), env);

  interpret_must_error<IKind::TELL, zub>("constraint int_ne(x, 1);", env);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 1);", zub(0), env);

  both_interpret_must_error<zub>("constraint int_ge(x, 10);", env);
  both_interpret_must_error<zub>("constraint int_gt(x, 10);", env);
}

TEST(ArithBoundTest, ConjunctionDisjunction) {
  using zlb = local::ZLB;
  expect_both_interpret_equal_to("constraint true; constraint false;", zlb::bot());
  expect_both_interpret_equal_to("constraint false; constraint true;", zlb::bot());

  VarEnv<standard_allocator> env = env_with_x();
  expect_both_interpret_equal_to("constraint int_ge(x, 0); constraint int_ge(x, -2); constraint int_ge(x, 2);", zlb(2), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 0); constraint int_ge(x, 2); constraint int_ge(x, -2);", zlb(2), env);
  expect_both_interpret_equal_to("constraint int_ge(x, 2); constraint int_ge(x, -2); constraint int_ge(x, 0);", zlb(2), env);

  expect_both_interpret_equal_to("constraint bool_or(int_ge(x, 0), bool_or(int_ge(x, -2), int_ge(x, 2)), true);", zlb(-2), env);
  expect_both_interpret_equal_to("constraint bool_or(int_ge(x, 0), bool_or(int_ge(x, -2), int_ge(x, 2)), true);", zlb(-2), env);
  expect_both_interpret_equal_to("constraint bool_or(int_ge(x, 0), bool_or(int_ge(x, -2), int_ge(x, 2)), true);", zlb(-2), env);
}

