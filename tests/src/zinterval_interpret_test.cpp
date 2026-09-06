// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "lala/zinterval.hpp"
#include "abstract_testing.hpp"

using namespace lala;
using namespace battery;

using zlb = LB<int>;
using zub = UB<int>;
using Itv = ZInterval<int>;

TEST(ZIntervalTest, NoInterpret) {
  VarEnv<standard_allocator> env = env_with_x();
  // An interval cannot represent a hole, so `x != 10` is not tell-interpretable.
  interpret_must_error<IKind::TELL, Itv>("constraint int_ne(x, 10);", env);
  interpret_must_error<IKind::ASK, Itv>("constraint float_eq(x, 1111111111.0000000000001);", env);
}

TEST(ZIntervalTest, ValidInterpret) {
  VarEnv<standard_allocator> env;
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 10);", Itv(10, 10), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 10);", Itv(zlb(11), zub::top()), env, false);
}

TEST(ZIntervalTest, BoundsAndTypes) {
  VarEnv<standard_allocator> env = env_with_x();
  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, 3);", Itv(zlb(3), zub::top()), env);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 7);", Itv(zlb::top(), zub(7)), env);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, 3); constraint int_le(x, 7);", Itv(3, 7), env);
  // Unlike a bare bound, an interval represents a Boolean domain.
  expect_interpret_equal_to<IKind::TELL>("var bool: x;", Itv(0, 1));
}

TEST(ZIntervalTest, SetMembership) {
  VarEnv<standard_allocator> env = env_with_x();
  expect_interpret_equal_to<IKind::TELL>("constraint set_in(x, 1..10);", Itv(1, 10), env);
}
