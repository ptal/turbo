// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "lala/interval.hpp"
#include "abstract_testing.hpp"

using namespace lala;
using namespace battery;

using zlb = local::ZLB;
using zub = local::ZUB;
using Itv = Interval<zlb>;

TEST(IntervalTest, NoInterpret) {
  VarEnv<standard_allocator> env = env_with_x();
  interpret_must_error<IKind::TELL, Itv>("constraint int_ne(x, 10);", env);
  interpret_must_error<IKind::ASK, Itv>("constraint float_eq(x, 1111111111.0000000000001);", env);
}
TEST(IntervalTest, ValidInterpret) {
  VarEnv<standard_allocator> env;
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 10);", Itv(10, 10), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 10);", Itv(zlb(11), zub::top()), env, false);
}
TEST(IntervalTest, AbsTest) {
  generic_abs_test<Itv>();
}

