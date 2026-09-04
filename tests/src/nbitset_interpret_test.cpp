// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "lala/universes/nbitset.hpp"
#include "abstract_testing.hpp"

using namespace lala;
using namespace battery;

using NBit = NBitset<128, battery::local_memory, unsigned long long>;

TEST(NBitsetTest, TellInterpretation) {
  VarEnv<standard_allocator> env;
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, -100);", NBit(-1), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, -1);", NBit(-1), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", NBit(0), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 10);", NBit(10), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 125);", NBit(125), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 126);", NBit(1000), env, true);

  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0); constraint int_eq(x, 10);", NBit::from_set({}), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint nbool_or(int_eq(x, 0), int_eq(x, 10));", NBit::from_set({0, 10}), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint set_in(x, {0, 10});", NBit::from_set({0, 10}), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint nbool_or(int_eq(x, -1), int_eq(x, 100), int_eq(x, 126));", NBit::from_set({-1, 100, 1000}), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint set_in(x, {-1, 1000});", NBit::from_set({-1, 1000}), env, true);

  expect_interpret_equal_to<IKind::TELL>("var 0..32: x;", NBit(0,32), env, false);
  expect_interpret_equal_to<IKind::TELL>("var {0,32}: x;", NBit::from_set({0,32}), env, false);

  expect_interpret_equal_to<IKind::TELL>("constraint int_ne(x, 0);", NBit(0).complement(), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ne(x, 0); constraint int_ne(x, 10);", NBit::from_set({0, 10}).complement(), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ne(x, -1);", NBit(), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ne(x, 1000);", NBit(), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ne(x, -1); constraint int_ne(x, 0);", NBit(0).complement(), env, true);

  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, -1);", NBit(), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, 0);", NBit(0,1000), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, 10);", NBit(10,1000), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_gt(x, 10);", NBit(11,1000), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, 126);", NBit(1000), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_ge(x, 1000);", NBit(1000), env, true);

  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, -2);", NBit(-100), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, -1);", NBit(-100), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 0);", NBit(-1,0), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 10);", NBit(-1, 10), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_lt(x, 10);", NBit(-1, 9), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 1000);", NBit(), env, true);

  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 1000); constraint int_ge(x, 0);", NBit(0, 200), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 100); constraint int_ge(x, 0);", NBit(0, 100), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 1); constraint int_ge(x, 0);", NBit(0, 1), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 0); constraint int_ge(x, 0);", NBit(0, 0), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint int_le(x, 100); constraint int_ge(x, -100);", NBit(-1, 100), env, true);
  expect_interpret_equal_to<IKind::TELL>("constraint nbool_or(int_le(x, 1), int_ge(x, 0));", NBit(), env, false);
  expect_interpret_equal_to<IKind::TELL>("constraint nbool_or(int_le(x, 0), int_ge(x, 0));", NBit(), env, false);
}
TEST(NBitsetTest, AskInterpretation) {
  VarEnv<standard_allocator> env;
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, -100);", NBit::bot(), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, -1);", NBit::bot(), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, 0);", NBit(0), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, 10);", NBit(10), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, 125);", NBit(125), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, 126);", NBit::bot(), env, true);

  expect_interpret_equal_to<IKind::ASK>("constraint int_eq(x, 0); constraint int_eq(x, 10);", NBit::bot(), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint nbool_or(int_eq(x, 0), int_eq(x, 10));", NBit::from_set({0, 10}), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint set_in(x, {0, 10});", NBit::from_set({0, 10}), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint nbool_or(int_eq(x, -1), int_eq(x, 100), int_eq(x, 126));", NBit(100), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint set_in(x, {-1, 1000});", NBit::bot(), env, true);

  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 0);", NBit(0).complement(), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 0); constraint int_ne(x, 10);", NBit::from_set({0, 10}).complement(), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, -1);", NBit(0,1000), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, 1000);", NBit(-1,125), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ne(x, -1); constraint int_ne(x, 0);", NBit(1, 1000), env, true);

  expect_interpret_equal_to<IKind::ASK>("constraint int_ge(x, -1);", NBit(0, 1000), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ge(x, 0);", NBit(0,1000), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ge(x, 10);", NBit(10,1000), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_gt(x, 10);", NBit(11,1000), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ge(x, 126);", NBit(126), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_ge(x, 1000);", NBit::bot(), env, true);

  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, -2);", NBit::bot(), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, -1);", NBit(-1), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 0);", NBit(-1,0), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 10);", NBit(-1, 10), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_lt(x, 10);", NBit(-1, 9), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 1000);", NBit(-1, 125), env, true);

  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 1000); constraint int_ge(x, 0);", NBit(0, 125), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 100); constraint int_ge(x, 0);", NBit(0, 100), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 1); constraint int_ge(x, 0);", NBit(0, 1), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 0); constraint int_ge(x, 0);", NBit(0, 0), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint int_le(x, 100); constraint int_ge(x, -100);", NBit(0, 100), env, true);
  expect_interpret_equal_to<IKind::ASK>("constraint nbool_or(int_le(x, 1), int_ge(x, 0));", NBit(), env, false);
  expect_interpret_equal_to<IKind::ASK>("constraint nbool_or(int_le(x, 0), int_ge(x, 0));", NBit(), env, false);
}
TEST(NBitsetTest, AbsTest) {
  generic_abs_test<NBit>();
}

