// Copyright 2022 Pierre Talbot

#include <gtest/gtest.h>
#include "lala/logic/logic.hpp"
#include "lala/universes/flat_universe.hpp"
#include "battery/allocator.hpp"
#include "abstract_testing.hpp"

using namespace lala;
using namespace battery;

using ZF = local::ZFlat;


TEST(FlatUniverseTest, InterpretIntegerType) {
  std::cout << "Z ";
  expect_interpret_equal_to<IKind::TELL>("var int: x;", ZF::top());
  std::cout << "F ";
  expect_interpret_equal_to<IKind::TELL>("var int: x;", local::FFlat::top(), VarEnv<standard_allocator>{}, true);
}

TEST(FlatUniverseTest, InterpretRealType) {
  std::cout << "Z ";
  interpret_must_error<IKind::TELL, ZF>("var real: x;");
  std::cout << "F ";
  expect_interpret_equal_to<IKind::TELL>("var real: x;", local::FFlat::top());
}

TEST(FlatUniverseTest, InterpretBoolType) {
  std::cout << "Z ";
  interpret_must_error<IKind::TELL, ZF>("var bool: x;");
  std::cout << "F ";
  interpret_must_error<IKind::TELL, local::FFlat>("var bool: x;");
}

TEST(FlatUniverseTest, ZFlatInterpretation) {
  expect_both_interpret_equal_to("constraint true;", ZF::top());
  expect_both_interpret_equal_to("constraint false;", ZF::bot());

  VarEnv<standard_allocator> env = env_with_x();
  expect_interpret_equal_to<IKind::TELL>("constraint int_eq(x, 0);", ZF(0), env);
  both_interpret_must_error<ZF>("constraint int_ne(x, 1);", env);
}
