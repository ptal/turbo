// Copyright 2021 Pierre Talbot

#include "lala/vstore.hpp"
#include "lala/interval.hpp"
#include "abstract_testing.hpp"

using zlb = local::ZLB;
using zub = local::ZUB;
using Itv = Interval<zlb>;
using ZStore = VStore<zlb, standard_allocator>;
using IStore = VStore<Itv, standard_allocator>;

template<class L>
L interpret_and_test(const char* fzn, const vector<typename L::universe_type>& expect) {
  L s = create_and_interpret_and_tell<L>(fzn);
  EXPECT_EQ(s.vars(), expect.size());
  for(int i = 0; i < expect.size(); ++i) {
    EXPECT_EQ(s[i], expect[i]);
  }
  return std::move(s);
}

TEST(VStoreTest, InterpretationZStore) {
  interpret_must_error<IKind::TELL, ZStore>("constraint int_gt(x, 4);"); // (undeclared variable)
  interpret_and_test<ZStore>("var int: x; constraint int_gt(x, 4);", {zlb(5)});
  interpret_must_error<IKind::TELL, ZStore>("constraint int_gt(x, 4); var int: x;"); // (declaration after usage)
  interpret_must_error<IKind::TELL, ZStore>("var int: x; constraint int_lt(x, 4);"); // (x < 4 not supported in increasing integers abstract universe).
  interpret_and_test<ZStore>("var int: x; constraint int_gt(x, 4); constraint int_gt(x, 5);", {zlb(6)});
}

TEST(VStoreTest, InterpretationIStore) {
  IStore s1 = interpret_and_test<IStore>("var int: x; constraint int_gt(x, 4); constraint int_lt(x, 4);", {Itv(5, 3)});
  EXPECT_TRUE(s1.is_bot());
  IStore s2 = interpret_and_test<IStore>("var int: x; var int: y; constraint int_gt(x, 4); constraint int_lt(x, 4); constraint int_lt(y, 2);", {Itv::bot(), Itv(zlb::top(), zub(1))});
  EXPECT_TRUE(s2.is_bot());
  IStore s3 = interpret_and_test<IStore>("var int: x; constraint int_ge(x, 4); constraint int_le(x, 4);", {Itv(4, 4)});
  interpret_and_test<IStore>("var int: x; constraint int_eq(x, 4);", {Itv(4, 4)});
  IStore s4 = interpret_and_test<IStore>("var 1..10: x;", {Itv(1, 10)});
  interpret_and_test<IStore>("var 5..10: x; var -5..5: y;", {Itv(5, 10), Itv(-5, 5)});
  VarEnv<standard_allocator> env = env_with("var int: x :: abstract(0); var int: y :: abstract(0);");
  EXPECT_TRUE(interpret_and_ask("constraint int_eq(x, 4);", s1, env));
  EXPECT_TRUE(interpret_and_ask("constraint int_eq(x, 4);", s2, env));
  EXPECT_TRUE(interpret_and_ask("constraint int_eq(x, 4);", s3, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_eq(x, 4);", s4, env));
  EXPECT_TRUE(interpret_and_ask("constraint int_ne(x, 4);", s1, env));
  EXPECT_TRUE(interpret_and_ask("constraint int_ne(x, 4);", s2, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_ne(x, 4);", s3, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_ne(x, 4);", s4, env));
}

TEST(VStoreTest, AskOperation) {
  VarEnv<standard_allocator> env;
  ZStore store = create_and_interpret_and_tell<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 1);", env);
  EXPECT_TRUE(interpret_and_ask("constraint int_ge(x, 0); constraint int_ge(y, 1);", store, env));
  EXPECT_TRUE(interpret_and_ask("constraint int_ge(y, -1);", store, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_ge(x, 0); constraint int_ge(y, 2);", store, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_ge(x, 10); constraint int_ge(y, 2);", store, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_ge(x, 10);", store, env));
}

TEST(VStoreTest, AskOperationInfiniteDom) {
  VarEnv<standard_allocator> env;
  IStore store = create_and_interpret_and_tell<IStore>("var int: x;", env);
  EXPECT_FALSE(interpret_and_ask("constraint int_le(x, 5);", store, env));
  EXPECT_FALSE(interpret_and_ask("constraint int_gt(x, 5);", store, env));
}

TEST(VStoreTest, Idempotence) {
  check_interpret_idempotence<ZStore>("var int: x; var int: y; constraint int_ge(x, 1); constraint int_ge(y, 10);");
  check_interpret_idempotence<IStore>("array[1..10] of var int: x;");
  check_interpret_idempotence<IStore>("array[1..10] of var 1..10: x;");
}

TEST(VStoreTest, BotTopInterpretation) {
  bot_top_interpret_test<ZStore>();
}
