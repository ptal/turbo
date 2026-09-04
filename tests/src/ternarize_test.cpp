#include <gtest/gtest.h>
#include "battery/allocator.hpp"
#include "lala/logic/logic.hpp"
#include "lala/logic/ternarize.hpp"
#include "lala/flatzinc_parser.hpp"
#include "lala/vstore.hpp"
#include "lala/simplifier.hpp"
#include "lala/interval.hpp"
#include "lala/fixpoint.hpp"
#include "abstract_testing.hpp"
#include <optional>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

using namespace lala;
using namespace battery;
namespace fs = std::filesystem;
using Itv = Interval<local::ZLB>;
using IStore = VStore<Itv, standard_allocator>;

template <class F>
bool contains(const F &f, const F &must_contain)
{
  F reversed = must_contain;
  if(must_contain.is_binary() && must_contain.sig() == EQ) {
    reversed = F::make_binary(must_contain.seq(1), EQ, must_contain.seq(0));
  }
  for (int i = 0; i < f.seq().size(); ++i)
  {
    if (f.seq(i) == must_contain || f.seq(i) == reversed)
    {
      return true;
    }
  }
  return false;
}

bool is_PIR_symbol(Sig &sig)
{
  return sig == ADD || sig == MUL || is_division(sig) || is_modulo(sig) || sig == EQ || sig == LEQ || sig == MAX || sig == MIN;
}

void test_ternarize(
  std::pair<std::string, std::optional<std::string>> formulas)
{
  VarEnv<standard_allocator> env;
  // Cannot be interpreted in IStore, but after applying Simplifier, it can be interpreted.
  auto f1 = *parse_flatzinc<standard_allocator>(std::get<0>(formulas));
  if (f1.seq().size() < 100)
  {
    printf("\n Orignal formula : ");
    f1.print(false);
    printf("\n");
  }

  std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
  auto ternarized = ternarize(f1);
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
  if (ternarized.seq().size() < 100)
  {
    printf("Ternarized formula : ");
    ternarized.print(false);
    printf("\n");
  }

  printf("Time difference = %ld\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

  printf("NB variables original formula : %zu\n", num_quantified_vars(f1));
  printf("NB constraints original formula : %zu\n", f1.seq().size() - num_quantified_vars(f1));
  printf("NB variables ternarized formula : %zu\n", num_quantified_vars(ternarized));
  printf("NB constraints ternarized formula : %zu\n", ternarized.seq().size() - num_quantified_vars(ternarized));

  if (std::get<1>(formulas).has_value())
  {
    auto f2 = *parse_flatzinc<standard_allocator>(std::get<1>(formulas).value());
    if (f2.seq().size() < 100)
    {
      printf("\n Expected formula : ");
      f2.print(false);
      printf("\n");
    }

    if (std::get<1>(formulas).has_value())
    {
      EXPECT_EQ(ternarized.seq().size(), f2.seq().size());
      for (int i = 0; i < ternarized.seq().size(); ++i)
      {
        if(!contains(f2, ternarized.seq(i))) {
          ternarized.seq(i).print(); printf(" not found\n");
        }
        EXPECT_TRUE(contains(f2, ternarized.seq(i)));
      }
    }
  }
  else {
    EXPECT_TRUE(ternarized.seq().size() > 0); // it is a sequence
  }

  for (int i = 0; i < ternarized.seq().size(); ++i) {
    auto f = ternarized.seq(i);
    // test the form of the formula
    if (!::lala::impl::Ternarizer<F, VarEnv<standard_allocator>>::is_extended_ternary_form(f)) {
      if(!(f.is(F::ESeq)
        || (f.is(F::Seq) && (f.sig() == MINIMIZE || f.sig() == MAXIMIZE || f.sig() == IN))))
      {
        f.print(); printf("\n");
      }
      EXPECT_TRUE(f.is(F::ESeq)
        || (f.is(F::Seq) && (f.sig() == MINIMIZE || f.sig() == MAXIMIZE || f.sig() == IN)));
    }
  }
}

std::vector<std::pair<std::string, std::optional<std::string>>> get_test_cases(const std::string &test_directory)
{
  std::vector<std::pair<std::string, std::optional<std::string>>> test_cases;
  std::regex test_file_pattern(R"(.*\.fzn)");

  std::filesystem::path current_path = std::filesystem::current_path();
  std::filesystem::path back("..");

  for (const auto &entry : fs::directory_iterator(current_path / back / back / test_directory))
  {
    if (entry.is_regular_file())
    {
      const std::string filename = entry.path().filename().string();
      // Check if the file matches the format t<n>.input.fzn
      if (std::regex_match(filename, test_file_pattern) && filename.find(".output.fzn") == std::string::npos)
      {
        std::string input_file = entry.path().string();
        // Replace ".input.fzn" by ".output.fzn" to find the corresponding expected output file.
        std::string output_file = input_file;
        if (output_file.find(".input.fzn") != std::string::npos) {
          output_file.replace(output_file.find(".input.fzn"), 10, ".output.fzn");
        }
        else {
          output_file = "";
        }
        if (fs::exists(output_file)) {
          test_cases.emplace_back(input_file, output_file);
        }
        else {
          test_cases.emplace_back(input_file, std::nullopt);
        }
      }
    }
  }
  return test_cases;
}

class TernarizeTest : public ::testing::TestWithParam<std::pair<std::string, std::optional<std::string>>>
{
};

std::string testNameGenerator(const ::testing::TestParamInfo<std::pair<std::string, std::optional<std::string>>> &info)
{
  // Extraire le nom du fichier sans l'extension pour un nom plus lisible
  std::string filename = fs::path(std::get<0>(info.param)).stem().stem().string();
  filename.erase(std::remove(filename.begin(), filename.end(), '-'), filename.end());
  filename.erase(std::remove(filename.begin(), filename.end(), '_'), filename.end());
  filename.erase(std::remove(filename.begin(), filename.end(), '.'), filename.end());
  return filename;
}

TEST_P(TernarizeTest, RunTest)
{
  auto test_case = GetParam();
  test_ternarize(test_case);
}

INSTANTIATE_TEST_SUITE_P(
  DomainTest,
  TernarizeTest,
  ::testing::ValuesIn(get_test_cases("tests/data/fzn/domain")),
  testNameGenerator
);

INSTANTIATE_TEST_SUITE_P(
  BinaryConstraintTest,
  TernarizeTest,
  ::testing::ValuesIn(get_test_cases("tests/data/fzn/binary")),
  testNameGenerator
);

INSTANTIATE_TEST_SUITE_P(
  NaryTest,
  TernarizeTest,
  ::testing::ValuesIn(get_test_cases("tests/data/fzn/nary")),
  testNameGenerator
);

INSTANTIATE_TEST_SUITE_P(
  WordpressInstancesTest,
  TernarizeTest,
  ::testing::ValuesIn(get_test_cases("tests/data/fzn/wordpress")),
  testNameGenerator
);

// INSTANTIATE_TEST_SUITE_P(
//     MZN2022InstancesTest,
//     TernarizeTest,
//     ::testing::ValuesIn(get_test_cases("../turbo/benchmarks/data/mzn2022/")),
//     testNameGenerator
// );

// INSTANTIATE_TEST_SUITE_P(
//     EasyInstancesTest,
//     TernarizeTest,
//     ::testing::ValuesIn(get_test_cases("../turbo/benchmarks/data/")),
//     testNameGenerator
// );