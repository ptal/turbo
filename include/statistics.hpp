// Copyright 2021 Pierre Talbot

#ifndef TURBO_STATISTICS_HPP
#define TURBO_STATISTICS_HPP

#include <chrono>
#include <algorithm>
#include "battery/utility.hpp"
#include "battery/allocator.hpp"
#include "lala/logic/ast.hpp"

struct Statistics {
  size_t variables;
  size_t constraints;
  bool optimization;
  int64_t duration;
  int64_t interpretation_duration;
  size_t nodes;
  size_t fails;
  size_t solutions;
  size_t depth_max;
  size_t exhaustive;
  size_t eps_num_subproblems;
  size_t eps_solved_subproblems;
  size_t eps_skipped_subproblems;

  CUDA Statistics(size_t variables, size_t constraints, bool optimization):
    variables(variables), constraints(constraints), optimization(optimization),
    duration(0), interpretation_duration(0),
    nodes(0), fails(0), solutions(0),
    depth_max(0), exhaustive(true),
    eps_solved_subproblems(0), eps_num_subproblems(1), eps_skipped_subproblems(0) {}

  CUDA Statistics(): Statistics(0,0,false) {}
  Statistics(const Statistics&) = default;
  Statistics(Statistics&&) = default;

  CUDA void join(const Statistics& other) {
    duration = battery::max(other.duration, duration);
    interpretation_duration = battery::max(other.interpretation_duration, interpretation_duration);
    nodes += other.nodes;
    fails += other.fails;
    solutions += other.solutions;
    depth_max = battery::max(depth_max, other.depth_max);
    exhaustive = exhaustive && other.exhaustive;
    eps_solved_subproblems += other.eps_solved_subproblems;
    eps_skipped_subproblems += other.eps_skipped_subproblems;
  }

private:
  CUDA void print_stat(const char* name, size_t value) const {
    printf("%%%%%%mzn-stat: %s=%lu\n", name, value);
  }

  CUDA void print_stat(const char* name, double value) const {
    printf("%%%%%%mzn-stat: %s=%lf\n", name, value);
  }

  CUDA double to_sec(int64_t dur) const {
    return ((double) dur / 1000.);
  }

public:
  CUDA void print_mzn_statistics() const {
    print_stat("nodes", nodes);
    print_stat("failures", fails);
    print_stat("variables", variables);
    print_stat("propagators", constraints);
    print_stat("peakDepth", depth_max);
    print_stat("initTime", to_sec(interpretation_duration));
    print_stat("solveTime", to_sec(duration));
    print_stat("solutions", solutions);
    print_stat("eps_num_subproblems", eps_num_subproblems);
    print_stat("eps_solved_subproblems", eps_solved_subproblems);
    print_stat("eps_skipped_subproblems", eps_skipped_subproblems);
  }

  CUDA void print_mzn_end_stats() const {
    printf("%%%%%%mzn-stat-end\n");
  }

  CUDA void print_mzn_objective(const auto& obj, bool is_minimization) const {
    printf("%%%%%%mzn-stat: objective=");
    if(is_minimization) {
      obj.lb().template deinterpret<lala::TFormula<battery::standard_allocator>>().print(false);
    }
    else {
      obj.ub().template deinterpret<lala::TFormula<battery::standard_allocator>>().print(false);
    }
    printf("\n");
  }

  CUDA void print_mzn_separator() const {
    printf("----------\n");
  }

  CUDA void print_mzn_final_separator() const {
    if(solutions > 0) {
      if(exhaustive) {
        printf("==========\n");
      }
    }
    else {
      assert(solutions == 0);
      if(exhaustive) {
        printf("=====UNSATISFIABLE=====\n");
      }
      else if(optimization) {
        printf("=====UNBOUNDED=====\n");
      }
      else {
        printf("=====UNKNOWN=====\n");
      }
    }
  }
};

#endif
