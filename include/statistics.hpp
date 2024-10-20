// Copyright 2021 Pierre Talbot

#ifndef TURBO_STATISTICS_HPP
#define TURBO_STATISTICS_HPP

#include <chrono>
#include <algorithm>
#include "battery/utility.hpp"
#include "battery/allocator.hpp"
#include "lala/logic/ast.hpp"

inline void print_memory_statistics(const char* key, size_t bytes) {
  printf("%% %s=%zu [", key, bytes);
  if(bytes < 1000 * 1000) {
    printf("%.2fKB", static_cast<double>(bytes) / 1000);
  }
  else if(bytes < 1000 * 1000 * 1000) {
    printf("%.2fMB", static_cast<double>(bytes) / (1000 * 1000));
  }
  else {
    printf("%.2fGB", static_cast<double>(bytes) / (1000 * 1000 * 1000));
  }
  printf("]\n");
}

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
  size_t num_blocks_done;
  size_t fixpoint_iterations;
  size_t eliminated_variables;
  size_t eliminated_formulas;
  double search_time;
  double propagation_time;

  CUDA Statistics(size_t variables, size_t constraints, bool optimization):
    variables(variables), constraints(constraints), optimization(optimization),
    duration(0), interpretation_duration(0),
    nodes(0), fails(0), solutions(0),
    depth_max(0), exhaustive(true),
    eps_solved_subproblems(0), eps_num_subproblems(1), eps_skipped_subproblems(0),
    num_blocks_done(0), fixpoint_iterations(0),
    eliminated_variables(0), eliminated_formulas(0),
    search_time(0.0), propagation_time(0.0)
    {}

  CUDA Statistics(): Statistics(0,0,false) {}
  Statistics(const Statistics&) = default;
  Statistics(Statistics&&) = default;

  CUDA void meet(const Statistics& other) {
    duration = battery::max(other.duration, duration);
    interpretation_duration = battery::max(other.interpretation_duration, interpretation_duration);
    nodes += other.nodes;
    fails += other.fails;
    solutions += other.solutions;
    depth_max = battery::max(depth_max, other.depth_max);
    exhaustive = exhaustive && other.exhaustive;
    eps_solved_subproblems += other.eps_solved_subproblems;
    eps_skipped_subproblems += other.eps_skipped_subproblems;
    num_blocks_done += other.num_blocks_done;
    fixpoint_iterations += other.fixpoint_iterations;
    search_time += other.search_time;
    propagation_time += other.propagation_time;
  }

private:
  CUDA void print_stat(const char* name, size_t value) const {
    printf("%%%%%%mzn-stat: %s=%" PRIu64 "\n", name, value);
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
    print_stat("num_solutions", solutions);
    print_stat("eps_num_subproblems", eps_num_subproblems);
    print_stat("eps_solved_subproblems", eps_solved_subproblems);
    print_stat("eps_skipped_subproblems", eps_skipped_subproblems);
    print_stat("num_blocks_done", num_blocks_done);
    print_stat("fixpoint_iterations", fixpoint_iterations);
    print_stat("eliminated_variables", eliminated_variables);
    print_stat("eliminated_formulas", eliminated_formulas);
#ifdef TURBO_PROFILE_MODE
    print_stat("search_time", search_time);
    print_stat("propagation_time", propagation_time);
#endif
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
