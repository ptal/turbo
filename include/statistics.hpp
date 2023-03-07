// Copyright 2021 Pierre Talbot, Frédéric Pinel

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <chrono>
#include <algorithm>
#include "utility.hpp"

struct Statistics {
  size_t variables;
  size_t constraints;
  bool optimization;
  int64_t duration;
  int64_t interpretation_duration;
  int nodes;
  int fails;
  int solutions;
  int best_bound;
  int depth_max;
  int exhaustive;

  CUDA Statistics(size_t variables, size_t constraints, bool optimization):
    variables(variables), constraints(constraints), optimization(optimization),
    duration(0), interpretation_duration(0),
    nodes(0), fails(0), solutions(0),
    best_bound(-1), depth_max(0), exhaustive(true) {}

  CUDA Statistics(): Statistics(0,0,false) {}
  Statistics(const Statistics&) = default;
  Statistics(Statistics&&) = default;

  /** Reset the statistics of a subtree of the search tree, but keep the global statistics: max_depth, best_bound, variables, constraints, optimization, exhaustive. */
  CUDA void reset_local_stats() {
    duration = 0;
    interpretation_duration = 0;
    nodes = 0;
    fails = 0;
    solutions = 0;
  }

  CUDA void join(const Statistics& other) {
    duration += other.duration;
    interpretation_duration += other.interpretation_duration;
    nodes += other.nodes;
    fails += other.fails;
    solutions += other.solutions;
    if(best_bound == -1) {
      best_bound = other.best_bound;
    }
    else if(other.best_bound != -1) {
      best_bound = std::min(best_bound, other.best_bound);
    }
    depth_max = std::max(depth_max, other.depth_max);
    exhaustive = exhaustive && other.exhaustive;
  }

  static void print_csv_header() {
    printf("nodes, fails, solutions, depthmax, variables, constraints, satisfiability, exhaustivity, time, optimum\n");
  }

  void print_csv() const {
    printf("%d, %d, %d, %d, %ld, %ld, ", nodes, fails, solutions, depth_max, variables, constraints);
    if(best_bound != -1) {
      printf("sat, ");
      if(exhaustive) { printf("true, "); }
      else { printf("false, "); }
      printf("%.2lf, %d\n", to_sec(duration), best_bound);
    }
    else if(exhaustive) {
      printf("unsat, true, %.2lf, unsat\n", to_sec(duration));
    }
    else {
      printf("unknown, false, %.2lf, none\n", to_sec(duration));
    }
  }

private:
  CUDA void print_stat(const char* name, int value) const {
    printf("%%%%%%mzn-stat: %s=%d\n", name, value);
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
    print_stat("variables", (int)variables);
    print_stat("propagators", (int)constraints);
    print_stat("peakDepth", depth_max);
    if(best_bound != -1) {
      print_stat("objective", best_bound);
    }
    print_stat("initTime", to_sec(interpretation_duration));
    print_stat("solveTime", to_sec(duration));
    print_stat("solutions", solutions);
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
