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
#include "cuda_helper.hpp"

struct Statistics {
  int nodes;
  int fails;
  int sols;
  int best_bound;
  int depth_max;
  int exhaustive;

  CUDA Statistics(): nodes(0), fails(0), sols(0),
    best_bound(-1), depth_max(0), exhaustive(true) {}

  CUDA void join(const Statistics& other) {
    nodes += other.nodes;
    fails += other.fails;
    sols += other.sols;
    if(best_bound == -1) {
      best_bound = other.best_bound;
    }
    else if(other.best_bound != -1) {
      best_bound = min(best_bound, other.best_bound);
    }
    depth_max = max(depth_max, other.depth_max);
    exhaustive = exhaustive && other.exhaustive;
  }

  CUDA void print() {
    printf("nodes=%d\n", nodes);
    printf("fails=%d\n", fails);
    printf("solutions=%d\n", sols);
    if(best_bound != -1) {
      printf("objective=%d\n", best_bound);
      printf("satisfiability=true\n");
    }
    else if(exhaustive) {
      printf("satisfiability=false\n");
    }
    else {
      printf("satisfiability=unknown\n");
    }
    if(exhaustive) {
      printf("exhaustivity=true\n");
    }
    else {
      printf("exhaustivity=false\n");
    }
    printf("depthMax=%d\n", depth_max);
  }
};

struct GlobalStatistics {
  size_t variables;
  size_t constraints;
  int64_t duration;
  Statistics local;

  GlobalStatistics(size_t variables, size_t constraints, int64_t duration, Statistics local):
    variables(variables), constraints(constraints), duration(duration), local(local) {}

  void print() {
    std::cout << "variables=" << variables << std::endl;
    std::cout << "constraints=" << constraints << std::endl;
    local.print();
    std::cout << "solveTime=" << duration << std::endl;
  }

  // nodes, fails, solutions, depthmax, variables, constraints, satisfiability, exhaustivity, time, optimum
  void print_csv() {
    double duration_sec =  ((double) duration) / 1000.;
    printf("%d, %d, %d, %d, %d, %d, ", local.nodes, local.fails, local.sols, local.depth_max, variables, constraints);
    if(local.best_bound != -1) {
      printf("sat, ");
      if(local.exhaustive) { printf("true, "); }
      else { printf("false, "); }
      printf("%.2lf, %d\n", duration_sec, local.best_bound);
    }
    else if(local.exhaustive) {
      printf("unsat, true, %.2lf, unsat\n", duration_sec);
    }
    else {
      printf("unknown, false, %.2lf, none\n", duration_sec);
    }
  }
};

#endif
