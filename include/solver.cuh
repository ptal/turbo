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

#ifndef TURBO_SOLVER_HPP
#define TURBO_SOLVER_HPP

#include "vstore.cuh"
#include "propagators.cuh"

#define OR_NODES 48
#define AND_NODES 256
#define SUBPROBLEMS_POWER 12 // 2^N

struct Configuration {
  int timeout;
  int and_nodes;
  int or_nodes;
  int subproblems_power;
  std::string problem_path;

  Configuration(): timeout(INT_MAX), and_nodes(AND_NODES),
    or_nodes(OR_NODES), subproblems_power(SUBPROBLEMS_POWER) {}
};

void solve(VStore* vstore, Constraints constraints, Var minimize_x, Configuration config);

#endif
