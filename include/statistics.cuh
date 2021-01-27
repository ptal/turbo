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

#include "cuda_helper.hpp"

struct Statistics {
  int nodes;
  int fails;
  int sols;
  int best_bound;
  int peak_depth;

  CUDA Statistics(): nodes(0), fails(0), sols(0),
    best_bound(-1), peak_depth(0) {}

  void print() {
    printf("nodes=%d\n", nodes);
    printf("fails=%d\n", fails);
    printf("solutions=%d\n", sols);
    if(best_bound != -1) {
      printf("objective=%d\n", best_bound);
    }
    else {
      printf("objective=unsat\n");
    }
    printf("peakDepth=%d\n", peak_depth);
  }
};

#endif
