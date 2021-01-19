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

#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <new>

#include "solver.cuh"
#include "vstore.cuh"
#include "constraints.cuh"
#include "cuda_helper.hpp"

__global__
void is_active_k() {
  __shared__ uint prev;  // shared: avoids spillage to global (tbc)
  __shared__ uint curr;
  __shared__ bool inactive;
  inactive = false;
  while (1) {
    asm("nanosleep.u32 1000000000;");
    prev = curr;
    curr = Act_cnt;
    if (!inactive && curr == prev) {
      inactive = true;
      printf("no activity\n");
      break;  // for now
    } else {
      printf("active!\n");
    }
  }
}

template<typename Constraint>
CUDA_GLOBAL void propagate_k(Constraint c, VStore* vstore) {
  c.propagate(*vstore);
}

void solve(VStore* vstore, Constraints constraints, const char** var2name_raw) {
  // concurrent execution of monitoring and solving using streams:
  cudaStream_t monitor, solve;
  CUDIE(cudaStreamCreate(&monitor));
  CUDIE(cudaStreamCreate(&solve));

  vstore->print(var2name_raw);

  for(auto c : constraints.xPlusYleqC) {
    propagate_k<XplusYleqC><<<1,1,0,solve>>>(c, vstore);
  }
  for(auto c : constraints.reifiedLogicalAnd) {
    propagate_k<ReifiedLogicalAnd><<<1,1,0,solve>>>(c, vstore);
  }
  for(auto c : constraints.logicalOr) {
    propagate_k<LogicalOr><<<1,1,0,solve>>>(c, vstore);
  }
  for(auto c : constraints.linearIneq) {
    propagate_k<LinearIneq><<<1,1,0,solve>>>(c, vstore);
  }
  is_active_k<<<1,1,0,monitor>>>();
  CUDIE0();

  CUDIE(cudaDeviceSynchronize());

  printf("\n\nAfter propagation:\n");
  vstore->print(var2name_raw);

  CUDIE(cudaStreamDestroy(monitor));
  CUDIE(cudaStreamDestroy(solve));
}
