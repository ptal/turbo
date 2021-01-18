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

void solve() {
  // I. Declare the variable's domains.
  int nvar = 4;
  int x = 0;
  int y = 1;
  int z = 2;
  int b = 3;

  void* v;
  CUDIE(cudaMallocManaged(&v, sizeof(VStore)));
  VStore* vstore = new(v) VStore(nvar);

  vstore->dom(x, {0, 2});
  vstore->dom(y, {1, 3});
  vstore->dom(z, {2, 4});
  vstore->dom(b, {0,1});

  vstore->print_store();

  // II. Declare the constraints
  XplusYleqC c1(x,y,2);
  XplusYleqC c2(y,z,2);
  ReifiedLogicalAnd c3(b, c1, c2);

  // III. Solve the problem.
  //c3.propagate(*vstore);
  // concurrent execution of monitoring and solving using streams:
  cudaStream_t monitor, solve;
  CUDIE(cudaStreamCreate(&monitor));
  CUDIE(cudaStreamCreate(&solve));

  is_active_k<<<1,1,0,monitor>>>();
  propagate_k<ReifiedLogicalAnd><<<1,1,0,solve>>>(c3, vstore);
  CUDIE0();

  CUDIE(cudaDeviceSynchronize());

  vstore->print_store();

  CUDIE(cudaStreamDestroy(monitor));
  CUDIE(cudaStreamDestroy(solve));
  vstore->free();
  CUDIE(cudaFree(v));
}
