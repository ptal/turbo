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

CUDA_VAR volatile int Act_cnt = 0;
CUDA_VAR volatile bool Exploring = 1;

__global__
void is_active_k() {
  printf("starting\n");
  while (1) {
    asm("nanosleep.u32 1000000000;");
    if (Act_cnt <= 0) {
      printf("no activity\n");
      Exploring = 0;  // temporairement
      break;
    } else {
      printf("active! (%d)\n", Act_cnt);
    }
  }
}

template<typename Constraint>
CUDA_GLOBAL void propagate_k(Constraint *c, VStore* vstore) {
  int ic = threadIdx.x + blockIdx.x*blockDim.x;
  bool worked, preworked = 0;
  bool first = 1;
  while (Exploring) {
    worked = c[ic].propagate(*vstore);
    if (first && !worked) { Act_cnt = Act_cnt -1; first = 0; }
    else if (first && worked) { first = 0; }
    else if (!preworked && worked) { Act_cnt = Act_cnt +1; }
    else if (preworked && !worked) { Act_cnt = Act_cnt -1; }
    preworked = worked;
    printf("t=%d, worked=%d, act=%d\n", ic, worked, Act_cnt);
  }
}

template<typename ConstraintT>
ConstraintT* launch(std::vector<ConstraintT> &c, cudaStream_t s, VStore *vstore) {
  printf("launching %d threads on stream %d\n", c.size(), s);
  ConstraintT *constraints;
  CUDIE(cudaMallocManaged(&constraints, c.size()*sizeof(ConstraintT)));
  for (int i=0; i<c.size(); ++i) {
    constraints[i] = c[i];
  }
  propagate_k<ConstraintT><<<1, c.size(), 0, s>>>(constraints, vstore);
  CUDIE0();
  return constraints;
}

void solve(VStore* vstore, Constraints constraints, const char** var2name_raw) {
  vstore->print(var2name_raw);

  cudaStream_t monitor;
  CUDIE(cudaStreamCreate(&monitor));
  const int NCT = 3;
  cudaStream_t sConstraint[NCT];
  for (int i=0; i<NCT; ++i) {
    CUDIE(cudaStreamCreate(&sConstraint[i]));
  }

  Act_cnt = constraints.xPlusYleqC.size() + constraints.reifiedLogicalAnd.size() + constraints.linearIneq.size();
  is_active_k<<<1,1,0,monitor>>>();
  CUDIE0();

  auto c0 = launch<XplusYleqC>(constraints.xPlusYleqC, sConstraint[0], vstore);
  auto c1 = launch<ReifiedLogicalAnd>(constraints.reifiedLogicalAnd, sConstraint[1], vstore);
  auto c2 = launch<LinearIneq>(constraints.linearIneq, sConstraint[2], vstore);
  
  printf("here\n");

  CUDIE(cudaDeviceSynchronize());

  printf("\n\nAfter propagation:\n");
  vstore->print(var2name_raw);

  CUDIE(cudaFree(c0));
  CUDIE(cudaFree(c1));
  CUDIE(cudaFree(c2));
  CUDIE(cudaStreamDestroy(monitor));
  for (int i=0; i<NCT; ++i) {
    CUDIE(cudaStreamDestroy(sConstraint[i]));
  }
}
