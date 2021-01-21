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

CUDA_VAR volatile bool Exploring = 1;

const int PROPS_TYPE = 3;
const int PROP_OPS = 3;

struct PropagatorsStatus {
  bool* entailed;
  bool* disentailed;
  bool* idle;
  int n;
  PropagatorsStatus(int n) {
    this->n = n;
    CUDIE(cudaMallocManaged(&entailed, n*sizeof(bool)));
    CUDIE(cudaMallocManaged(&disentailed, n*sizeof(bool)));
    CUDIE(cudaMallocManaged(&idle, n*sizeof(bool)));
    for(int i = 0; i < n; ++i) {
      entailed[i] = false;
      disentailed[i] = false;
      idle[i] = false;
    }
  }

  __device__ PropagatorsStatus(PropagatorsStatus& other) {
    n = other.n;
    entailed = new bool[n];
    disentailed = new bool[n];
    idle = new bool[n];
    for(int i = 0; i < n; ++i) {
      entailed[i] = other.entailed[i];
      disentailed[i] = other.disentailed[i];
      // NOTE: no need to copy "idle", must actually be initialized at false for the next node.
    }
  }

  CUDA bool all(bool* array) {
    for(int i = 0; i < n; ++i) {
      if(!array[i]) {
        return false;
      }
    }
    return true;
  }

  CUDA bool all_entailed() {
    return all(entailed);
  }

  CUDA bool all_disentailed() {
    return all(disentailed);
  }

  CUDA bool all_idle() {
    return all(idle);
  }
};

template <typename T>
struct Engine {
  PropagatorsStatus* status;
  VStore* vstore;
  T* props;

  CUDA Engine(PropagatorsStatus* status, VStore* vstore, T* props)
    : status(status), vstore(vstore), props(props) {}

  CUDA inline void updateIsEntailed(size_t id) {
    T& p = props[id];
    status->entailed[p.uid] = p.is_entailed(*vstore);
  }

  CUDA inline void updateIsDisentailed(size_t id) {
    T& p = props[id];
    status->disentailed[p.uid] = p.is_disentailed(*vstore);
  }

  CUDA inline void propagate(size_t id) {
    T& p = props[id];
    status->idle[p.uid] = !p.propagate(*vstore);
  }
};

CUDA_GLOBAL void is_active_k(PropagatorsStatus* status) {
  printf("starting\n");
  while (true) {
    Exploring = status->all_idle();
    if (!Exploring) {
      printf("no activity\n");
      break;
    }
  }
}

template<typename T>
CUDA_GLOBAL void entail_k(Engine<T>* engine) {
  size_t id = threadIdx.x + blockIdx.x*blockDim.x;
  while (Exploring) {
    engine->updateIsEntailed(id);
  }
}

template<typename T>
CUDA_GLOBAL void disentail_k(Engine<T>* engine) {
  size_t id = threadIdx.x + blockIdx.x*blockDim.x;
  while (Exploring) {
    engine->updateIsDisentailed(id);
  }
}

<<<<<<< HEAD
template<typename ConstraintT>
ConstraintT* launch(std::vector<ConstraintT> &c, cudaStream_t s, VStore *vstore) {
  printf("launching %lu threads\n", c.size());
  ConstraintT *constraints;
  CUDIE(cudaMallocManaged(&constraints, c.size()*sizeof(ConstraintT)));
  for (int i=0; i<c.size(); ++i) {
    constraints[i] = c[i];
=======
template<typename T>
CUDA_GLOBAL void propagate_k(Engine<T>* engine) {
  size_t id = threadIdx.x + blockIdx.x*blockDim.x;
  while (Exploring) {
    engine->propagate(id);
>>>>>>> 5de0e4adc09dee2bf814afb8be4b3ed911fc973a
  }
}

// The status and vstore are shared among all propagators of all types.
// The UID inside a propagator, e.g., `TemporalProp::uid`, refers to the index of the propagator in the various arrays of `status`.
template<typename T>
Engine<T>* launch(PropagatorsStatus* status, VStore* vstore, std::vector<T> &c, cudaStream_t s[3])
{
  printf("launching %d threads on stream %d\n", c.size(), s[0]);

  T* props;
  CUDIE(cudaMallocManaged(&props, c.size() * sizeof(T)));
  for (int i=0; i < c.size(); ++i) {
    props[i] = c[i];
  }

  Engine<T>* engine;
  CUDIE(cudaMallocManaged(&engine, sizeof(Engine<T>)));
  *engine = Engine<T>(status, vstore, props);

  propagate_k<T><<<1, c.size(), 0, s[0]>>>(engine);
  entail_k<T><<<1, c.size(), 0, s[1]>>>(engine);
  disentail_k<T><<<1, c.size(), 0, s[2]>>>(engine);
  CUDIE0();
  return engine;
}

void solve(VStore* vstore, Constraints constraints, const char** var2name_raw)
{
  vstore->print(var2name_raw);

  PropagatorsStatus* status;
  CUDIE(cudaMallocManaged(&status, sizeof(PropagatorsStatus)));
  *status = PropagatorsStatus(constraints.size());

  cudaStream_t monitor;
  CUDIE(cudaStreamCreate(&monitor));
  cudaStream_t streams[PROPS_TYPE][PROP_OPS];
  for (int i=0; i < PROPS_TYPE; ++i) {
    for (int j=0; j < PROP_OPS; ++j) {
      CUDIE(cudaStreamCreate(&streams[i][j]));
    }
  }

  is_active_k<<<1,1,0,monitor>>>(status);
  CUDIE0();

  auto engines_0 = launch<TemporalProp>(status, vstore, constraints.temporal, streams[0]);
  auto engines_1 = launch<ReifiedLogicalAnd>(status, vstore, constraints.reifiedLogicalAnd, streams[1]);
  auto engines_2 = launch<LinearIneq>(status, vstore, constraints.linearIneq, streams[2]);

  printf("here\n");

  CUDIE(cudaDeviceSynchronize());

  printf("\n\nAfter propagation:\n");
  vstore->print(var2name_raw);

  CUDIE(cudaFree(engines_0->props));
  CUDIE(cudaFree(engines_0));
  CUDIE(cudaFree(engines_1->props));
  CUDIE(cudaFree(engines_1));
  CUDIE(cudaFree(engines_2->props));
  CUDIE(cudaFree(engines_2));

  CUDIE(cudaStreamDestroy(monitor));
  for (int i=0; i < PROPS_TYPE; ++i) {
    for (int j=0; j < PROP_OPS; ++j) {
      CUDIE(cudaStreamDestroy(streams[i][j]));
    }
  }
}
