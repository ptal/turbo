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

CUDA_VAR bool Exploring = true;

const int PROPS_TYPE = 3;
const int PROP_OPS = 3;
const int MAX_DEPTH_TREE = 100;

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


  CUDA bool any(bool* array) {
    for(int i = 0; i < n; ++i) {
      if(array[i]) {
        return true;
      }
    }
    return false;
  }

  CUDA bool all_entailed() {
    return all(entailed);
  }

  CUDA bool any_disentailed() {
    return any(disentailed);
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

// Select the variable with the smallest domain in the store.
CUDA Var first_fail(VStore& vstore) {
  Var x = 0;
  int lowest_lb = vstore[x].lb;
  for(int i = 1; i < vstore.size(); ++i) {
    if (vstore[i].lb < lowest_lb) {
      x = i;
      lowest_lb = vstore[i].lb;
    }
  }
  return x;
}

CUDA void assign_lb(VStore& vstore, Var x) {
  vstore.update(x, {vstore[x].lb, vstore[x].lb});
}

CUDA Interval not_assign_lb(VStore& vstore, Var x) {
  return {vstore[x].lb + 1, vstore[x].ub};
}

struct BacktrackingFrame {
  VStore vstore;
  Var var;
  Interval itv;
};

CUDA_GLOBAL void brancher(PropagatorsStatus* status, VStore* current) {
  printf("starting brancher\n");
  BacktrackingFrame* stack = new BacktrackingFrame[MAX_DEPTH_TREE];
  size_t stack_size = 0;
  while (Exploring) {
    if (status->all_idle()) {
      BacktrackingFrame frame;
      frame.var = first_fail(*current);
      frame.itv = not_assign_lb(*current, frame.var);
      frame.vstore = *current;
      stack[stack_size] = std::move(frame);
      ++stack_size;
      assert(stack_size < MAX_DEPTH_TREE);
      assign_lb(*current, frame.var);
    }
    else if(status->any_disentailed()) {
      printf("backtracking on failed node...\n");
      if(stack_size == 0) {
        Exploring = false;
      }
      else {
        Exploring = false;
      }
    }
    else if(status->all_entailed()) {
      printf("backtracking on solution...\n");
      if(stack_size == 0) {
        Exploring = false;
      }
      else {
        Exploring = false;
      }
    }
  }
  printf("stop brancher\n");
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

template<typename T>
CUDA_GLOBAL void propagate_k(Engine<T>* engine) {
  size_t id = threadIdx.x + blockIdx.x*blockDim.x;
  while (Exploring) {
    engine->propagate(id);
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

  brancher<<<1,1,0,monitor>>>(status, vstore);
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
