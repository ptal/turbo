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
const int MAX_DEPTH_TREE = 8;

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

  __host__ PropagatorsStatus(const PropagatorsStatus& other) {
    n = other.n;
    CUDIE(cudaMallocManaged(&entailed, n*sizeof(bool)));
    CUDIE(cudaMallocManaged(&disentailed, n*sizeof(bool)));
    CUDIE(cudaMallocManaged(&idle, n*sizeof(bool)));
    // switch to memcpy on device
    for(int i = 0; i < n; ++i) {
      entailed[i] = other.entailed[i];
      disentailed[i] = other.disentailed[i];
      // NOTE: no need to copy "idle", must actually be initialized at false for the next node.
    }
  }

  void free() {
    CUDIE(cudaFree(entailed));
    CUDIE(cudaFree(disentailed));
    CUDIE(cudaFree(idle));
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
        printf("Propagator %d disentailed.\n", i);
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

  CUDA void wake_up_all() {
    for(int i = 0; i < n; ++i) {
      idle[i] = false;
    }
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
    //printf("Propagate %lu\n", id);
  }
};

// Select the variable with the smallest domain in the store.
CUDA Var first_fail(const VStore& vstore, Var* vars) {
  Var x = -1;
  int lowest_lb = limit_max();
  for(int k = 0; vars[k] != -1; ++k) {
    int i = vars[k];
    if (vstore[i].lb < lowest_lb && !vstore[i].is_assigned()) {
      x = i;
      lowest_lb = vstore[i].lb;
    }
  }
  assert(x != -1);
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

CUDA_GLOBAL void search(PropagatorsStatus* status, VStore* current, VStore* best_sol, Var minimize_x, Var* temporal_vars) {
  printf("starting search\n");
  BacktrackingFrame* stack = BacktrackingFrame[MAX_DEPTH_TREE];
  size_t stack_size = 0;
  Interval best_bound = {limit_min(), limit_max()};
  while (Exploring) {
    if (status->all_idle()) {
      bool all_entailed = status->all_entailed();
      bool any_disentailed = status->any_disentailed();
      if (current->all_assigned() && !all_entailed) {
        printf("entailment invariant inconsistent.\n");
        for(int i = 0; i < status->n; ++i) {
          if (!status->entailed[i]) {
            printf("not entailed %d\n", i);
          }
        }
      }
      if (!any_disentailed && current->is_top()) {
        printf("disentailment invariant inconsistent.\n");
      }
      if(all_entailed || any_disentailed) {
        if(any_disentailed) {
          printf("backtracking on failed node...\n");
        }
        else if(all_entailed) {
          best_bound = (*current)[minimize_x];
          best_bound.ub = best_bound.lb;
          printf("backtracking on solution...(bound = %d)\n", best_bound.ub);
          best_bound.lb = limit_min();
          *best_sol = *current;
        }
        // If nothing is left in the stack, we stop the search, it means we explored the full search tree.
        if(stack_size == 0) {
          Exploring = false;
        }
        else {
          printf("pop frame %d\n", stack_size - 1);
          BacktrackingFrame& frame = stack[stack_size - 1];
          --stack_size;
          printf("frame popped.\n");
          // Commit to the branch.
          printf("Right branching on %s = %d..%d\n", frame.vstore.name_of(frame.var), frame.itv.lb, frame.itv.ub);
          frame.vstore.update(frame.var, frame.itv);
          // Adjust the objective.
          frame.vstore.update(minimize_x, frame.vstore[minimize_x].join(best_bound));
          // Swap the current branch with the backtracked one.
          *current = frame.vstore;
        }
      }
      // At this stage, the current node is neither failed nor a solution yet.
      // We must branch.
      // The left branch is executed right away, and the right branch is pushed on the stack.
      else {
        printf("All IDLE, depth = %d\n", stack_size);
        current->print();
        // Copy the current store.
        printf("stack[..] = *current, %d\n", current->size());
        stack[stack_size].vstore = *current;
        printf("current->size(): %d, %d\n",current->size(), stack[stack_size].vstore.size());
        // Select the variable and its value on which we want to branch.
        Var x = first_fail(*current, temporal_vars);
        printf("Selected %s.\n", current->name_of(x));
        assign_lb(*current, x);
        printf("Correctly created left branch on %s.\n", current->name_of(x));
        // Create the right branch.
        stack[stack_size].var = x;
        stack[stack_size].itv = not_assign_lb(stack[stack_size].vstore, x);
        printf("Left branching on %s: %d..%d \\/ %d..%d\n",
          current->name_of(x), (*current)[x].lb, (*current)[x].ub,
          stack[stack_size].itv.lb, stack[stack_size].itv.ub);
        assert(stack_size < MAX_DEPTH_TREE);
        ++stack_size;
      }
      // Change the IDLE status of propagators.
      status->wake_up_all();
    }
  }
  delete[] stack;
  printf("stop search\n");
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
Engine<T>* launch(PropagatorsStatus* status, VStore* vstore, std::vector<T> &c, cudaStream_t s[PROP_OPS])
{
  // printf("launching %lu threads on stream %p\n", c.size(), s[0]);

  T* props;
  CUDIE(cudaMallocManaged(&props, c.size() * sizeof(T)));
  for (int i=0; i < c.size(); ++i) {
    props[i] = c[i];
  }

  Engine<T> *engine;
  CUDIE(cudaMallocManaged(&engine, sizeof(Engine<T>)));
  *engine = Engine<T>(status, vstore, props);

  propagate_k<T><<<1, c.size(), 0, s[0]>>>(engine);
  entail_k<T><<<1, c.size(), 0, s[1]>>>(engine);
  disentail_k<T><<<1, c.size(), 0, s[2]>>>(engine);
  CUDIE0();
  return engine;
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x)
{
  // std::cout << "Before propagation: " << std::endl;
  // vstore->print(var2name_raw);

  void* status_raw;
  CUDIE(cudaMallocManaged(&status_raw, sizeof(PropagatorsStatus)));
  PropagatorsStatus* status = new(status_raw) PropagatorsStatus(constraints.size());

  void* best_sol_raw;
  CUDIE(cudaMallocManaged(&best_sol_raw, sizeof(VStore)));
  VStore* best_sol = new(best_sol_raw) VStore(*vstore);

  constraints.print(*vstore);

  Var* temporal_vars = constraints.temporal_vars(vstore->size());

  cudaStream_t monitor;
  CUDIE(cudaStreamCreate(&monitor));
  cudaStream_t streams[PROPS_TYPE][PROP_OPS];
  for (int i=0; i < PROPS_TYPE; ++i) {
    for (int j=0; j < PROP_OPS; ++j) {
      CUDIE(cudaStreamCreate(&streams[i][j]));
    }
  }

  search<<<1,1,0,monitor>>>(status, vstore, best_sol, minimize_x, temporal_vars);
  CUDIE0();

  auto engines_0 = launch<TemporalProp>(status, vstore, constraints.temporal, streams[0]);
  CUDIE0();
  auto engines_1 = launch<ReifiedLogicalAnd>(status, vstore, constraints.reifiedLogicalAnd, streams[1]);
  CUDIE0();
  auto engines_2 = launch<LinearIneq>(status, vstore, constraints.linearIneq, streams[2]);
  CUDIE0();

  CUDIE(cudaDeviceSynchronize());

  if(best_sol->size() == 0) {
    printf("Could not find a solution.\n");
  }
  else {
    printf("Best bound found is %d.\n", (*best_sol)[minimize_x].lb);
    best_sol->free();
  }
  CUDIE(cudaFree(best_sol_raw));
  CUDIE(cudaFree(temporal_vars));

  status->free();
  CUDIE(cudaFree(status_raw));

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
