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
#include "status.cuh"

CUDA_VAR bool Exploring = true;

const int PROPS_TYPE = 3;
const int MAX_DEPTH_TREE = 100;

template <typename T>
struct SharedData {
  PropagatorsStatus* pstatus;
  VStore* vstore;
  T* props;

  CUDA SharedData(PropagatorsStatus* pstatus, VStore* vstore, T* props)
    : pstatus(pstatus), vstore(vstore), props(props) {}
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

CUDA_GLOBAL void search(PropagatorsStatus* pstatus, VStore* current, VStore* best_sol, Var minimize_x, Var* temporal_vars) {
  printf("starting search\n");
  //BacktrackingFrame* stack = new BacktrackingFrame[MAX_DEPTH_TREE];
  BacktrackingFrame* stack;
  cudaError_t rc = cudaMalloc(&stack, sizeof(BacktrackingFrame)*MAX_DEPTH_TREE);
  if (rc != cudaSuccess) {
	  printf("Could not allocate the stack (error = %d)\n", rc);
	  return;
  }
  for (int i=0; i<MAX_DEPTH_TREE; ++i) {
		stack[i] = BacktrackingFrame();
	}
  size_t stack_size = 0;
  Interval best_bound = {limit_min(), limit_max()};
  while (Exploring) {
    Status res = pstatus->join();
    if (res != UNKNOWN) {
      if (current->all_assigned() && res != ENTAILED) {
        printf("entailment invariant inconsistent (status = %s).\n",
          string_of_status(res));
        for(int i = 0; i < pstatus->size(); ++i) {
          if (pstatus->of(i) != ENTAILED) {
            printf("not entailed %d\n", i);
          }
        }
      }
      if (res != DISENTAILED && current->is_top()) {
        printf("disentailment invariant inconsistent.\n");
      }
      if(res != IDLE) {
        if(res == DISENTAILED) {
          printf("backtracking on failed node...\n");
        }
        else if(res == ENTAILED) {
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
          LOG(printf("pop frame %d\n", stack_size - 1);)
          BacktrackingFrame& frame = stack[stack_size - 1];
          --stack_size;
          // Commit to the branch.
          LOG(printf("Right branching on %s = %d..%d\n", frame.vstore.name_of(frame.var), frame.itv.lb, frame.itv.ub);)
          frame.vstore.update(frame.var, frame.itv);
          // Adjust the objective.
          frame.vstore.update(minimize_x, frame.vstore[minimize_x].join(best_bound));
          // Propagators that are now entailed or disentailed might not be anymore, therefore we reinitialize everybody to UNKNOWN.
          pstatus->backtrack();
          // Swap the current branch with the backtracked one.
          *current = frame.vstore;
        }
      }
      // At this stage, the current node is neither failed nor a solution yet.
      // We must branch.
      // The left branch is executed right away, and the right branch is pushed on the stack.
      else {
        LOG(printf("All IDLE, depth = %d\n", stack_size);)
        LOG(printf("res = %s\n", string_of_status(res));)
        LOG(current->print();)
        // Copy the current store.
        stack[stack_size].vstore = *current;
        // Select the variable and its value on which we want to branch.
        Var x = first_fail(*current, temporal_vars);
        LOG(printf("Selected %s.\n", current->name_of(x));)
        assign_lb(*current, x);
        LOG(printf("Correctly created left branch on %s.\n", current->name_of(x));)
        // Create the right branch.
        stack[stack_size].var = x;
        stack[stack_size].itv = not_assign_lb(stack[stack_size].vstore, x);
        LOG(printf("Left branching on %s: %d..%d \\/ %d..%d\n",
          current->name_of(x), (*current)[x].lb, (*current)[x].ub,
          stack[stack_size].itv.lb, stack[stack_size].itv.ub);)
        ++stack_size;
        assert(stack_size < MAX_DEPTH_TREE);
        // Change the IDLE status of propagators.
        pstatus->wake_up_all();
      }
    }
  }
  cudaFree(stack);
  printf("stop search\n");
}

template<typename T>
CUDA_GLOBAL void propagate_k(SharedData<T>* shared_data) {
  size_t id = threadIdx.x + blockIdx.x*blockDim.x;
  PropagatorsStatus& pstatus = *(shared_data->pstatus);
  T& p = shared_data->props[id];
  while (Exploring) {
    Status s = p.propagate(*(shared_data->vstore)) ? UNKNOWN : IDLE;
    if(p.is_entailed(*(shared_data->vstore))) {
      s = ENTAILED;
    }
    if(p.is_disentailed(*(shared_data->vstore))) {
      s = DISENTAILED;
    }
    pstatus.inplace_join(p.uid, s);
  }
}

// The variables pstatus and vstore are shared among all propagators of all types.
// The UID inside a propagator, e.g., `TemporalProp::uid`, refers to the index of the propagator in the status array of `pstatus`.
template<typename T>
SharedData<T>* launch(PropagatorsStatus* pstatus, VStore* vstore, std::vector<T> &c, cudaStream_t s)
{
  // printf("launching %lu threads on stream %p\n", c.size(), s[0]);

  T* props;
  CUDIE(cudaMallocManaged(&props, c.size() * sizeof(T)));
  for (int i=0; i < c.size(); ++i) {
    props[i] = c[i];
  }

  SharedData<T> *shared_data;
  CUDIE(cudaMallocManaged(&shared_data, sizeof(SharedData<T>)));
  *shared_data = SharedData<T>(pstatus, vstore, props);

  propagate_k<T><<<1, c.size(), 0, s>>>(shared_data);
  CUDIE0();
  return shared_data;
}

void solve(VStore* vstore, Constraints constraints, Var minimize_x)
{
  // std::cout << "Before propagation: " << std::endl;
  // vstore->print(var2name_raw);

  void* pstatus_raw;
  CUDIE(cudaMallocManaged(&pstatus_raw, sizeof(PropagatorsStatus)));
  PropagatorsStatus* pstatus = new(pstatus_raw) PropagatorsStatus(constraints.size());

  void* best_sol_raw;
  CUDIE(cudaMallocManaged(&best_sol_raw, sizeof(VStore)));
  VStore* best_sol = new(best_sol_raw) VStore(*vstore);

  constraints.print(*vstore);

  Var* temporal_vars = constraints.temporal_vars(vstore->size());

  cudaStream_t monitor;
  CUDIE(cudaStreamCreate(&monitor));
  cudaStream_t streams[PROPS_TYPE];
  for (int i=0; i < PROPS_TYPE; ++i) {
    CUDIE(cudaStreamCreate(&streams[i]));
  }

  search<<<1,1,0,monitor>>>(pstatus, vstore, best_sol, minimize_x, temporal_vars);
  CUDIE0();

  auto shared_data_0 = launch<TemporalProp>(pstatus, vstore, constraints.temporal, streams[0]);
  CUDIE0();
  auto shared_data_1 = launch<ReifiedLogicalAnd>(pstatus, vstore, constraints.reifiedLogicalAnd, streams[1]);
  CUDIE0();
  auto shared_data_2 = launch<LinearIneq>(pstatus, vstore, constraints.linearIneq, streams[2]);
  CUDIE0();

  CUDIE(cudaDeviceSynchronize());

  if(best_sol->size() == 0) {
    printf("Could not find a solution.\n");
  }
  else {
    printf("Best bound found is %d..%d.\n",
      (*best_sol)[minimize_x].lb, (*best_sol)[minimize_x].ub);
    best_sol->free();
  }
  CUDIE(cudaFree(best_sol_raw));
  CUDIE(cudaFree(temporal_vars));

  pstatus->free();
  CUDIE(cudaFree(pstatus_raw));

  CUDIE(cudaFree(shared_data_0->props));
  CUDIE(cudaFree(shared_data_0));
  CUDIE(cudaFree(shared_data_1->props));
  CUDIE(cudaFree(shared_data_1));
  CUDIE(cudaFree(shared_data_2->props));
  CUDIE(cudaFree(shared_data_2));

  CUDIE(cudaStreamDestroy(monitor));
  for (int i=0; i < PROPS_TYPE; ++i) {
    CUDIE(cudaStreamDestroy(streams[i]));
  }
}
