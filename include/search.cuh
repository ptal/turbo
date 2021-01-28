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

#ifndef SEARCH_HPP
#define SEARCH_HPP

#include "vstore.cuh"
#include "status.cuh"
#include "statistics.cuh"
#include "cuda_helper.hpp"

const int MAX_DEPTH_TREE = 100;

// Data shared among all threads (search and propagators).
struct SharedData {
  int n;
  PropagatorsStatus* pstatus;
  // Swap with pstatus when entering a new node.
  PropagatorsStatus* pstatus2;
  VStore* vstore;
  bool exploring;

  SharedData(VStore* vstore, int n): n(n), vstore(vstore), exploring(true) {
    void* pstatus_raw;
    CUDIE(cudaMallocManaged(&pstatus_raw, sizeof(PropagatorsStatus)));
    pstatus = new(pstatus_raw) PropagatorsStatus(n);
    void* pstatus2_raw;
    CUDIE(cudaMallocManaged(&pstatus2_raw, sizeof(PropagatorsStatus)));
    pstatus2 = new(pstatus2_raw) PropagatorsStatus(n);
  }

  // Allocate memory on the device and returned the store allocated from host.
  __device__ void into_device_mem() {
    void* vstore_raw;
    MALLOC_CHECK(cudaMalloc(&vstore_raw, sizeof(VStore)));
    vstore = new(vstore_raw) VStore(*vstore);
  }

  __host__ ~SharedData() {
    if(pstatus != nullptr) {
      pstatus->~PropagatorsStatus();
      cudaFree(pstatus);
      pstatus = nullptr;
    }
    if(pstatus2 != nullptr) {
      pstatus2->~PropagatorsStatus();
      cudaFree(pstatus2);
      pstatus2 = nullptr;
    }
  }
};

struct BacktrackingFrame {
  VStore* vstore;
  Var var;
  Interval itv;

  __device__ BacktrackingFrame() {
    MALLOC_CHECK(cudaMalloc(&vstore, sizeof(VStore)));
  }

  __device__ void init(const VStore& root) {
    *vstore = VStore(root);
  }

  __device__ ~BacktrackingFrame() {
    // vstore->~VStore();
    // cudaFree(vstore);
  }

  __device__ void commit() {
    LOG(printf("Taking right branch %s = %d..%d\n", vstore->name_of(var), itv.lb, itv.ub));
    vstore->update(var, itv);
  }

  __device__ void join_objective(Var minimize_x, const Interval& best_bound) {
    vstore->update(minimize_x, best_bound);
  }
};

class Stack {
  BacktrackingFrame stack[MAX_DEPTH_TREE];
  size_t stack_size;
public:
  __device__ Stack(const VStore& root) {
    for (int i=0; i<MAX_DEPTH_TREE; ++i) {
      stack[i].init(root);
    }
    stack_size = 0;
  }

  __device__ ~Stack() {
    for(int i = 0; i < MAX_DEPTH_TREE; ++i) {
      stack[i].~BacktrackingFrame();
    }
  }

  __device__ BacktrackingFrame& pop() {
    LOG(printf("pop frame %lu\n", stack_size - 1));
    --stack_size;
    return stack[stack_size];
  }

  __device__ void emplace_push(const VStore& vstore, Var var, Interval itv) {
    assert((stack_size + 1) < MAX_DEPTH_TREE);
    INFO(printf("Push %p\n", stack[stack_size].vstore));
    stack[stack_size].vstore->reset(vstore);
    stack[stack_size].var = var;
    stack[stack_size].itv = itv;
    ++stack_size;
  }

  __device__ BacktrackingFrame& top() {
    assert(stack_size > 0);
    return stack[stack_size - 1];
  }

  __device__ bool is_empty() const {
    return stack_size == 0;
  }

  __device__ size_t size() const {
    return stack_size;
  }
};


// Select the variable with the smallest domain in the store.
__device__ Var first_fail(const VStore& vstore, Var* vars) {
  Var x = -1;
  int lowest_lb = limit_max();
  for(int k = 0; vars[k] != -1; ++k) {
    int i = vars[k];
    if (vstore.lb(i) < lowest_lb && !vstore.view_of(i).is_assigned()) {
      x = i;
      lowest_lb = vstore.lb(i);
    }
  }
  assert(x != -1);
  return x;
}

__device__ void branch(Stack& stack, VStore& current, Var* temporal_vars) {
  Var x = first_fail(current, temporal_vars);
  stack.emplace_push(current, x, {current.lb(x) + 1, current.ub(x)});
  current.assign(x, current.lb(x));
  LOG(printf("Branching on %s: %d..%d \\/ %d..%d\n",
    current.name_of(x), current.lb(x), current.ub(x),
    stack.top().itv.lb, stack.top().itv.ub));
}

__device__ void check_consistency(SharedData* shared_data, Status res) {
  LOG(printf("Node status: %s\n", string_of_status(res)));
  if (shared_data->vstore->all_assigned() && res != ENTAILED) {
    printf("entailment invariant inconsistent (status = %s).\n",
      string_of_status(res));
    printf("Status join again: %s\n", string_of_status(shared_data->pstatus->join()));
    shared_data->vstore->print();
    for(int i = 0; i < shared_data->pstatus->size(); ++i) {
      if (shared_data->pstatus->of(i) != ENTAILED) {
        printf("not entailed %d\n", i);
      }
    }
    assert(0);
  }
  if (res != DISENTAILED && shared_data->vstore->is_top()) {
    printf("disentailment invariant inconsistent.\n");
    printf("Status join again: %s\n", string_of_status(shared_data->pstatus->join()));
    for(int i = 0; i < shared_data->pstatus->size(); ++i) {
      printf("%d: %s\n", i, string_of_status(shared_data->pstatus->of(i)));
    }
    shared_data->vstore->print();
    assert(0);
  }
}

__device__ void check_decreasing_bound(const Interval& current_bound, const Interval& new_bound) {
  if (current_bound.ub < new_bound.lb) {
    printf("Current bound: %d..%d.\n", current_bound.lb, current_bound.ub);
    printf("New bound: %d..%d.\n", new_bound.lb, new_bound.ub);
    printf("Found a new bound that is worst than the current one...\n");
    assert(0);
  }
}

__device__ void update_best_bound(const VStore& current, Var minimize_x, Interval& best_bound, VStore* best_sol) {
  check_decreasing_bound(best_bound, current.view_of(minimize_x));
  best_bound = current.view_of(minimize_x);
  best_bound.ub = best_bound.lb;
  INFO(printf("backtracking on solution...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
  best_bound.lb = limit_min();
  best_sol->reset(current);
}

CUDA_GLOBAL void search(SharedData* shared_data, Statistics* stats, VStore* best_sol, Var minimize_x, Var* temporal_vars) {
  shared_data->into_device_mem();
  Stack stack(*(shared_data->vstore));
  Interval best_bound = {limit_min(), limit_max()};
  INFO(printf("starting search with %p\n", shared_data->vstore));
  while (shared_data->exploring) {
    Status res = shared_data->pstatus->join();
    res = (shared_data->vstore->is_top() ? DISENTAILED : res);
    if (res != UNKNOWN) {
      stats->nodes += 1;
      stats->peak_depth = max<int>(stats->peak_depth, stack.size());
      check_consistency(shared_data, res);
      LOG(printf("Current bound: %d..%d, best bound: %d..%d\n", shared_data->vstore->lb(minimize_x), shared_data->vstore->ub(minimize_x), best_bound.lb, best_bound.ub));
      if(res != IDLE) {
        if(res == DISENTAILED) {
          stats->fails += 1;
          INFO(printf("backtracking on failed node %p...\n", shared_data->vstore));
        }
        else if(res == ENTAILED) {
          update_best_bound(*(shared_data->vstore), minimize_x, best_bound, best_sol);
          stats->sols += 1;
          stats->best_bound = best_bound.ub;
        }
        // If nothing is left in the stack, we stop the search, it means we explored the full search tree.
        if(stack.is_empty()) {
          shared_data->exploring = false;
        }
        else {
          BacktrackingFrame& frame = stack.pop();
          INFO(frame.vstore->print_view(temporal_vars));
          frame.commit();
          frame.join_objective(minimize_x, best_bound);
          // Swap the current branch with the backtracked one.
          INFO(printf("Backtrack from (%p, %p) to (%p, %p).\n", shared_data->vstore, shared_data->pstatus, frame.vstore, shared_data->pstatus2));
          INFO(frame.vstore->print_view(temporal_vars));
          swap(&shared_data->vstore, &frame.vstore);
          // Propagators that are now entailed or disentailed might not be anymore, therefore we reinitialize everybody to UNKNOWN.
          shared_data->pstatus2->reset();
          swap(&shared_data->pstatus, &shared_data->pstatus2);
        }
      }
      // At this stage, the current node is neither failed nor a solution yet.
      // We must branch.
      // The left branch is executed right away, and the right branch is pushed on the stack.
      else {
        LOG(printf("All IDLE, depth = %lu\n", stack.size()));
        LOG(printf("res = %s\n", string_of_status(res)));
        INFO(shared_data->vstore->print_view(temporal_vars));
        branch(stack, *(shared_data->vstore), temporal_vars);
        shared_data->pstatus->wake_up_all();
      }
    }
  }
  INFO(printf("stop search\n"));
}

#endif