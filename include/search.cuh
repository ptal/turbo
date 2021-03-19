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

#define MAX_DEPTH 200

// Select the variable with the smallest domain in the store.
CUDA Var first_fail(const VStore& vstore, Array<Var>& vars) {
  Var x = -1;
  int lowest_lb = limit_max();
  for(int k = 0; k < vars.size(); ++k) {
    int i = vars[k];
    if (vstore.lb(i) < lowest_lb && !vstore[i].is_assigned()) {
      x = i;
      lowest_lb = vstore.lb(i);
    }
  }
  INFO(if (x == -1) { vstore.print(); })
  assert(x != -1);
  return x;
}

struct Delta
{
  Var x;
  Interval next;
  Interval right;
  CUDA Delta(): x(0), next(), right() {}
  CUDA Delta(Var x, Interval l, Interval r):
    x(x), next(l), right(r) {}
};

#define NODES_LIMIT 100000

class TreeAndPar
{
  VStore root;
  VStore current;
  PropagatorsStatus pstatus;
  const Array<Pointer<Propagator>>& props;
  Array<Delta> deltas;
  int deltas_size;
  Array<Var> branching_vars;
  Interval& best_bound;
  VStore best_sol;
  Var minimize_x;
  Statistics stats;
  int decomposition;
  int decomposition_size;

public:
  template<typename Allocator>
  __device__ TreeAndPar(
   const VStore &root,
   const Array<Pointer<Propagator>> &props,
   const Array<Var> &branching_vars,
   Interval &best_bound,
   Var min_x,
   Allocator &allocator)
  : root(root, allocator),
    current(root, allocator),
    pstatus(props.size(), allocator),
    props(props),
    deltas(MAX_DEPTH, allocator),
    deltas_size(0),
    branching_vars(branching_vars, allocator),
    best_bound(best_bound),
    best_sol(root, global_allocator),
    minimize_x(min_x)
  {}

  __device__ void search(int tid, int stride, const VStore& root, int decomposition, int decomposition_size) {
    reset(tid, root, decomposition, decomposition_size);
    before_propagation(tid);
    __syncthreads();
    Interval b;
    while (deltas_size >= 0) {
      if(stats.nodes >= NODES_LIMIT) break;
      propagation(tid, stride);
      __syncthreads();
      after_propagation(tid);
      before_propagation(tid);
      __syncthreads();
    }
  }

  __device__ void reset(int tid, const VStore& root, int decomposition, int decomposition_size) {
    if(tid == 0) {
      this->root.reset(root);
      this->current.reset(root);
      this->deltas_size = 0;
      this->pstatus.reset();
      this->decomposition = decomposition;
      this->decomposition_size = decomposition_size;
      this->stats = Statistics();
    }
  }

  __device__ const VStore& best() const {
    return best_sol;
  }

  __device__ const Statistics& statistics() const {
    return stats;
  }

private:
  __device__ void before_propagation(int tid) {
    if (tid == 0) {
      stats.nodes++;
      current.update(minimize_x, {best_bound.lb, best_bound.ub - 1});
      pstatus.reset();
    }
  }

  __device__ void propagation(int tid, int stride) {
    Status s = UNKNOWN;
    while(s == UNKNOWN || pstatus.has_changed()) {
      __syncthreads();
      if (tid == 0) {
        pstatus.reset_changed();
      }
      __syncthreads();
      for (int t = tid; t < props.size(); t += stride) {
        propagate_one(t);
      }
      __syncthreads();
      s = pstatus.join();
    }
  }

  __device__ void after_propagation(int tid) {
    if (tid == 0) {
      Status res = (current.is_top() ? DISENTAILED : pstatus.join());
      switch(res) {
        case DISENTAILED: on_failure(); break;
        case ENTAILED: on_solution(); break;
        case IDLE: on_unknown(); break;
        default: assert(false);
      }
    }
  }

  __device__ void on_failure() {
    INFO(printf("backtracking on failed node %p...\n", this));
    stats.fails++;
    stats.peak_depth = max(stats.peak_depth, deltas_size);
    backtrack();
    replay();
  }

  __device__ void on_solution() {
    INFO(printf("previous best...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
    stats.sols++;
    stats.peak_depth = max(stats.peak_depth, deltas_size);
    const Interval& new_bound = current[minimize_x];
    // Due to parallelism, it is possible that several bounds are found in one iteration, thus we need to perform a (lattice) join on the best bound.
    atomicMin(&best_bound.ub, new_bound.lb);
    stats.best_bound = best_bound.ub;
    INFO(printf("backtracking on solution...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
    best_sol.reset(current);
    backtrack();
    replay();
  }

  __device__ void on_unknown() {
    INFO(printf("branching on unknown node... (bound %d..%d)\n", best_bound.lb, best_bound.ub));
    // end_bootstrap();
    branch();
    bootstrap_branch();
    commit_branch();
  }

  __device__ void end_bootstrap() {
    if(decomposition_size == 0) { // collapse the beginning of the tree to ignore the bootstrapped path of the decomposition, and avoid recomputing on root.
      root.reset(current);
      deltas_size = 0;
    }
  }

  __device__ void bootstrap_branch() {
    decomposition_size -= 1;
    if(decomposition_size >= 0) {
      if (!(decomposition & 1)) { // left branch
        LOG(printf("decomposition %d: %d, %d, left\n", blockIdx.x, decomposition, decomposition_size));
        deltas[deltas_size - 1].right = deltas[deltas_size - 1].next;
      }
      else {
        LOG(printf("decomposition %d: %d, %d, right\n", blockIdx.x, decomposition, decomposition_size));
      }
      deltas[deltas_size - 1].next = deltas[deltas_size - 1].right;
      decomposition >>= 1;
    }
  }

  __device__ void replay() {
    if(deltas_size >= 0) {
      current.reset(root);
      deltas[deltas_size].next = deltas[deltas_size].right;
      for (int i = 0; i <= deltas_size; ++i) {
        current.update(deltas[i].x, deltas[i].next);
      }
      deltas_size++;
    }
  }

  __device__ void backtrack() {
    --deltas_size;
    while (deltas_size >= 0 && deltas[deltas_size].next == deltas[deltas_size].right) {
      --deltas_size;
    }
  }

  __device__ void commit_branch() {
    current.update(deltas[deltas_size - 1].x, deltas[deltas_size - 1].next);
  }

  __device__ void branch() {
    assert(deltas_size < MAX_DEPTH);
    Var x = first_fail(current, branching_vars);
    deltas[deltas_size].x = x;
    deltas[deltas_size].next = {current.lb(x), current.lb(x)};
    deltas[deltas_size].right = {current.lb(x) + 1, current.ub(x)};
    LOG(printf("Branching on %s: %d..%d \\/ %d..%d\n",
      current.name_of(x), deltas[deltas_size].next.lb, deltas[deltas_size].next.ub,
      deltas[deltas_size].right.lb, deltas[deltas_size].right.ub));
    deltas_size++;
  }

  __device__ void propagate_one(int i) {
    const Pointer<Propagator>& p = props[i];
    bool has_changed = p->propagate(current);
    Status s = has_changed ? UNKNOWN : IDLE;
    if(p->is_entailed(current)) {
      s = ENTAILED;
    }
    if(p->is_disentailed(current)) {
      s = DISENTAILED;
    }
    pstatus.inplace_join(p->uid, s);
  }
};

#endif
