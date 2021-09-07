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

#include "vstore.hpp"
#include "statistics.hpp"
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

class TreeAndPar
{
  VStore root;
  VStore current;
  const Array<Pointer<Propagator>>& props;
  Array<Delta> deltas;
  int deltas_size;
  int depth;
  Array<Var> branching_vars;
  Interval& best_bound;
  VStore best_sol;
  Var minimize_x;
  Statistics stats;
  int decomposition;
  int decomposition_size;

  // We maintain a set of unknown propagators.
  // The array `punknowns` contains the indices of all propagators.
  // For each index 0 <= i < unknown_props, props[i] is not yet entailed.
  // All the entailed propagators are stored in punknowns[unknown_props..props.size()].
  // When a propagator becomes entailed, we swap it with an unknown propagator at the position unknown_props, and decrease unknown_props (see `after_propagation` below).
  // NOTE1: We do not save `unknown_props` in `deltas` because, due to the recomputation-based scheme, all the propagators must be repropagated again when we replay.
  // NOTE2: We actually only use this at the root node because it does not seem more efficient to do it in every node.
  Array<int> punknowns;
  int unknown_props;

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
    props(props),
    deltas(MAX_DEPTH, allocator),
    deltas_size(0),
    depth(0),
    branching_vars(branching_vars, allocator),
    best_bound(best_bound),
    best_sol(root, global_allocator),
    minimize_x(min_x),
    punknowns(props.size(), allocator)
  {}

  __device__ int search(int tid, int stride, const VStore& root, int decomposition, int decomposition_size, bool& stop) {
    reset(tid, root, decomposition, decomposition_size);
    before_propagation(tid);
    __syncthreads();
    Interval b;
    while (deltas_size >= 0) {
      if(stop) {
        stats.exhaustive = false;
        break;
      }
      propagation(tid, stride);
      after_propagation(tid);
      before_propagation(tid);
      __syncthreads();
    }
    return this->decomposition_size;
  }

  __device__ void reset(int tid, const VStore& root, int decomposition, int decomposition_size) {
    if(tid == 0) {
      this->root.reset(root);
      this->current.reset(root);
      this->deltas_size = 0;
      this->decomposition = decomposition;
      this->decomposition_size = decomposition_size;
      this->stats = Statistics();
      assert(punknowns.size() == props.size());
      for(int i = 0; i < punknowns.size(); ++i) {
        punknowns[i] = i;
      }
      unknown_props = punknowns.size();
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
    }
  }

  __device__ void propagation(int tid, int stride) {
    __shared__ bool has_changed[3];
    has_changed[0] = true;
    has_changed[1] = true;
    has_changed[2] = true;
    for(int i = 1; !current.is_top() && has_changed[(i-1)%3]; ++i) {
      for (int t = tid; t < unknown_props; t += stride) {
        if(props[punknowns[t]]->propagate(current)) {
          has_changed[i%3] = true;
        }
      }
      has_changed[(i+1)%3] = false;
      __syncthreads();
    }
  }

  __device__ void after_propagation(int tid) {
    if (tid == 0) {
      if (current.is_top()) {
        on_failure();
      }
      else {
        bool all_entailed = true;
        for (int i = 0; all_entailed && i < unknown_props; ++i) {
          if(!props[punknowns[i]]->is_entailed(current)) {
            all_entailed = false;
          }
        }
        if(all_entailed) {
          on_solution();
        }
        else {
          INFO(assert(!current.all_assigned()));
          on_unknown();
        }
      }
    }
  }

  __device__ void eliminate_entailed_props() {
    for (int i = 0; i < unknown_props; ++i) {
      if(props[punknowns[i]]->is_entailed(root)) {
        --unknown_props;
        swap(&punknowns[i], &punknowns[unknown_props]);
        --i;
      }
    }
    INFO(printf("Propagators remaining at root: %d / %lu \n", unknown_props, props.size()));
  }

  __device__ void on_failure() {
    INFO(printf("backtracking on failed node %p...\n", this));
    stats.fails++;
    stats.depth_max = max(stats.depth_max, depth);
    backtrack();
    replay();
  }

  __device__ void on_solution() {
    INFO(printf("previous best...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
    stats.sols++;
    stats.depth_max = max(stats.depth_max, depth);
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
    end_bootstrap();
    branch();
    bootstrap_branch();
    commit_branch();
  }

  __device__ void end_bootstrap() {
    if(decomposition_size == 0) { // collapse the beginning of the tree to ignore the bootstrapped path of the decomposition, and avoid recomputing on root.
      root.reset(current);
      deltas_size = 0;
      eliminate_entailed_props();
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
    --depth;
    while (deltas_size >= 0 && deltas[deltas_size].next == deltas[deltas_size].right) {
      --deltas_size;
      --depth;
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
    depth++;
  }
};

#endif
