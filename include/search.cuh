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

const int INITIAL_STACK_SIZE = 20000; // 100000;
const int MAX_NODE_ARRAY =  8000;

class Stack;
struct TreeData;
class NodeArray;
class TreeData;

struct NodeData {
  int n;
  PropagatorsStatus* pstatus;
  VStore* vstore;

  CUDA NodeData() = default;

  CUDA NodeData(const VStore& root, int n): n(n) {
    malloc2_managed<PropagatorsStatus>(pstatus, 1);
    new(pstatus) PropagatorsStatus(n);
    malloc2_managed(vstore, 1);
    new(vstore) VStore(root, no_copy_tag());
    vstore->reset(root);
  }

  CUDA ~NodeData() {
    if(pstatus != nullptr) {
      pstatus->~PropagatorsStatus();
      free2(pstatus);
      pstatus = nullptr;
    }
  }
};


class Stack {
  VStore*** stacks;
  size_t stacks_capacity;
  size_t stack_no;
  size_t stack_size;
public:

  CUDA /* static */ VStore** init_stack(const VStore& root) {
    VStore** stack;
    malloc2_managed(stack, INITIAL_STACK_SIZE);
    for (int i=0; i < INITIAL_STACK_SIZE; ++i) {
      malloc2(stack[i], 1);
      new(stack[i]) VStore(root, no_copy_tag());
      stack[i]->reset(root);
    }
    return stack;
  }

  CUDA /* static */ VStore** init_stack2(const VStore& root) {
    VStore** stack;
    malloc2(stack, INITIAL_STACK_SIZE);
    for (int i=0; i < INITIAL_STACK_SIZE; ++i) {
      malloc2_managed(stack[i], 1);
      new(stack[i]) VStore(root, no_copy_tag());
    }
    return stack;
  }

  CUDA Stack(const VStore& root): stacks_capacity(1),
    stack_no(0), stack_size(1)
  {
    malloc2_managed(stacks, stacks_capacity);
    stacks[0] = init_stack(root);
    stacks[0][0]->reset(root);
  }

  CUDA void realloc_stacks() {
    VStore*** old_stacks = stacks;
    ++stacks_capacity;
    malloc2_managed(stacks, stacks_capacity);
    for(int i = 0; i < stacks_capacity - 1; ++i) {
      stacks[i] = old_stacks[i];
    }
    stacks[stacks_capacity - 1] = init_stack2(*(stacks[0][0]));
    free2(old_stacks);
  }

  CUDA VStore*& pop() {
    LOG(printf("pop frame %lu\n", size() - 1));
    assert(stack_no > 0 || stack_size > 0);
    if(stack_size == 0) {
      --stack_no;
      stack_size = INITIAL_STACK_SIZE - 1;
    }
    else {
      --stack_size;
    }
    return stacks[stack_no][stack_size];
  }

  CUDA VStore*& next_frame() {
    if((stack_size + 1) >= INITIAL_STACK_SIZE) {
      if (stack_no + 1 >= stacks_capacity) {
        realloc_stacks();
      }
      ++stack_no;
      stack_size = 1;
    }
    else {
      ++stack_size;
    }
    return stacks[stack_no][stack_size - 1];
  }

  CUDA bool is_empty() const {
    return stack_no == 0 && stack_size == 0;
  }

  CUDA size_t size() const {
    return stack_no * INITIAL_STACK_SIZE + stack_size;
  }

  CUDA size_t capacity() {
    return INITIAL_STACK_SIZE * stacks_capacity;
  }
};

// Select the variable with the smallest domain in the store.
CUDA Var first_fail(const VStore& vstore, Var* vars) {
  Var x = -1;
  int lowest_lb = limit_max();
  for(int k = 0; vars[k] != -1; ++k) {
    int i = vars[k];
    if (vstore.lb(i) < lowest_lb && !vstore.view_of(i).is_assigned()) {
      x = i;
      lowest_lb = vstore.lb(i);
    }
  }
  INFO(if (x == -1) { vstore.print(); })
  assert(x != -1);
  return x;
}

CUDA void check_consistency(NodeData* node_data, Status res) {
  INFO(printf("Node status: %s\n", string_of_status(res)));

  // Can be disentailed with all variable assigned...
  if (node_data->vstore->all_assigned() && res != ENTAILED) {
    INFO(printf("entailment invariant inconsistent (status = %s).\n",
      string_of_status(res)));
    INFO(printf("Status join again: %s\n", string_of_status(node_data->pstatus->join())));
    INFO(node_data->vstore->print());
    for(int i = 0; i < node_data->pstatus->size(); ++i) {
      if (node_data->pstatus->of(i) != ENTAILED) {
        INFO(printf("not entailed %d\n", i));
      }
    }
    assert(0);
  }
  if (res != DISENTAILED && node_data->vstore->is_top()) {
    INFO(printf("disentailment invariant inconsistent.\n"));
    INFO(printf("Status join again: %s\n", string_of_status(node_data->pstatus->join())));
    for(int i = 0; i < node_data->pstatus->size(); ++i) {
      INFO(printf("%d: %s\n", i, string_of_status(node_data->pstatus->of(i))));
    }
    node_data->vstore->print();
    assert(0);
  }
}

class NodeArray {
  NodeData data[MAX_NODE_ARRAY];
  size_t data_size;
public:
  CUDA NodeArray(const VStore& root, int np) {
    for (int i=0; i < MAX_NODE_ARRAY; ++i) {
      new(&data[i]) NodeData(root, np);
    }
    data_size = 0;
  }

  CUDA NodeData& operator[](size_t i) {
    return data[i];
  }

  CUDA void reset() {
    data_size = 0;
  }

  CUDA void push_swap(VStore*& vstore) {
    assert(data_size < capacity());
    data[data_size].pstatus->reset();
    swap(&vstore, &data[data_size].vstore);
    ++data_size;
  }

  CUDA bool is_empty() const {
    return data_size == 0;
  }

  CUDA size_t size() const {
    return data_size;
  }

  CUDA size_t capacity() {
    return MAX_NODE_ARRAY;
  }
};

struct TreeData {
  Statistics stats;
  Interval best_bound;
  VStore best_sol;
  Var minimize_x;
  Var* temporal_vars;
  Stack stack;
  NodeArray node_array;

  CUDA TreeData(Var* temporal_vars, Var minimize_x, const VStore& root, int np):
    temporal_vars(temporal_vars), minimize_x(minimize_x),
    best_sol(root.size(), no_copy_tag()), stack(root), node_array(root, np)
  {}

  CUDA void on_solution(const VStore& leaf) {
    INFO(printf("previous best...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
    Interval new_bound = leaf.view_of(minimize_x);
    // Due to parallelism, it is possible that several bounds are found in one iteration, thus we need to perform a (lattice) join on the best bound.
    best_bound.ub = min(best_bound.ub, new_bound.lb);
    INFO(printf("backtracking on solution...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
    best_sol.reset(leaf);
    stats.sols += 1;
    stats.best_bound = best_bound.ub;
  }

  CUDA void on_fail(const VStore& leaf) {
    stats.fails += 1;
    INFO(printf("backtracking on failed node %p...\n", &leaf));
  }

  CUDA void on_unknown(NodeData& parent) {
    VStore*& left = stack.next_frame();
    VStore*& right = stack.next_frame();
    swap(&left, &parent.vstore);
    right->reset(*left);
    Var x = first_fail(*left, temporal_vars);
    right->update(x, {left->lb(x) + 1, left->ub(x)});
    left->assign(x, left->lb(x));
    LOG(printf("Branching on %s: %d..%d \\/ %d..%d\n",
      left->name_of(x), left->lb(x), left->ub(x),
      right->lb(x), right->ub(x)));
  }

  CUDA void on_node(VStore& node) {
    stats.nodes = stats.nodes + 1;
    Interval b = {best_bound.lb, best_bound.ub - 1};
    node.update(minimize_x, b);
  }

  // NOTE: possible optimization:
  //   delete all nodes from the stack that cannot improve the best bound.
  CUDA void transferToSearch() {
    for(int i = 0; i < node_array.size(); ++i) {
      Status res = node_array[i].pstatus->join();
      res = (node_array[i].vstore->is_top() ? DISENTAILED : res);
      if (res == DISENTAILED) {
        on_fail(*(node_array[i].vstore));
      }
      else if (res == ENTAILED) {
        on_solution(*(node_array[i].vstore));
      }
      else {
        assert(res == IDLE);
        on_unknown(node_array[i]);
      }
    }
  }

  CUDA void transferToSearch_i(int i) {
    // does not work, because all i access the common stack
    Status res = node_array[i].pstatus->join();
    res = (node_array[i].vstore->is_top() ? DISENTAILED : res);
    if (res == DISENTAILED) {
      on_fail(*(node_array[i].vstore));
    }
    else if (res == ENTAILED) {
      on_solution(*(node_array[i].vstore));
    }
    else {
      assert(res == IDLE);
      on_unknown(node_array[i]);
    }
  }

  CUDA void transferFromSearch() {
    node_array.reset();
    while (node_array.size() < node_array.capacity() && !stack.is_empty()) {
      VStore*& current = stack.pop();
      node_array.push_swap(current);
      on_node(*(node_array[node_array.size() -1].vstore));
    }
  }
};

// -------------------------------------------------------------------------------------

#define DEPTH_MAX 200

struct Delta
{
  Var x;
  Interval next, right;
  CUDA Delta(Var x, Interval l, Interval r) : x(x), next(l), right(r) {}
};

__global__ void propagate_k(TreeAndPar *tree) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  Status s = UNKNOWN;
  while(s == UNKNOWN || s.has_changed()) {
    if (tid == 0) {
      tree->pstatus.reset_changed();
    }
    __threadfence_block();
    for (int t=tid; t<tree->props_sz; t+=stride) {
      tree->propagate(t);
    }
    __threadfence_block();
    s = tree->pstatus.join();
  }
}

struct TreeAndPar
{
  Var *branching_vars;
  Propagator *props;
  int props_sz;
  VStore root;
  VStore current;
  Delta deltas[DEPTH_MAX];
  int deltas_sz;
  PropagatorsStatus pstatus;
  Interval *best_bound;
  VStore best_sol;
  Var minimize_x;

  CUDA TreeAndPar(VStore root, Propagator *props, int props_sz, Var *vars, 
      Interval *best_bound, Var min_x) : 
    root(root), props(props), props_sz(props_sz), delta_sz(0), pstatus(props_sz), 
    branching_vars(vars), best_sol(), minimize_x(min_x), best_bound(best_bound) 
  {}

  CUDA void replay() {
   current = root;
   deltas[deltas_sz -1].next = delta[deltas_sz -1].right;
   for (int i=0; i < deltas_sz; ++i) {
    current.update(deltas[i].x, deltas[i].next);
   }
  }

  CUDA void propagate(int i) {
    Propagator& p = props[i];
    bool has_changed = p.propagate(current);
    Status s = has_changed ? UNKNOWN : IDLE;
    if(p.is_entailed(current)) {
      s = ENTAILED;
    }
    if(p.is_disentailed(current)) {
      s = DISENTAILED;
    }
    pstatus.inplace_join(p.uid, s);
  }

  CUDA void backtrack() {
    while (deltas_sz >= 0 && deltas[deltas_sz -1].next == deltas[deltas_sz -1].right) {
      --deltas_sz;
    }
  }

  CUDA void branch() {
    assert(deltas_sz < MAX_DEPTH);
    Var x = first_fail(current, branching_vars);
    deltas[deltas_sz].x = x;
    deltas[deltas_sz].next = {current->lb(x), current->lb(x)};
    deltas[deltas_sz].right = {current->lb(x) + 1, current->ub(x)};
    current.update(x, deltas[deltas_sz].next);
    deltas_sz++;
    LOG(printf("Branching on %s: %d..%d \\/ %d..%d\n",
      current->name_of(x), current->lb(x), current->lb(x),
      current->lb(x), current->ub(x)));
  }

  CUDA void search() {
    Interval b;
    while (deltas_sz >= 0) {
      b = {best_bound.lb, best_bound.ub - 1};
      current.update(minimize_x, b);
      propagate_k<<<1, min(props_sz, 256)>>>(this);
      CUDIE(cudaDeviceSynchronize());
      Status res = pstatus.join();
      res = (current.is_top() ? DISENTAILED : res);
      if (res == DISENTAILED) {
        INFO(printf("backtracking on failed node %p...\n", this));
        backtrack();
        replay();
      }
      else if (res == ENTAILED) {
        INFO(printf("previous best...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
        Interval new_bound = current.view_of(minimize_x);
        // Due to parallelism, it is possible that several bounds are found in one iteration, thus we need to perform a (lattice) join on the best bound.
        best_bound.ub = min(best_bound.ub, new_bound.lb);
        INFO(printf("backtracking on solution...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
        best_sol.reset(current);
        backtrack();
        replay();
      }
      else {
        assert(res == IDLE);
        INFO(printf("branching on unknown...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
        branch();
      }
    }
  }
};

#endif
