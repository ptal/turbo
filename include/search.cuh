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

const int MAX_STACK_SIZE = 1000;
const int MAX_NODE_ARRAY = 10;

struct NodeData {
  int n;
  PropagatorsStatus* pstatus;
  VStore* vstore;

  NodeData(VStore* vstore, int n): n(n), vstore(vstore) {
    PropagatorsStatus* pstatus_raw;
    malloc2_managed<PropagatorsStatus>(pstatus_raw, 1);
    pstatus = new(pstatus_raw) PropagatorsStatus(n);
  }

  ~NodeData() {
    if(pstatus != nullptr) {
      pstatus->~PropagatorsStatus();
      free2(pstatus);
      pstatus = nullptr;
    }
  }
};

class Stack {
  VStore* stack[MAX_STACK_SIZE];
  size_t stack_size;
public:
  Stack(const VStore& root) {
    for (int i=0; i < MAX_STACK_SIZE; ++i) {
      malloc2_managed(stack[i], 1);
      new(stack[i]) VStore(root);
    }
    stack_size = 1;
  }

  CUDA_DEVICE VStore*& pop() {
    LOG(printf("pop frame %lu\n", stack_size - 1));
    assert(stack_size > 0);
    --stack_size;
    return stack[stack_size];
  }

  CUDA_DEVICE VStore*& next_frame() {
    assert((stack_size + 1) < MAX_STACK_SIZE);
    ++stack_size;
    return stack[stack_size - 1];
  }

  CUDA_DEVICE bool is_empty() const {
    return stack_size == 0;
  }

  CUDA_DEVICE size_t size() const {
    return stack_size;
  }

  CUDA size_t capacity() {
    return MAX_STACK_SIZE;
  }
};

// Select the variable with the smallest domain in the store.
CUDA_DEVICE Var first_fail(const VStore& vstore, Var* vars) {
  Var x = -1;
  int lowest_lb = limit_max();
  for(int k = 0; vars[k] != -1; ++k) {
    int i = vars[k];
    if (vstore.lb(i) < lowest_lb && !vstore.view_of(i).is_assigned()) {
      x = i;
      lowest_lb = vstore.lb(i);
    }
  }
  if (x == -1) { vstore.print(); }
  assert(x != -1);
  return x;
}

CUDA_DEVICE void check_consistency(NodeData* node_data, Status res) {
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
    // assert(0);
  }
  if (res != DISENTAILED && node_data->vstore->is_top()) {
    INFO(printf("disentailment invariant inconsistent.\n"));
    INFO(printf("Status join again: %s\n", string_of_status(node_data->pstatus->join())));
    for(int i = 0; i < node_data->pstatus->size(); ++i) {
      INFO(printf("%d: %s\n", i, string_of_status(node_data->pstatus->of(i))));
    }
    node_data->vstore->print();
    // assert(0);
  }
}

struct TreeData {
  Statistics stats;
  Interval best_bound;
  VStore best_sol;
  Var minimize_x;
  Var* temporal_vars;
  Stack stack;
  NodeArray node_array;

  TreeData(Var* temporal_vars, Var minimize_x, const VStore& root):
    temporal_vars(temporal_vars), minimize_x(minimize_x),
    best_sol(VStore(root->size())), stack(root), node_array(root)
  {}

  CUDA_DEVICE void check_decreasing_bound(const VStore& current) {
    const Interval& new_bound = current.view_of(minimize_x);
    if (best_bound.ub <= new_bound.lb) {
      printf("Current bound: %d..%d.\n", best_bound.lb, best_bound.ub);
      printf("New bound: %d..%d.\n", new_bound.lb, new_bound.ub);
      printf("Found a new bound that is worst than the current one...\n");
      //assert(0);
    }
  }

  CUDA_DEVICE void on_solution(const VStore& leaf) {
    check_decreasing_bound(leaf);
    INFO(printf("previous best...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
    best_bound = leaf.view_of(minimize_x);
    best_bound.ub = best_bound.lb;
    INFO(printf("backtracking on solution...(bound %d..%d)\n", best_bound.lb, best_bound.ub));
    best_bound.lb = limit_min();
    best_sol.reset(leaf);
    stats.sols += 1;
    stats.best_bound = best_bound.ub;
  }

  CUDA_DEVICE void on_fail(const VStore& leaf) {
    stats.fails += 1;
    INFO(printf("backtracking on failed node %p...\n", leaf));
  }

  CUDA_DEVICE void on_unknown(Stack* stack, NodeData& parent) {
    VStore*& left = stack->next_frame();
    VStore*& right = stack->next_frame();
    swap(left, parent.vstore);
    right->reset(*left);
    Var x = first_fail(*left, temporal_vars);
    left->assign(x, left->lb(x));
    right->update(x, {left->lb(x) + 1, left->ub(x)});
    LOG(printf("Branching on %s: %d..%d \\/ %d..%d\n",
      left->name_of(x), left->lb(x), left->ub(x),
      right->lb(x), right->ub(x)));
  }

  CUDA_DEVICE void on_node(VStore& node) {
    Interval b = {best_bound.lb, best_bound.ub - 1};
    node.update(minimize_x, b);
  }
};

class NodeArray {
  NodeData data[MAX_NODE_ARRAY];
  size_t size;
public:
  NodeArray(const VStore& root) {
    for (int i=0; i < MAX_NODE_ARRAY; ++i) {
      data[i].init(root);
    }
    size = 0;
  }

  CUDA NodeData& operator[](size_t i) const {
    return data[i];
  }

  CUDA_DEVICE bool is_empty() const {
    return size == 0;
  }

  CUDA_DEVICE size_t size() const {
    return size;
  }

  CUDA void transferToSearch(Stack* searchStack, TreeData& td) {
    for(int i = 0; i < size; ++i) {
      Status res = data[i]->pstatus->join();
      res = (data[i].vstore->is_top() ? DISENTAILED : res);
      if (res == DISENTAILED) {
        td.on_fail(*(data[i].vstore));
      }
      else if (res == ENTAILED) {
        td.on_solution(*(data[i].vstore));
      }
      else {
        assert(res == IDLE);
        td.on_unknown(stack, data[i]);
      }
    }
  }

  CUDA size_t capacity() {
    return MAX_NODE_ARRAY;
  }

  CUDA void transferFromSearch(Stack* searchStack, TreeData& td) {
    for(size = 0; size < capacity() && !searchStack->is_empty(); ++size) {
      VStore*& current = searchStack->pop();
      swap(current, data[size].vstore);
      td.on_node(*(data[size].vstore));
    }
  }
};

#endif
