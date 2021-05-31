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

#ifndef CONSTRAINTS_HPP
#define CONSTRAINTS_HPP

#include <cassert>
#include <cmath>
#include <stdexcept>
#include "cuda_helper.hpp"
#include "vstore.hpp"
#include "terms.hpp"

class Propagator {
public:
  int uid;
public:
  Propagator() = default;
  CUDA Propagator(int uid): uid(uid) {}
  CUDA virtual ~Propagator() {}
  CUDA virtual bool propagate(VStore& vstore) const = 0;
  CUDA virtual bool is_entailed(const VStore& vstore) const = 0;
  CUDA virtual bool is_disentailed(const VStore& vstore) const = 0;
  CUDA virtual void print(const VStore& vstore) const = 0;
  virtual Propagator* neg() const = 0;
  // This function is called from host, and copy the object on the device memory.
  virtual Propagator* to_device() const = 0;
  __device__ virtual Propagator* clone_in(SharedAllocator& allocator) const = 0;
};

CUDA_GLOBAL void init_logical_or(Propagator** p, int uid, Propagator* left, Propagator* right);
CUDA_GLOBAL void init_logical_and(Propagator** p, int uid, Propagator* left, Propagator* right);
CUDA_GLOBAL void init_reified_prop(Propagator** p, int uid, Var b, Propagator* rhs, Propagator* not_rhs);

template <typename Term>
CUDA_GLOBAL void init_leq_propagator(Propagator** p, int uid, Term t, int c);

template<typename TermX, typename TermY>
CUDA_GLOBAL void init_temporal_prop(Propagator** p, int uid, TermX x, TermY y, int c);

// A propagator for the constraint `expr <= k` where expr is a term (cf. terms.hpp) and `k` an integer constant.
template <typename Term>
class LEQPropagator: public Propagator {
  const Term t;
  const int c;
public:
  CUDA LEQPropagator(const Term& t, int c):
    Propagator(-1), t(t), c(c) {}

  CUDA ~LEQPropagator() {}

  CUDA bool propagate(VStore& vstore) const override {
    return t.update_ub(vstore, c);
  }

  CUDA bool is_entailed(const VStore& vstore) const override {
    return !t.is_top(vstore) && t.ub(vstore) <= c;
  }

  CUDA bool is_disentailed(const VStore& vstore) const override {
    return t.is_top(vstore) || t.lb(vstore) > c;
  }

  Propagator* neg() const {
    return new LEQPropagator<typename Term::neg_type>
      (t.neg(), -c - 1);
  }

  CUDA void print(const VStore& vstore) const override {
    printf("%d: ", uid);
    t.print(vstore);
    printf(" <= %d", c);
  }

  Propagator* to_device() const override {
    Propagator** p;
    malloc2_managed(p, 1);
    init_leq_propagator<<<1, 1>>>(p, uid, t, c);
    CUDIE(cudaDeviceSynchronize());
    return *p;
  }

  __device__ Propagator* clone_in(SharedAllocator& allocator) const override {
    Propagator* p = new(allocator) LEQPropagator(t, c);
    p->uid = uid;
    return p;
  }
};

template <typename Term>
CUDA_GLOBAL void init_leq_propagator(Propagator** p, int uid, Term t, int c) {
  *p = new LEQPropagator<Term>(t, c);
  (*p)->uid = uid;
}

// C1 \/ C2
class LogicalOr: public Propagator {
public:
  const Propagator* left;
  const Propagator* right;

  CUDA LogicalOr(Propagator* left, Propagator* right):
    Propagator(-1), left(left), right(right) {}

  CUDA ~LogicalOr() {}

  CUDA bool propagate(VStore& vstore) const override {
    if (left->is_disentailed(vstore)) {
      return right->propagate(vstore);
    }
    else if (right->is_disentailed(vstore)) {
      return left->propagate(vstore);
    }
    return false;
  }

  CUDA bool is_entailed(const VStore& vstore) const override {
    return left->is_entailed(vstore) || right->is_entailed(vstore);
  }

  CUDA bool is_disentailed(const VStore& vstore) const override {
    return left->is_disentailed(vstore) && right->is_disentailed(vstore);
  }

  CUDA void print(const VStore& vstore) const override {
    left->print(vstore);
    printf(" \\/ ");
    right->print(vstore);
  }

  Propagator* neg() const {
    throw new std::runtime_error("Negation of logical or constraints unimplemented.");
  }

  Propagator* to_device() const override {
    Propagator* l = left->to_device();
    Propagator* r = right->to_device();
    Propagator** p;
    malloc2_managed(p, 1);
    init_logical_or<<<1, 1>>>(p, uid, l, r);
    CUDIE(cudaDeviceSynchronize());
    return *p;
  }

  __device__ Propagator* clone_in(SharedAllocator& allocator) const override {
    Propagator* p = new(allocator) LogicalOr(left->clone_in(allocator), right->clone_in(allocator));
    p->uid = uid;
    return p;
  }
};

// C1 /\ C2
class LogicalAnd: public Propagator {
public:
  const Propagator* left;
  const Propagator* right;

  CUDA LogicalAnd(Propagator* left, Propagator* right):
    Propagator(-1), left(left), right(right) {}

  CUDA ~LogicalAnd() {}

  CUDA bool propagate(VStore& vstore) const override {
    bool has_changed = right->propagate(vstore);
    has_changed |= left->propagate(vstore);
    return has_changed;
  }

  CUDA bool is_entailed(const VStore& vstore) const override {
    return left->is_entailed(vstore) && right->is_entailed(vstore);
  }

  CUDA bool is_disentailed(const VStore& vstore) const override {
    return left->is_disentailed(vstore) || right->is_disentailed(vstore);
  }

  CUDA void print(const VStore& vstore) const override {
    left->print(vstore);
    printf(" /\\ ");
    right->print(vstore);
  }

  Propagator* neg() const {
    return new LogicalOr(left->neg(), right->neg());
  }

  Propagator* to_device() const override {
    Propagator* l = left->to_device();
    Propagator* r = right->to_device();
    Propagator** p;
    malloc2_managed(p, 1);
    init_logical_and<<<1, 1>>>(p, uid, l, r);
    CUDIE(cudaDeviceSynchronize());
    return *p;
  }

  __device__ Propagator* clone_in(SharedAllocator& allocator) const {
    Propagator* p = new(allocator) LogicalAnd(left->clone_in(allocator), right->clone_in(allocator));
    p->uid = uid;
    return p;
  }
};

/// b <=> C
class ReifiedProp: public Propagator {
public:
  const Var b;
  const Propagator* rhs;
  const Propagator* not_rhs;

  ReifiedProp(Var b, Propagator* rhs) :
    Propagator(-1), b(b), rhs(rhs), not_rhs(rhs->neg()) {}

  CUDA ReifiedProp(Var b, Propagator* rhs, Propagator* not_rhs) :
    Propagator(-1), b(b), rhs(rhs), not_rhs(not_rhs) {}

  CUDA ~ReifiedProp() {}

  CUDA bool propagate(VStore& vstore) const override {
    if (vstore[b] == 0) {
      return not_rhs->propagate(vstore);
    }
    else if (vstore[b] == 1) {
      return rhs->propagate(vstore);
    }
    else if (rhs->is_entailed(vstore)) {
      return vstore.assign(b, 1);
    }
    else if (rhs->is_disentailed(vstore)) {
      return vstore.assign(b, 0);
    }
    return false;
  }

  CUDA bool is_entailed(const VStore& vstore) const override {
    return
         (vstore[b] == 0 && rhs->is_disentailed(vstore))
      || (vstore[b] == 1 && rhs->is_entailed(vstore));
  }

  CUDA bool is_disentailed(const VStore& vstore) const override {
    return
        (vstore[b] == 0 && rhs->is_entailed(vstore))
     || (vstore[b] == 1 && rhs->is_disentailed(vstore));
  }

  CUDA void print(const VStore& vstore) const override {
    printf("%d: ", uid);
    vstore.print_var(b);
    printf(" <=> (");
    rhs->print(vstore);
    printf(" )");
  }

  Propagator* neg() const {
    throw new std::runtime_error("Negation of reified constraints (b <=> C) unimplemented.");
  }

  Propagator* to_device() const override {
    Propagator* device_rhs = rhs->to_device();
    Propagator* device_not_rhs = not_rhs->to_device();
    Propagator** p;
    malloc2_managed(p, 1);
    init_reified_prop<<<1, 1>>>(p, uid, b, device_rhs, device_not_rhs);
    CUDIE(cudaDeviceSynchronize());
    return *p;
  }

  __device__ Propagator* clone_in(SharedAllocator& allocator) const {
    Propagator* p = new(allocator) ReifiedProp(b, rhs->clone_in(allocator), not_rhs->clone_in(allocator));
    p->uid = uid;
    return p;
  }
};

struct Constraints {
  std::vector<Propagator*> propagators;
  std::vector<Var> temporal_vars;

  size_t size() {
    return propagators.size();
  }

  // Retrieve the temporal variables (those in temporal constraints).
  // It is useful for branching.
  Array<Var>* branching_vars() {
    return new(managed_allocator) Array<Var>(temporal_vars);
  }

  void init_uids() {
    for(int i = 0; i < size(); ++i) {
      propagators[i]->uid = i;
    }
  }

  void print(const VStore& vstore) {
    for(auto p : propagators) {
      p->print(vstore);
      printf("\n");
    }
  }
};

#endif
