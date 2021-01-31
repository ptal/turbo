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
#include "cuda_helper.hpp"
#include "vstore.cuh"

/// x + y <= c
struct TemporalProp {
  int uid;
  Var x;
  Var y;
  int c;

  CUDA TemporalProp(Var x, Var y, int c) : uid(-1), x(x), y(y), c(c) {}

  CUDA bool propagate(VStore& vstore)
  {
    bool has_changed = vstore.update_ub(x, c - vstore.lb(y));
    has_changed |= vstore.update_ub(y, c - vstore.lb(x));
    return has_changed;
  }

  CUDA bool is_entailed(const VStore& vstore) const {
    return
      !vstore.is_top(x) &&
      !vstore.is_top(y) &&
      vstore.ub(x) + vstore.ub(y) <= c;
  }

  CUDA bool is_disentailed(const VStore& vstore) const {
    // if(vstore[x].lb + vstore[y].lb > c) {
    //   printf("Temporal constraint %d disentailed (x=%d, y=%d): %d + %d > %d\n",
    //     uid, x, y, vstore[x].lb, vstore[y].lb, c);
    // }
    return vstore.is_top(x) ||
           vstore.is_top(y) ||
           vstore.lb(x) + vstore.lb(y) > c;
  }

  CUDA TemporalProp neg() {
    return TemporalProp(-x, -y, -c - 1);
  }

  CUDA void print(const VStore& vstore) const {
    printf("%d: ", uid);
    vstore.print_var(x);
    printf(" + ");
    vstore.print_var(y);
    printf(" <= %d", c);
  }
};

// C1 \/ C2
struct LogicalOr {
  TemporalProp left;
  TemporalProp right;

  CUDA LogicalOr(TemporalProp left, TemporalProp right):
    left(left), right(right) {}

  CUDA bool propagate(VStore& vstore) {
    if (left.is_disentailed(vstore)) {
      return right.propagate(vstore);
    }
    else if (right.is_disentailed(vstore)) {
      return left.propagate(vstore);
    }
    return false;
  }

  CUDA bool is_entailed(const VStore& vstore) const {
    return left.is_entailed(vstore) || right.is_entailed(vstore);
  }

  CUDA bool is_disentailed(const VStore& vstore) const {
    return left.is_disentailed(vstore) && right.is_disentailed(vstore);
  }

  CUDA void print(const VStore& vstore) const {
    left.print(vstore);
    printf(" \\/ ");
    right.print(vstore);
  }
};

/// b <=> left /\ right
struct ReifiedLogicalAnd {
  int uid;
  Var b;
  TemporalProp left;
  TemporalProp right;

  CUDA ReifiedLogicalAnd(Var b, TemporalProp left, TemporalProp right) :
    b(b), left(left), right(right) {}

  CUDA bool propagate(VStore& vstore) {
    if (vstore.view_of(b) == 0) {
      return LogicalOr(left.neg(), right.neg()).propagate(vstore);
    }
    else if (vstore.view_of(b) == 1) {
      bool has_changed = left.propagate(vstore);
      has_changed |= right.propagate(vstore);
      return has_changed;
    }
    else if (left.is_entailed(vstore) && right.is_entailed(vstore)) {
      return vstore.assign(b, 1);
    }
    else if (left.is_disentailed(vstore) || right.is_disentailed(vstore)) {
      return vstore.assign(b, 0);
    }
    return false;
  }

  CUDA bool is_entailed(const VStore& vstore) const {
    return
         !vstore.is_top(b)
     && ((vstore.ub(b) == 0 && (left.is_disentailed(vstore) || right.is_disentailed(vstore)))
      || (vstore.lb(b) == 1 && left.is_entailed(vstore) && right.is_entailed(vstore)));
  }

  CUDA bool is_disentailed(const VStore& vstore) const {
    return
         vstore.is_top(b)
     || (vstore.ub(b) == 0 && (left.is_entailed(vstore) && right.is_entailed(vstore)))
     || (vstore.lb(b) == 1 && (left.is_disentailed(vstore) || right.is_disentailed(vstore)));
  }

  CUDA void print(const VStore& vstore) const {
    printf("%d: ", uid);
    vstore.print_var(b);
    printf(" <=> (");
    left.print(vstore);
    printf(" /\\ ");
    right.print(vstore);
    printf(" )");
  }
};

// x1c1 + ... + xNcN <= max
struct LinearIneq {
  int uid;
  int n;
  Var* vars;
  int* constants;
  int max;

  LinearIneq(std::vector<Var> vvars, std::vector<int> vconstants, int max) {
    assert(vvars.size() == vconstants.size());
    n = vvars.size();
    malloc2_managed(vars, n);
    malloc2_managed(constants, n);
    for(int i=0; i < n; ++i) {
      vars[i] = vvars[i];
      constants[i] = vconstants[i];
    }
    this->max = max;
  }

  LinearIneq(const LinearIneq& other) {
    uid = other.uid;
    n = other.n;
    max = other.max;
    malloc2_managed(vars, n);
    malloc2_managed(constants, n);
    for(int i=0; i < n; ++i) {
      vars[i] = other.vars[i];
      constants[i] = other.constants[i];
    }
  }

  ~LinearIneq() {
    free2(vars);
    free2(constants);
  }

  // Returns the maximum amount of additional resources this constraint can use if we fix all remaining boolean variables to 1.
  // LATTICE: monotone function w.r.t. `vstore`.
  // The potential can only decrease for any evolution of non-top `vstore`.
  CUDA int potential(const VStore& vstore) const {
    int potential = 0;
    for(int i=0; i < n; ++i) {
      if (vstore.lb(vars[i]) == 0 && vstore.ub(vars[i]) == 1) {
        potential += constants[i];
      }
    }
    return potential;
  }

  // Returns the amount of resources that this constraint can consume while still being satisfiable.
  // LATTICE: monotone function w.r.t. `vstore`.
  // The slack can only decrease for any evolution of non-top `vstore`.
  CUDA int slack(const VStore& vstore) const {
    int current = 0;
    for(int i=0; i < n; ++i) {
      current += vstore.lb(vars[i]) * constants[i];
    }
    return max - current;
  }

  // Consider this diagram of resources used.
  // The full window represent the number of resources used if we set all Boolean variables to 1.
  //
  //     |-------------------|_____|____|------|
  //               ^               ^        ^
  //            lb = 1            max    ub = 0
  //                         |_________________|
  //                              potential
  //                         |_____|
  //                          slack
  //
  //  Propagating assign upper bound to 0, when c_i > slack.
  CUDA bool propagate(VStore& vstore) {
    int s = slack(vstore);
    bool has_changed = false;
    // CORRECTNESS: Even if the slack changes after its computation (or even when we are computing it), it does not hinder the correctness of the propagation.
    // The reason is that whenever `constants[i] > s` it will stay true for any slack s' since s > s' by def. of the function slack.
    for(int i=0; i < n; ++i) {
      Interval x = vstore.view_of(vars[i]);
      if (x.lb == 0 && x.ub == 1 && constants[i] > s) {
        has_changed |= vstore.assign(vars[i], 0);
      }
    }
    return has_changed;
  }

  CUDA bool one_top(const VStore& vstore) const {
    for(int i = 0; i < n; ++i) {
      if(vstore.is_top(vars[i])) {
        return true;
      }
    }
    return false;
  }

  // From the diagram above, it is clear that once `potential <= slack` holds, it holds forever in a non-top `vstore`.
  // So even if `vstore` is modified during or between the computation of the potential or slack.
  CUDA bool is_entailed(const VStore& vstore) const {
    return
         !one_top(vstore)
      && potential(vstore) <= slack(vstore);
  }

  CUDA bool is_disentailed(const VStore& vstore) const {
    return
         one_top(vstore)
      || slack(vstore) < 0;
  }

  CUDA void print(const VStore& vstore) const {
    printf("%d: ", uid);
    for(int i = 0; i < n; ++i) {
      printf("%d * ", constants[i]);
      vstore.print_var(vars[i]);
      if (i != n-1) printf(" + ");
    }
    printf(" <= %d", max);
  }
};

struct Constraints {
  std::vector<TemporalProp> temporal;
  std::vector<ReifiedLogicalAnd> reifiedLogicalAnd;
  std::vector<LinearIneq> linearIneq;

  size_t size() {
    return temporal.size() + reifiedLogicalAnd.size() + linearIneq.size();
  }

  // Retrieve the temporal variables (those in temporal constraints).
  // It is useful for branching.
  // The array is terminated by -1.
  Var* temporal_vars(int max) {
    bool is_temporal[max] = {false};
    for(int i=0; i < temporal.size(); ++i) {
      is_temporal[abs(temporal[i].x)] = true;
      is_temporal[abs(temporal[i].y)] = true;
    }
    int n=0;
    for(int i=0; i < max; ++i) {
      n += is_temporal[i];
    }
    Var* vars;
    malloc2_managed(vars, (n+1));
    int j = 0;
    for(int i=0; i < max; ++i) {
      if (is_temporal[i]) {
        vars[j] = i;
        j++;
      }
    }
    vars[n] = -1;
    return vars;
  }

  void init_uids() {
    int i = 0;
    int limit = temporal.size();
    for(int j=0; j < limit; ++j, ++i) {
      temporal[j].uid = i;
    }
    limit = reifiedLogicalAnd.size();
    for(int j=0; j < limit; ++j, ++i) {
      reifiedLogicalAnd[j].uid = i;
    }
    limit = linearIneq.size();
    for(int j=0; j < limit; ++j, ++i) {
      linearIneq[j].uid = i;
    }
  }

  void print(const VStore& vstore)
  {
    for(auto c : temporal) {
      c.print(vstore);
      printf("\n");
    }
    for(auto c : reifiedLogicalAnd) {
      c.print(vstore);
      printf("\n");
    }
    for(auto c : linearIneq) {
       c.print(vstore);
       printf("\n");
    }
  }
};

#endif
