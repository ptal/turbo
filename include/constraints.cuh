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
#include "cuda_helper.hpp"
#include "vstore.cuh"

/// x + y <= c
struct TemporalProp {
  int uid;
  Var x;
  Var y;
  int c;

  CUDA TemporalProp(Var x, Var y, int c) : x(x), y(y), c(c) {}

  CUDA bool propagate(VStore& vstore)
  {
    return
         vstore.update(x, {vstore[x].lb, c - vstore[y].lb})
      || vstore.update(y, {vstore[y].lb, c - vstore[x].lb});
  }

  CUDA bool is_entailed(VStore& vstore) {
    return
      !vstore[x].is_top() &&
      !vstore[y].is_top() &&
      vstore[x].ub + vstore[y].ub <= c;
  }

  CUDA bool is_disentailed(VStore& vstore) {
    // if(vstore[x].lb + vstore[y].lb > c) {
    //   printf("Temporal constraint %d disentailed (x=%d, y=%d): %d + %d > %d\n",
    //     uid, x, y, vstore[x].lb, vstore[y].lb, c);
    // }
    return vstore[x].is_top() ||
           vstore[y].is_top() ||
           vstore[x].lb + vstore[y].lb > c;
  }

  CUDA TemporalProp neg() {
    return TemporalProp(-x, -y, -c - 1);
  }

  CUDA void print(VStore& vstore) {
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

  CUDA bool is_entailed(VStore& vstore) {
    return left.is_entailed(vstore) || right.is_entailed(vstore);
  }

  CUDA bool is_disentailed(VStore& vstore) {
    return left.is_disentailed(vstore) && right.is_disentailed(vstore);
  }

  CUDA void print(VStore& vstore) {
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
    if (vstore[b] == 0) {
      // if (uid == 14) {printf("b=0\n");}
      return LogicalOr(left.neg(), right.neg()).propagate(vstore);
    }
    else if (vstore[b] == 1) {
      // if (uid == 14) {printf("b=1\n");}
      return
        left.propagate(vstore) ||
        right.propagate(vstore);
    }
    else if (left.is_entailed(vstore) && right.is_entailed(vstore)) {
      // if (uid == 14) {printf("l/r entailed\n");}
      return vstore.update(b, {1, 1});
    }
    else if (left.is_disentailed(vstore) || right.is_disentailed(vstore)) {
      // if (uid == 14) {printf("l || r disentailed\n");}
      return vstore.update(b, {0, 0});
    }
    // if (uid == 14) {
    //   vstore.print_var(b);
    //   printf("\n");
    //   vstore.print_var(left.x);
    //   printf("\n");
    //   vstore.print_var(left.y);
    //   printf("\n");
    //   vstore.print_var(right.x);
    //   printf("\n");
    //   vstore.print_var(right.y);
    //   printf("\n");
    // }
    return false;
  }

  CUDA bool is_entailed(VStore& vstore) {
    return
         !vstore[b].is_top()
     && ((vstore[b].ub == 0 && (left.is_disentailed(vstore) || right.is_disentailed(vstore)))
      || (vstore[b].lb == 1 && left.is_entailed(vstore) && right.is_entailed(vstore)));
  }

  CUDA bool is_disentailed(VStore& vstore) {
    return
         vstore[b].is_top()
     || (vstore[b].ub == 0 && (left.is_entailed(vstore) && right.is_entailed(vstore)))
     || (vstore[b].lb == 1 && (left.is_disentailed(vstore) || right.is_disentailed(vstore)));
  }

  CUDA void print(VStore& vstore) {
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
    CUDIE(cudaMallocManaged(&vars, sizeof(*vars) * n));
    CUDIE(cudaMallocManaged(&constants, sizeof(*constants) * n));
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
    CUDIE(cudaMallocManaged(&vars, sizeof(*vars) * n));
    CUDIE(cudaMallocManaged(&constants, sizeof(*constants) * n));
    for(int i=0; i < n; ++i) {
      vars[i] = other.vars[i];
      constants[i] = other.constants[i];
    }
  }

  ~LinearIneq() {
    CUDIE(cudaFree(vars));
    CUDIE(cudaFree(constants));
  }

  CUDA int leftover(VStore& vstore) {
    int leftover = 0;
    for(int i=0; i < n; ++i) {
      if (vstore[vars[i]].lb == 0 && vstore[vars[i]].ub == 1) {
        leftover += constants[i];
      }
    }
    return leftover;
  }

  CUDA int slack(VStore& vstore) {
    int current = 0;
    for(int i=0; i < n; ++i) {
      current += vstore[vars[i]].lb * constants[i];
    }
    return max - current;
  }

  CUDA bool propagate(VStore& vstore) {
    int s = slack(vstore);
    bool has_changed = false;
    for(int i=0; i < n; ++i) {
      Interval x = vstore[vars[i]];
      if (x.lb == 0 && x.ub == 1 && constants[i] > s) {
        has_changed |= vstore.update(vars[i], {0,0});
      }
    }
    return has_changed;
  }

  CUDA bool one_top(VStore& vstore) {
    for(int i = 0; i < n; ++i) {
      if(vstore[vars[i]].is_top()) {
        return true;
      }
    }
    return false;
  }

  CUDA bool is_entailed(VStore& vstore) {
    return
         !one_top(vstore)
      && leftover(vstore) <= slack(vstore);
  }

  CUDA bool is_disentailed(VStore& vstore) {
    return
         one_top(vstore)
      || slack(vstore) < 0;
  }

  CUDA void print(VStore& vstore) {
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
    CUDIE(cudaMallocManaged(&vars, (n+1)*sizeof(Var)));
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

  void print(VStore& vstore)
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
