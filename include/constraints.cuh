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

CUDA_VAR uint Act_cnt = 0;

/// x + y <= c
struct XplusYleqC {
  Var x;
  Var y;
  int c;

  CUDA XplusYleqC(Var x, Var y, int c) : x(x), y(y), c(c) {}

  CUDA void propagate(VStore& vstore)
  {
    Act_cnt++;  // abstract?
    vstore.update(x,
        vstore[x].join({vstore[x].lb, c - vstore[y].lb}));
    vstore.update(y,
        vstore[y].join({vstore[y].lb, c - vstore[x].lb}));
  }

  CUDA bool is_entailed(VStore& vstore) {
    return vstore[x].ub + vstore[y].ub <= c;
  }

  CUDA bool is_disentailed(VStore& vstore) {
    return vstore[x].lb + vstore[y].lb > c;
  }

  CUDA XplusYleqC neg() {
    return XplusYleqC(-x, -y, -c - 1);
  }

  CUDA void print(Var2Name var2name) {
    VStore::print_var(x, var2name);
    printf(" + ");
    VStore::print_var(y, var2name);
    printf(" <= %d", c);
  }
};

// C1 \/ C2
struct LogicalOr {
  XplusYleqC left;
  XplusYleqC right;

  CUDA LogicalOr(XplusYleqC left, XplusYleqC right):
    left(left), right(right) {}

  CUDA void propagate(VStore& vstore) {
    if (left.is_disentailed(vstore)) {
      right.propagate(vstore);
    }
    else if (right.is_disentailed(vstore)) {
      left.propagate(vstore);
    }
  }

  CUDA bool is_entailed(VStore& vstore) {
    return left.is_entailed(vstore) || right.is_entailed(vstore);
  }

  CUDA bool is_disentailed(VStore& vstore) {
    return left.is_disentailed(vstore) && right.is_disentailed(vstore);
  }

  CUDA void print(Var2Name var2name) {
    left.print(var2name);
    printf(" \\/ ");
    right.print(var2name);
  }
};

/// b <=> left /\ right
struct ReifiedLogicalAnd {
  Var b;
  XplusYleqC left;
  XplusYleqC right;

  CUDA ReifiedLogicalAnd(Var b, XplusYleqC left, XplusYleqC right) :
    b(b), left(left), right(right) {}

  CUDA void propagate(VStore& vstore) {
    if (vstore[b] == 0) {
      LogicalOr(left.neg(), right.neg()).propagate(vstore);
    }
    else if (vstore[b] == 1) {
      left.propagate(vstore);
      right.propagate(vstore);
    }
    else if (left.is_entailed(vstore) && right.is_entailed(vstore)) {
      vstore.update(b, {1, 1});
    }
    else if (left.is_disentailed(vstore) || right.is_disentailed(vstore)) {
      vstore.update(b, {0, 0});
    }
  }

  CUDA void print(Var2Name var2name) {
    VStore::print_var(b, var2name);
    printf(" <=> (");
    left.print(var2name);
    printf(" /\\ ");
    right.print(var2name);
    printf(" )");
  }
};

template<typename Constraint>
CUDA_GLOBAL void propagate_k(Constraint c, VStore* vstore) {
  c.propagate(*vstore);
}

// x1c1 + ... + xNcN <= max
struct LinearIneq {
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
      printf("i=%d/%d\n", i,n);
      vars[i] = vvars[i];
      constants[i] = vconstants[i];
    }
    this->max = max;
  }

  LinearIneq(const LinearIneq& other) {
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

  CUDA void propagate(VStore& vstore) {
    int s = slack(vstore);
    for(int i=0; i < n; ++i) {
      Interval x = vstore[vars[i]];
      if (x.lb == 0 && x.ub == 1 && constants[i] > s) {
        vstore.update(vars[i], {0,0});
      }
    }
  }

  CUDA bool is_entailed(VStore& vstore) {
    return leftover(vstore) <= slack(vstore);
  }

  CUDA bool is_disentailed(VStore& vstore) {
    return slack(vstore) < 0;
  }

  CUDA void print(Var2Name var2name) {
    for(int i = 0; i < n; ++i) {
      printf("%d * ", constants[i]);
      VStore::print_var(vars[i], var2name);
      if (i != n-1) printf(" + ");
    }
    printf(" <= %d", max);
  }
};

struct Constraints {
  std::vector<XplusYleqC> xPlusYleqC;
  std::vector<ReifiedLogicalAnd> reifiedLogicalAnd;
  std::vector<LogicalOr> logicalOr;
  std::vector<LinearIneq> linearIneq;

  void print(Var2Name var2name)
  {
    for(auto c : xPlusYleqC) {
      c.print(var2name);
      printf("\n");
    }
    for(auto c : reifiedLogicalAnd) {
      c.print(var2name);
      printf("\n");
    }
    for(auto c : logicalOr) {
      c.print(var2name);
      printf("\n");
    }
    for(auto c : linearIneq) {
       c.print(var2name);
       printf("\n");
    }
  }
};

#endif
