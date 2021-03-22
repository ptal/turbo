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

#ifndef TERMS_HPP
#define TERMS_HPP

#include "cuda_helper.hpp"

// NOTE: Bitwise OR and AND are necessary to avoid short-circuit of Boolean operators.

class Constant {
  int c;
public:
  typedef Constant neg_type;
  CUDA Constant(int c) : c(c) {}
  CUDA bool update_lb(VStore&, int) const { return false; }
  CUDA bool update_ub(VStore&, int) const { return false; }
  CUDA int lb(const VStore&) const { return c; }
  CUDA int ub(const VStore&) const { return c; }
  CUDA bool is_top(const VStore&) const { return false; }
  CUDA neg_type neg() const { return Constant(-c); }
  CUDA void print(const VStore&) const { printf("%d", c); }
};

template <typename Term>
class Negation {
  Term t;
public:
  typedef Term neg_type;
  CUDA Negation(Term t) : t(t) {}
  CUDA bool update_lb(VStore& vstore, int lb) const {
    return t.update_ub(vstore, -lb);
  }
  CUDA bool update_ub(VStore& vstore, int ub) const {
    return t.update_lb(vstore, -ub);
  }
  CUDA int lb(const VStore& vstore) const { return -t.ub(vstore); }
  CUDA int ub(const VStore& vstore) const { return -t.lb(vstore); }
  CUDA bool is_top(const VStore& vstore) const { return t.is_top(vstore); }
  CUDA neg_type neg() const { return t; }
  CUDA void print(const VStore& vstore) const { printf("-"); t.print(vstore); }
};

class Variable {
  int idx;
public:
  typedef Negation<Variable> neg_type;
  CUDA Variable(int idx) : idx(idx) {
    assert(idx >= 0);
  }
  CUDA bool update_lb(VStore& vstore, int lb) const {
    return vstore.update_lb(idx, lb);
  }
  CUDA bool update_ub(VStore& vstore, int ub) const {
    return vstore.update_ub(idx, ub);
  }
  CUDA int lb(const VStore& vstore) const { return vstore.lb(idx); }
  CUDA int ub(const VStore& vstore) const { return vstore.ub(idx); }
  CUDA bool is_top(const VStore& vstore) const { return vstore.is_top(idx); }
  CUDA neg_type neg() const { return neg_type(*this); }
  CUDA void print(const VStore& vstore) const { vstore.print_var(idx); }
};

#endif
