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

class Term {
public:
  CUDA virtual ~Term() {}
  CUDA virtual bool update_lb(VStore&, int) const = 0;
  CUDA virtual bool update_ub(VStore&, int) const = 0;
  CUDA virtual int lb(const VStore&) const = 0;
  CUDA virtual int ub(const VStore&) const = 0;
  CUDA virtual bool is_top(const VStore&) const = 0;
  CUDA virtual Term* negate() const = 0;
  CUDA virtual void print(const VStore&) const = 0;
};

class Constant : public Term
{
  int c;
public:
  typedef Constant neg_type;
  CUDA Constant(const Constant& other): c(other.c) {}
  CUDA Constant& operator=(const Constant& other) {
    c = other.c;
    return *this;
  }

  CUDA Constant(): c(limit_max()) {}
  CUDA Constant(int c) : c(c) {}
  CUDA bool update_lb(VStore&, int) const { return false; }
  CUDA bool update_ub(VStore&, int) const { return false; }
  CUDA int lb(const VStore&) const { return c; }
  CUDA int ub(const VStore&) const { return c; }
  CUDA bool is_top(const VStore&) const { return false; }
  CUDA neg_type neg() const { return Constant(-c); }
  CUDA void print(const VStore&) const { printf("%d", c); }
  CUDA virtual ~Constant() {}
  CUDA virtual Term* negate() const { return new neg_type(neg()); }
};

template <typename T>
class Negation : public Term {
  T t;
public:
  typedef T neg_type;

  CUDA Negation() {}
  CUDA Negation(const Negation& other): t(other.t) {}
  CUDA Negation& operator=(const Negation& other) {
    t = other.t;
    return *this;
  }

  CUDA Negation(T t) : t(t) {}
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

  CUDA virtual ~Negation() {}
  CUDA virtual Term* negate() const { return new neg_type(neg()); }
};

class Variable : public Term {
  int idx;
public:
  typedef Negation<Variable> neg_type;

  CUDA Variable(const Variable& other): idx(other.idx) {}

  CUDA Variable& operator=(const Variable& other) {
    idx = other.idx;
    return *this;
  }

  CUDA Variable(): idx(-1) {}
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

  CUDA virtual ~Variable() {}
  CUDA virtual Term* negate() const { return new neg_type(neg()); }
};

template <typename TermX, typename TermY>
class Add : public Term {
  TermX x;
  TermY y;
public:
  typedef Add<typename TermX::neg_type, typename TermY::neg_type> neg_type;

  CUDA Add() {}
  CUDA Add(const Add<TermX, TermY>& other): x(other.x), y(other.y) {}
  CUDA Add<TermX, TermY>& operator=(const Add<TermX, TermY>& other) {
    x = other.x;
    y = other.y;
    return *this;
  }

  CUDA Add(TermX x, TermY y) : x(x), y(y) {}

  // Enforce x + y >= k
  CUDA bool update_lb(VStore& vstore, int k) const {
    return x.update_lb(vstore, k - y.ub(vstore)) |
           y.update_lb(vstore, k - x.ub(vstore));
  }

  // Enforce x + y <= k
  CUDA bool update_ub(VStore& vstore, int k) const {
    return x.update_ub(vstore, k - y.lb(vstore)) |
           y.update_ub(vstore, k - x.lb(vstore));
  }

  CUDA int lb(const VStore& vstore) const { return x.lb(vstore) + y.lb(vstore); }
  CUDA int ub(const VStore& vstore) const { return x.ub(vstore) + y.ub(vstore); }

  CUDA bool is_top(const VStore& vstore) const { return x.is_top(vstore) || y.is_top(vstore); }

  CUDA neg_type neg() const { return neg_type(x.neg(), y.neg()); }

  CUDA void print(const VStore& vstore) const {
    x.print(vstore);
    printf(" + ");
    y.print(vstore);
  }

  CUDA virtual ~Add() {}
  CUDA virtual Term* negate() const { return new neg_type(neg()); }
};

template <typename TermX, typename TermY>
class Mul : public Term {
public:
  typedef Mul<typename TermX::neg_type, TermY> neg_type;
private:
  TermX x;
  TermY y;

  CUDA int div_up(int a, int b) const {
    assert(b != 0);
    int r = a / b;
    // division is rounded towards zero.
    // We add one only if `r` was truncated and `a, b` are of equal sign (so the division operated in the positive numbers).
    // Inspired by https://stackoverflow.com/questions/921180/how-can-i-ensure-that-a-division-of-integers-is-always-rounded-up/926806#926806
    return (a % b != 0 && a > 0 == b > 0) ? r + 1 : r;
  }

  CUDA int div_down(int a, int b) const {
    assert(b != 0);
    int r = a / b;
    return (a % b != 0 && a > 0 != b > 0) ? r - 1 : r;
  }
public:
  CUDA neg_type neg() const { return neg_type(x.neg(), y); }

  CUDA Mul(){}
  CUDA Mul(const Mul<TermX, TermY>& other): x(other.x), y(other.y) {}

  CUDA Mul<TermX, TermY>& operator=(const Mul<TermX, TermY>& other) {
    x = other.x;
    y = other.y;
    return *this;
  }

  CUDA Mul(TermX x, TermY y) : x(x), y(y) {}

  // Enforce x * y >= k
  CUDA bool update_lb(VStore& vstore, int k) const {
    // x * y >= k <=> -x * y <= -k - 1
    return neg().update_ub(vstore, -k - 1);
  }

  // Enforce x * y <= k
  CUDA bool update_ub(VStore& vstore, int k) const {
    int lx = x.lb(vstore);
    int ux = x.ub(vstore);
    int ly = y.lb(vstore);
    int uy = y.ub(vstore);
    // Check for `top` on x and y. This is necessary otherwise division by 0 might occur.
    if(lx > ux || ly > uy) return false;
    bool has_changed = false;
    if(k >= 0) {
      // Sign analysis: check if either x or y must be positive or negative according to the sign of k.
      // When the sign are reversed, e.g. -x and y, or x and -y, these rules will automatically fails one of the domain.
      if(ux < 0) { has_changed |= y.update_ub(vstore, -1); }
      if(uy < 0) { has_changed |= x.update_ub(vstore, -1); }
      if(lx >= 0) { has_changed |= y.update_lb(vstore, 0); }
      if(ly >= 0) { has_changed |= x.update_lb(vstore, 0); }
      // Both signs are positive.
      if(lx > 0 && ly >= 0) { has_changed |= y.update_ub(vstore, k / lx); }
      if(lx >= 0 && ly > 0) { has_changed |= x.update_ub(vstore, k / ly); }
      // Both signs are negative.
      if(ux < 0 && uy < 0) {
        has_changed |= y.update_ub(vstore, div_up(k, lx));
        has_changed |= x.update_ub(vstore, div_up(k, ly));
      }
    }
    else {
      // Sign analysis: check if either x or y must be positive or negative according to the sign of k.
      if(ux < 0) { has_changed |= y.update_lb(vstore, 0); }
      if(uy < 0) { has_changed |= x.update_lb(vstore, 0); }
      if(lx >= 0) { has_changed |= y.update_ub(vstore, -1); }
      if(ly >= 0) { has_changed |= x.update_ub(vstore, -1); }
      // When both variables have both signs.
      if(lx < 0 && ux > 0 && ly < 0 && uy > 0) {
        if(uy * lx > k) { has_changed |= x.update_ub(vstore, -1); }
        if(ux * ly > k) { has_changed |= y.update_ub(vstore, -1); }
      }
      // When the sign are reversed, e.g. -x and y, or x and -y.
      if(ux < 0 && uy > 0) { has_changed |= x.update_ub(vstore, div_up(k, uy)); }
      if(ux < 0 && uy >= 0) { has_changed |= y.update_lb(vstore, div_up(k, lx)); }
      if(ux >= 0 && uy < 0) { has_changed |= x.update_lb(vstore, div_up(k, ly)); }
      if(ux > 0 && uy < 0) { has_changed |= y.update_ub(vstore, div_up(k, ux)); }
    }
    return has_changed;
  }

  CUDA int lb(const VStore& vstore) const {
    int lx = x.lb(vstore);
    int ux = x.ub(vstore);
    int ly = y.lb(vstore);
    int uy = y.ub(vstore);
    return min(
      min(lx * ly, lx * uy),
      min(ux * ly, ux * uy));
  }

  CUDA int ub(const VStore& vstore) const {
    int lx = x.lb(vstore);
    int ux = x.ub(vstore);
    int ly = y.lb(vstore);
    int uy = y.ub(vstore);
    return max(
      max(lx * ly, lx * uy),
      max(ux * ly, ux * uy));
  }

  CUDA bool is_top(const VStore& vstore) const {
    return x.is_top(vstore) || y.is_top(vstore);
  }

  CUDA void print(const VStore& vstore) const {
    x.print(vstore);
    printf(" * ");
    y.print(vstore);
  }

  CUDA virtual ~Mul() {}
  CUDA virtual Term* negate() const { return new neg_type(neg()); }
};

template <
  template<typename X, typename Y> class Combinator,
  typename T, size_t n, size_t i>
struct BuildNaryTermHelper {
  typedef BuildNaryTermHelper<Combinator, T, n, i - 1> next_type;
  typedef Combinator<T, typename next_type::result_type> result_type;
  CUDA static result_type build(T terms[n]) {
    return result_type(terms[n-i], next_type::build(terms));
  }
};

template <
  template<typename X, typename Y> class Combinator,
  typename T, size_t n>
struct BuildNaryTermHelper<Combinator, T, n, 1> {
  typedef T result_type;
  CUDA static result_type build(T terms[n]) {
    return terms[n-1];
  }
};

template <
  template<typename X, typename Y> class Combinator,
  typename T, size_t n>
class NaryTerm : public Term {
  typedef BuildNaryTermHelper<Combinator, T, n, n> nary_constructor_type;
  typedef typename nary_constructor_type::result_type ast_type;
  // Flat representation of the terms
  T terms[n];
  // AST representation of the terms combined using Combinator.
  ast_type ast;
public:
  typedef NaryTerm<Combinator, T, n> this_type;
  typedef Negation<this_type> neg_type;

  CUDA NaryTerm(){}
  CUDA NaryTerm(const this_type& other): ast(other.ast) {
    for(int i = 0; i < n; ++i) {
      terms[i] = other.terms[i];
    }
  }
  CUDA NaryTerm& operator=(const this_type& other) {
    ast = other.ast;
    for(int i = 0; i < n; ++i) {
      terms[i] = other.terms[i];
    }
    return *this;
  }

  CUDA NaryTerm(T terms[n]) : ast(nary_constructor_type::build(terms)) {
    for(int i = 0; i < n; ++i) {
      this->terms[i] = terms[i];
    }
  }

  CUDA int lb(const VStore& vstore) const { return ast.lb(vstore); }
  CUDA int ub(const VStore& vstore) const { return ast.ub(vstore); }

  // Enforce t1 x t2 ... x tN >= k where `x` is the operation of Combinator.
  CUDA bool update_lb(VStore& vstore, int k) const {
    int ub = this->ub(vstore);
    bool has_changed = false;
    for(int i = 0; i < n; ++i) {
      has_changed |=
        terms[i].update_lb(vstore, k - (ub - terms[i].ub(vstore)));
    }
    return has_changed;
  }

  // Enforce x + y <= k
  CUDA bool update_ub(VStore& vstore, int k) const {
    int lb = this->lb(vstore);
    bool has_changed = false;
    for(int i = 0; i < n; ++i) {
      has_changed |=
        terms[i].update_ub(vstore, k - (lb - terms[i].lb(vstore)));
    }
    return has_changed;
  }

  CUDA bool is_top(const VStore& vstore) const {
    for(int i = 0; i < n; ++i) {
      if(terms[i].is_top(vstore)) {
        return true;
      }
    }
    return false;
  }

  CUDA neg_type neg() const {
    return neg_type(*this);
  }

  CUDA void print(const VStore& vstore) const {
    ast.print(vstore);
  }

  CUDA virtual ~NaryTerm() {}
  CUDA virtual Term* negate() const { return new neg_type(neg()); }
};

template<typename T, size_t n>
using NaryAdd = NaryTerm<Add, T, n>;


template<typename T, size_t n>
using NaryMul = NaryTerm<Mul, T, n>;

#endif
