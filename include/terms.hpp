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
  const int c;
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
  CUDA virtual ~Constant() {}
  CUDA virtual Term* negate() const { return new neg_type(neg()); }
};

template <typename T>
class Negation : public Term {
  const T t;
public:
  typedef T neg_type;
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
  const int idx;
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

  CUDA virtual ~Variable() {}
  CUDA virtual Term* negate() const { return new neg_type(neg()); }
};

template <typename TermX, typename TermY>
class Add : public Term {
  const TermX x;
  const TermY y;
public:
  typedef Add<typename TermX::neg_type, typename TermY::neg_type> neg_type;
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

template <
  template<typename X, typename Y> class Combinator,
  typename T, size_t n, size_t i>
struct BuildNaryTermHelper {
  typedef BuildNaryTermHelper<Combinator, T, n, i - 1> next_type;
  typedef Combinator<T, next_type> result_type;
  static result_type build(T terms[n]) {
    return result_type(terms[n-i], next_type::build(terms));
  }
};

template <
  template<typename X, typename Y> class Combinator,
  typename T, size_t n>
struct BuildNaryTermHelper<Combinator, T, n, 1> {
  typedef T result_type;
  static result_type build(T terms[n]) {
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
  const T terms[n];
  // AST representation of the terms combined using Combinator.
  ast_type ast;
public:
  typedef NaryTerm<Combinator, T, n> this_type;
  typedef Negation<this_type> neg_type;

  CUDA NaryTerm(T terms[n]): terms(terms), ast(ast_type::build(terms)) {}

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

#endif
