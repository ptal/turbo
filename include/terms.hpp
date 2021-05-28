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
// NOTE: The pointers are shared among terms.
//       This is because terms have a constant state, that is, terms are read-only once constructed.
//       Therefore, the copy constructor does not perform a deep copy.

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
  virtual Term* to_device() const = 0;
};

// `DynTerm` wraps a term and inherits from Term.
// A vtable will be created.
template<typename BaseTerm>
class DynTerm: public Term {
  BaseTerm t;
public:
  typedef DynTerm<typename BaseTerm::neg_type> neg_type;

  CUDA DynTerm() {}
  CUDA DynTerm(const BaseTerm& t): t(t) {}
  CUDA DynTerm(const DynTerm<BaseTerm>& other): DynTerm(other.t) {}

  CUDA bool update_lb(VStore& vstore, int k) const override {
    return t.update_lb(vstore, k);
  }

  CUDA bool update_ub(VStore& vstore, int k) const override {
    return t.update_ub(vstore, k);
  }

  CUDA int lb(const VStore& vstore) const override {
    return t.lb(vstore);
  }

  CUDA int ub(const VStore& vstore) const override {
    return t.ub(vstore);
  }

  CUDA bool is_top(const VStore& vstore) const override {
    return t.is_top(vstore);
  }

  CUDA neg_type neg() const {
    return DynTerm(t.neg());
  }

  CUDA void print(const VStore& vstore) const override {
    t.print(vstore);
  }

  CUDA ~DynTerm() {}

  CUDA Term* negate() const override {
    return new neg_type(neg());
  }

  Term* to_device() const override {
    Term** dterm;
    malloc2_managed(dterm, 1);
    init_term<<<1, 1>>>(dterm, t);
    CUDIE(cudaDeviceSynchronize());
    return *dterm;
  }
};

template <typename BaseTerm>
CUDA_GLOBAL void init_term(Term** dterm, BaseTerm t) {
  *dterm = new DynTerm<BaseTerm>(t);
}


// This function is used to dereference the attribute if T is a pointer.
// The rational behind that, is to be able to manipulate a type T as a pointer or a reference.
// In the following code, our term AST is either static (only with template) or dynamic (with virtual function call).
// But we did not want to duplicate the code to handle both.
template <typename T>
CUDA const typename std::remove_pointer<T>::type* ptr(const T& x) {
  if constexpr(std::is_pointer<T>()) {
    return x;
  }
  else {
    return &x;
  }
  return 0; // unreachable (to avoid a compiler warning)
}

class Constant {
  int c;

public:

  typedef Constant neg_type;
  CUDA Constant(): c(limit_max()) {}
  CUDA Constant(int c) : c(c) {}
  CUDA Constant(const Constant& other): c(other.c) {}
  // CUDA Constant share() const { return Constant(c); }
  CUDA bool update_lb(VStore&, int) const { return false; }
  CUDA bool update_ub(VStore&, int) const { return false; }
  CUDA int lb(const VStore&) const { return c; }
  CUDA int ub(const VStore&) const { return c; }
  CUDA bool is_top(const VStore&) const { return false; }
  CUDA neg_type neg() const { return Constant(-c); }
  CUDA void print(const VStore&) const { printf("%d", c); }
};

template <typename T>
class Negation {
  T term;

  // See `ptr` function for explanations.
  CUDA __forceinline__ const T& t() const {
    return *(ptr(term));
  }
public:
  typedef T neg_type;

  CUDA Negation() {}
  CUDA Negation(const T& term): term(term) {}
  CUDA Negation(const Negation<T>& negation): Negation(negation.term) {}

  CUDA bool update_lb(VStore& vstore, int lb) const {
    return t().update_ub(vstore, -lb);
  }
  CUDA bool update_ub(VStore& vstore, int ub) const {
    return t().update_lb(vstore, -ub);
  }
  CUDA int lb(const VStore& vstore) const { return -t().ub(vstore); }
  CUDA int ub(const VStore& vstore) const { return -t().lb(vstore); }
  CUDA bool is_top(const VStore& vstore) const { return t().is_top(vstore); }
  CUDA neg_type neg() const { return term; }
  CUDA void print(const VStore& vstore) const {
    printf("-");
    t().print(vstore);
  }
};

class Variable {
  int idx;
public:
  typedef Negation<Variable> neg_type;

  CUDA Variable(): idx(-1) {}
  CUDA Variable(int idx) : idx(idx) {
    assert(idx >= 0);
  }
  CUDA Variable(const Variable& variable): Variable(variable.idx) {}

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

template <typename TermX, typename TermY>
class Add {
  TermX x_term;
  TermY y_term;

  CUDA __forceinline__ const TermX& x() const {
    return *(ptr(x_term));
  }

  CUDA __forceinline__ const TermY& y() const {
    return *(ptr(y_term));
  }
public:
  typedef Add<typename TermX::neg_type, typename TermY::neg_type> neg_type;

  CUDA Add() {}
  CUDA Add(const TermX& x_term, const TermY& y_term): x_term(x_term), y_term(y_term) {}
  CUDA Add(const Add<TermX, TermY>& other): Add(other.x_term, other.y_term) {}

  // Enforce x + y >= k
  CUDA bool update_lb(VStore& vstore, int k) const {
    return x().update_lb(vstore, k - y().ub(vstore)) |
           y().update_lb(vstore, k - x().ub(vstore));
  }

  // Enforce x + y <= k
  CUDA bool update_ub(VStore& vstore, int k) const {
    return x().update_ub(vstore, k - y().lb(vstore)) |
           y().update_ub(vstore, k - x().lb(vstore));
  }


  CUDA static int lb(int lx, int ux, int ly, int uy) {
    return lx + ly;
  }

  CUDA static int ub(int lx, int ux, int ly, int uy) {
    return ux + uy;
  }

  CUDA int lb(const VStore& vstore) const { return x().lb(vstore) + y().lb(vstore); }
  CUDA int ub(const VStore& vstore) const { return x().ub(vstore) + y().ub(vstore); }

  CUDA bool is_top(const VStore& vstore) const { return x().is_top(vstore) || y().is_top(vstore); }

  CUDA neg_type neg() const { return neg_type(x().neg(), y().neg()); }

  CUDA static char op() { return '+'; }

  CUDA void print(const VStore& vstore) const {
    x().print(vstore);
    printf(" + ");
    y().print(vstore);
  }
};

template <typename TermX, typename TermY>
class Mul {
public:
  typedef Mul<typename TermX::neg_type, TermY> neg_type;
private:
  TermX x_term;
  TermY y_term;

  CUDA __forceinline__ const TermX& x() const {
    return *(ptr(x_term));
  }

  CUDA __forceinline__ const TermY& y() const {
    return *(ptr(y_term));
  }

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

  CUDA Mul(){}
  CUDA Mul(const TermX& x_term, const TermY& y_term): x_term(x_term), y_term(y_term) {}
  CUDA Mul(const Mul<TermX, TermY>& other): Mul(other.x_term, other.y_term) {}

  CUDA neg_type neg() const { return neg_type(x().neg(), y_term); }
  // Enforce x * y >= k
  CUDA bool update_lb(VStore& vstore, int k) const {
    // x * y >= k <=> -x * y <= -k - 1
    return neg().update_ub(vstore, -k - 1);
  }

  // Enforce x * y <= k
  CUDA bool update_ub(VStore& vstore, int k) const {
    int ly = y().lb(vstore);
    int uy = y().ub(vstore);
    int lx = x().lb(vstore);

    // This small optimization is for the case where `c1 * x1` and `x1` is a Boolean variable.
    // This kind of expression occurs in pseudo-boolean expressions which are quite common.
    // The gain in efficiency is not a lot (~5%), so it would deserve to be benchmarked on more example.
    constexpr bool is_constant = std::is_same<TermX, Constant>();
    if(is_constant && ly >= 0 && uy <= 1 && lx > 0) {
      return y().update_ub(vstore, k/lx);
    }
    else {
      int ux = x().ub(vstore);
      // Check for `top` on x and y. This is necessary otherwise division by 0 might occur.
      if(lx > ux || ly > uy) return false;
      bool has_changed = false;
      if(k >= 0) {
        // Sign analysis: check if either x or y must be positive or negative according to the sign of k.
        // When the sign are reversed, e.g. -x and y, or x and -y, these rules will automatically fails one of the domain.
        if(ux < 0) { has_changed |= y().update_ub(vstore, -1); }
        if(uy < 0) { has_changed |= x().update_ub(vstore, -1); }
        if(lx >= 0) { has_changed |= y().update_lb(vstore, 0); }
        if(ly >= 0) { has_changed |= x().update_lb(vstore, 0); }
        // Both signs are positive.
        if(lx > 0 && ly >= 0) { has_changed |= y().update_ub(vstore, k / lx); }
        if(lx >= 0 && ly > 0) { has_changed |= x().update_ub(vstore, k / ly); }
        // Both signs are negative.
        if(ux < 0 && uy < 0) {
          has_changed |= y().update_ub(vstore, div_up(k, lx));
          has_changed |= x().update_ub(vstore, div_up(k, ly));
        }
      }
      else {
        // Sign analysis: check if either x or y must be positive or negative according to the sign of k.
        if(ux < 0) { has_changed |= y().update_lb(vstore, 0); }
        if(uy < 0) { has_changed |= x().update_lb(vstore, 0); }
        if(lx >= 0) { has_changed |= y().update_ub(vstore, -1); }
        if(ly >= 0) { has_changed |= x().update_ub(vstore, -1); }
        // When both variables have both signs.
        if(lx < 0 && ux > 0 && ly < 0 && uy > 0) {
          if(uy * lx > k) { has_changed |= x().update_ub(vstore, -1); }
          if(ux * ly > k) { has_changed |= y().update_ub(vstore, -1); }
        }
        // When the sign are reversed, e.g. -x and y, or x and -y.
        if(ux < 0 && uy > 0) { has_changed |= x().update_ub(vstore, div_up(k, uy)); }
        if(ux < 0 && uy >= 0) { has_changed |= y().update_lb(vstore, div_up(k, lx)); }
        if(ux >= 0 && uy < 0) { has_changed |= x().update_lb(vstore, div_up(k, ly)); }
        if(ux > 0 && uy < 0) { has_changed |= y().update_ub(vstore, div_up(k, ux)); }
      }
      return has_changed;
    }
  }

  CUDA static int lb(int lx, int ux, int ly, int uy) {
    return min(
      min(lx * ly, lx * uy),
      min(ux * ly, ux * uy));
  }

  CUDA static int ub(int lx, int ux, int ly, int uy) {
    return max(
      max(lx * ly, lx * uy),
      max(ux * ly, ux * uy));
  }

  CUDA int lb(const VStore& vstore) const {
    int lx = x().lb(vstore);
    int ux = x().ub(vstore);
    int ly = y().lb(vstore);
    int uy = y().ub(vstore);
    return lb(lx, ux, ly, uy);
  }

  CUDA int ub(const VStore& vstore) const {
    int lx = x().lb(vstore);
    int ux = x().ub(vstore);
    int ly = y().lb(vstore);
    int uy = y().ub(vstore);
    return ub(lx, ux, ly, uy);
  }

  CUDA bool is_top(const VStore& vstore) const {
    return x().is_top(vstore) || y().is_top(vstore);
  }

  CUDA static char op() { return '*'; }

  CUDA void print(const VStore& vstore) const {
    x().print(vstore);
    printf(" * ");
    y().print(vstore);
  }
};

template <
  template<typename X, typename Y> class Combinator,
  typename T>
class NaryTerm {
  T* terms;
  size_t n;

  CUDA __forceinline__ const T& t(size_t i) const {
    return *(ptr(terms[i]));
  }
public:
  typedef Combinator<T, T> fold_type;
  typedef NaryTerm<Combinator, T> this_type;
  typedef Negation<this_type> neg_type;

  CUDA NaryTerm(): terms(nullptr), n(0) {}
  CUDA NaryTerm(T* terms, size_t n): terms(terms), n(n) {}
  CUDA NaryTerm(const this_type& other): NaryTerm(other.terms, other.n) {}

  CUDA Interval fold(const VStore& vstore) const {
    assert(n > 0);
    int accu_lb = t(0).lb(vstore);
    int accu_ub = t(0).ub(vstore);
    for(int i = 1; i < n; ++i) {
      int lb = t(i).lb(vstore);
      int ub = t(i).ub(vstore);
      accu_lb = fold_type::lb(accu_lb, accu_ub, lb, ub);
      accu_ub = fold_type::ub(accu_lb, accu_ub, lb, ub);
    }
    return Interval(accu_lb, accu_ub);
  }

  CUDA int lb(const VStore& vstore) const { return fold(vstore).lb; }
  CUDA int ub(const VStore& vstore) const { return fold(vstore).ub; }

  // Enforce t1 x t2 ... x tN >= k where `x` is the operation of Combinator.
  CUDA bool update_lb(VStore& vstore, int k) const {
    int ub = this->ub(vstore);
    bool has_changed = false;
    for(int i = 0; i < n; ++i) {
      has_changed |=
        t(i).update_lb(vstore, k - (ub - t(i).ub(vstore)));
    }
    return has_changed;
  }

  // Enforce x + y <= k
  CUDA bool update_ub(VStore& vstore, int k) const {
    int lb = this->lb(vstore);
    bool has_changed = false;
    for(int i = 0; i < n; ++i) {
      has_changed |=
        t(i).update_ub(vstore, k - (lb - t(i).lb(vstore)));
    }
    return has_changed;
  }

  CUDA bool is_top(const VStore& vstore) const {
    for(int i = 0; i < n; ++i) {
      if(t(i).is_top(vstore)) {
        return true;
      }
    }
    return false;
  }

  CUDA neg_type neg() const {
    return neg_type(*this);
  }

  CUDA void print(const VStore& vstore) const {
    t(0).print(vstore);
    for(int i = 1; i < n; ++i) {
      printf(" %c ", fold_type::op());
      t(i).print(vstore);
    }
  }
};

template<typename T>
using NaryAdd = NaryTerm<Add, T>;

template<typename T>
using NaryMul = NaryTerm<Mul, T>;

#endif
