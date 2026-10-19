// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_AFFINE_V2_HPP
#define TURBO_JET_AFFINE_V2_HPP

#include "battery/utility.hpp"
#include "lala/finterval.hpp"

/** Second version of the deduction of an affine neuron of JET, selected with `-jet_affine v2`.
 *
 * It computes the same projection as `FastNNRelu::deduce_neuron` (version 1): the forward sum
 * \f$ s = \sum_c w_c x_c + b \f$, the ReLU (or the identity), then for every input the projection
 * \f$ x_c \in (N - \sum_{c' \neq c} w_{c'} x_{c'}) / w_c \f$ where `N` is the narrowed sum. The
 * differences are only in how much work is done:
 *
 *  1. A constant weight times an interval costs 2 directed multiplications instead of the generic
 *     interval product (8 multiplications plus min/max), and the weight is never "narrowed".
 *  2. A term \f$ t_c = w_c x_c \f$ with width \f$ d_c \f$ can only be tightened from above if
 *     \f$ d_c > \overline{N} - L \f$ and from below if \f$ d_c > U - \underline{N} \f$, where
 *     `[L, U]` is the forward sum. If no term passes this test, the whole backward pass is skipped;
 *     otherwise only the terms passing it are divided by their weight.
 *
 * Skipping a term or a row only ever loses precision, never soundness. The tests are nevertheless
 * rounded conservatively (widths up, gaps down), so that nothing the projection could compute is
 * skipped. Every bound that is written to the store is rounded outward.
 *
 * Invariant (concurrency): in the backward pass, the term \f$ t_c \f$ subtracted from the forward
 * sum must be computed from bounds of \f$ x_c \f$ read *after* the forward sum was formed. Domains
 * only shrink, so the subtracted lower bound is then at least the one that was added, and
 * `L - tl_c` remains a valid lower bound of the other terms. Reading `x_c` once and reusing the value
 * of the forward pass is also sound; reading an *older* value is not.
 */
namespace jet {
namespace affine_v2 {

/** `[tl, tu] = w * [l, u]` for a constant weight `w`, rounded outward. */
template <class T>
CUDA INLINE void scalar_term(T w, T l, T u, T& tl, T& tu) {
  if(w > T{0}) {
    tl = battery::mul_down<T>(w, l);
    tu = battery::mul_up<T>(w, u);
  }
  else if(w < T{0}) {
    tl = battery::mul_down<T>(w, u);
    tu = battery::mul_up<T>(w, l);
  }
  else {
    tl = T{0};
    tu = T{0};
  }
}

/** Deduction of the affine neuron `target` = (ReLU of) \f$ \sum_{c < fan\_in} W(c) \cdot x_{prev\_base + c} + bias \f$.
 *
 * `Neurons` gives access to the store of neurons:
 *   * `T lb(int i) const`, `T ub(int i) const`: the current bounds of the neuron `i`;
 *   * `bool meet(int i, T l, T u)`: intersect the neuron `i` with `[l, u]`, return `true` if it shrank
 *     (an infinite bound means "no information"; a NaN bound must be ignored, as `LB::meet` does);
 *   * `bool fail(int i)`: make the neuron `i` empty, return `true`.
 * `WeightFn` is a callable `T W(int c)` returning the weight of the connection `c -> target`.
 * With `backward == false` (`-forward_only`), only the forward projection is done (STEP 1 and 2): the
 * neurons of the previous layer are never tightened.
 *
 * \return `true` if the store changed. */
template <class T, class Neurons, class WeightFn>
CUDA bool deduce(Neurons& neurons, const WeightFn& W, int prev_base, int fan_in, int target, T bias, bool has_relu,
  bool backward = true)
{
  using RItv = lala::FInterval<T>;
  constexpr T inf = battery::limits<T>::inf();
  constexpr T neg_inf = battery::limits<T>::neg_inf();

  // STEP 1: forward sum [sl, su] of the terms, and the largest term width (for the skip test).
  T sl = T{0};
  T su = T{0};
  T dmax = T{0};
  for(int c = 0; c < fan_in; ++c) {
    const RItv x(neurons.lb(prev_base + c), neurons.ub(prev_base + c));
    if(x.is_bot()) {
      return neurons.fail(target);
    }
    T tl, tu;
    scalar_term(W(c), x.lb().load(), x.ub().load(), tl, tu);
    sl = battery::add_down<T>(sl, tl);
    su = battery::add_up<T>(su, tu);
    dmax = battery::max(dmax, battery::sub_up<T>(tu, tl));
  }

  // STEP 2: bias and ReLU, exactly as in version 1.
  RItv pre(battery::add_down<T>(sl, bias), battery::add_up<T>(su, bias));
  RItv y(neurons.lb(target), neurons.ub(target));
  if(has_relu) {
    RItv zero(T{0});
    lala::tell::fmax(y, pre, zero);  // forward on `y`, backward on `pre`.
  }
  else {
    y.meet(pre);
    pre = y;
  }
  bool has_changed = neurons.meet(target, y.lb().load(), y.ub().load());
  if(!backward || pre.is_bot()) {
    return has_changed;
  }

  // STEP 3: backward. `[nl, nu]` is the narrowed sum without the bias.
  const T nl = battery::max(sl, battery::sub_down<T>(pre.lb().load(), bias));
  const T nu = battery::min(su, battery::sub_up<T>(pre.ub().load(), bias));
  if(nl > nu) {
    return neurons.fail(target);
  }
  const T gap_u = battery::sub_down<T>(nu, sl);  // a term can only be tightened from above if its width exceeds gap_u,
  const T gap_l = battery::sub_down<T>(su, nl);  // and from below if its width exceeds gap_l.
  if(dmax <= gap_u && dmax <= gap_l) {  // false on NaN: we then do not skip.
    return has_changed;
  }
  for(int c = 0; c < fan_in; ++c) {
    const T w = W(c);
    if(w == T{0}) { continue; }
    const int src = prev_base + c;
    /** Re-read after the forward sum was formed (see the invariant above). */
    T tl, tu;
    scalar_term(w, neurons.lb(src), neurons.ub(src), tl, tu);
    const T d = battery::sub_up<T>(tu, tl);
    const bool up = !(d <= gap_u);
    const bool lo = !(d <= gap_l);
    if(!up && !lo) { continue; }
    /** New bounds of the term: the narrowed sum minus the other terms, subtracted side by side. */
    const T ntu = up ? battery::sub_up<T>(nu, battery::sub_down<T>(sl, tl)) : inf;
    const T ntl = lo ? battery::sub_down<T>(nl, battery::sub_up<T>(su, tu)) : neg_inf;
    T xl, xu;
    if(w > T{0}) {
      xl = lo ? battery::div_down<T>(ntl, w) : neg_inf;
      xu = up ? battery::div_up<T>(ntu, w) : inf;
    }
    else {
      xl = up ? battery::div_down<T>(ntu, w) : neg_inf;
      xu = lo ? battery::div_up<T>(ntl, w) : inf;
    }
    has_changed |= neurons.meet(src, xl, xu);
  }
  return has_changed;
}

} // namespace affine_v2
} // namespace jet

#endif
