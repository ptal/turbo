// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_JACOBIAN_HPP
#define TURBO_JET_JACOBIAN_HPP

#include "battery/utility.hpp"
#include "jet_symbolic.hpp"
#include "jet_sigmoid.hpp"
#include "jet_backward.hpp"

/** The interval Jacobian of the comparisons of the property with respect to the inputs, on the box of a
 * node of `-arch jet`, for the smear of `-jet_var_select smear|abs` (`jet_abs.hpp`).
 *
 * In the terms of interval constraint programming (Kearfott and Novoa, Algorithm 681 INTBIS, ACM TOMS
 * 1990; Granvilliers, Adaptive Bisection of Numerical CSPs, CP 2012), the smear of the input `x_k` is
 * `max_r max(|lb(J_rk)|, |ub(J_rk)|) * w(x_k)`, where `J` is an inclusion of the Jacobian of the functions
 * `f_r` over the box. Here the functions are the rows of the property, composed with the network:
 *   * a comparison `x - y <= k`: `f = n_x - n_y`;
 *   * an output with a bound in the region of the disjunct: `f = n_o` (the sign does not matter).
 * The derivatives are computed by reverse-mode differentiation over intervals (backpropagation): a row is
 * a vector `[glo, ghi]` over the neurons, seeded at the neurons of the row, and pushed from the layers to
 * their operands in reverse topological order:
 *   1. through the activation of the layer, whose derivative is taken from the domain of the output `y`
 *      in the store: ReLU `0` if `ub(y) <= 0`, `1` if `lb(y) > 0`, `[0, 1]` otherwise; sigmoid
 *      `y (1 - y)` over the domain of `y` clamped to `[0, 1]`; `1` without activation;
 *   2. through the operation: an affine (dense or sparse) layer `W^T g`, an `Add` to both operands, a
 *      scale `s g`.
 * When only the inputs are left, `[glo_k, ghi_k]` contains `d f / d x_k` at every point of the box where
 * the network is differentiable (the ReLU at 0 aside), which is what the heuristic needs.
 *
 * This is a heuristic: it decides which input to bisect, never what is pruned, so the arithmetic is the
 * floating point of the type `T`, without directed rounding. A non-finite derivative (an infinite bound in
 * the store, an overflow) is reported (`run` returns `false`) and the caller falls back to the widest
 * input, as Ibex's `SmearFunction` does.
 *
 * The derivatives with respect to the hidden neurons. The reverse pass goes through every neuron: before
 * the activation of the layer `l` is crossed (step 1), `[glo_n, ghi_n]` contains `d f / d y_n` for the
 * output `y_n` of every neuron of `l`. With `gy`, `max(|glo_n|, |ghi_n|)` is kept (the largest over the
 * rows): the derivative of the rows with respect to the variable `y_n` of the store, for the score of a ReLU
 * (`-relu_order smear|abs`, `-split_on mixed`, `jet_abs.hpp`).
 *
 * The relaxed derivatives (`-abs_grad relax` for the ReLUs). With the bounds `[plb_n, pub_n]` of the
 * pre-activations (`preactivation`), step 1 multiplies by the slope of the line of the linear relaxation
 * that bounds the row from below, chosen by the sign of the coefficient (CROWN, Zhang et al., NeurIPS
 * 2018; the rule of `jet_backward.hpp`): a ReLU of unknown phase `l < 0 < u` has the slope `1` if
 * `u > -l` and `0` otherwise for a coefficient `>= 0`, the chord slope `u / (u - l)` for one `< 0`; a
 * sigmoid the slopes of `sigmoid::relax`. The coefficients are then points (`glo == ghi`), and the
 * entries of the neurons (the inputs, `gy`) are those of the lower bound of the row back-substituted
 * through the relaxation: the `lambda` of the neurons in the branching scores of BaBSR (as computed by
 * alpha-beta-CROWN), without the intervals and the directed rounding of `jet_backward.hpp`, since they
 * only rank the variables.
 *
 * All the functions are run by a group of threads (`rank`, `size`, `sync`: a CUDA block, or one thread on
 * the host), and read the network through the fields of `FastNNReluNetwork` and the store through
 * `lb(i)` / `ub(i)` of `Bounds`. */
namespace jet {
namespace jacobian {

/** A row of the Jacobian: `sx * n_x + sy * n_y` (`y < 0`: no second neuron). */
struct Row {
  int x;
  int y;
  float sx;
  float sy;
};

/** The product of the intervals `[a, b]` and `[c, d]`. */
template <class T>
CUDA INLINE void mul(T a, T b, T c, T d, T& lo, T& hi) {
  const T p1 = a * c, p2 = a * d, p3 = b * c, p4 = b * d;
  lo = battery::min(battery::min(p1, p2), battery::min(p3, p4));
  hi = battery::max(battery::max(p1, p2), battery::max(p3, p4));
}

/** The derivative of the activation of the layer `L` at a neuron whose output `y` has the domain
 * `[ylb, yub]` in the store. */
template <class T, class Layer>
CUDA INLINE void activation_derivative(const Layer& L, T ylb, T yub, T& dlo, T& dhi) {
  if(L.has_relu != 0) {
    if(yub <= T{0}) { dlo = T{0}; dhi = T{0}; }
    else if(ylb > T{0}) { dlo = T{1}; dhi = T{1}; }
    else { dlo = T{0}; dhi = T{1}; }
  }
  else if(L.sigmoid != 0) {
    /** `s' = y (1 - y)`, concave in `y`, maximal (1/4) at `y = 1/2`. */
    const T a = battery::max(ylb, T{0});
    const T b = battery::min(yub, T{1});
    if(!(a <= b)) { dlo = T{0}; dhi = T{0.25}; return; }
    const T fa = a * (T{1} - a), fb = b * (T{1} - b);
    dlo = battery::min(fa, fb);
    dhi = (a <= T{0.5} && T{0.5} <= b) ? T{0.25} : battery::max(fa, fb);
  }
  else { dlo = T{1}; dhi = T{1}; }
}

template <class T>
CUDA INLINE bool finite(T v) { return v == v && v < battery::limits<T>::inf() && v > battery::limits<T>::neg_inf(); }

/** The slope of the line of the linear relaxation of the activation of `L` that bounds a row from below
 * when the coefficient of its output `y` (domain `[ylb, yub]`) is `c`, with `[l, u]` the bounds of the
 * pre-activation. */
template <class T, class Layer>
CUDA INLINE T relax_slope(const Layer& L, T c, T ylb, T yub, T l, T u) {
  if(L.has_relu != 0) {
    if(yub <= T{0} || u <= T{0}) { return T{0}; }
    if(ylb > T{0} || l >= T{0}) { return T{1}; }
    u = battery::min(u, yub);
    if(c >= T{0}) { return (u > -l) ? T{1} : T{0}; }
    if(!finite(l) || !finite(u) || !(u > l)) { return T{0}; }
    return u / (u - l);
  }
  if(L.sigmoid != 0) {
    T aL, bL, aU, bU;
    sigmoid::relax<T>(l, u, aL, bL, aU, bU);
    return c >= T{0} ? aL : aU;
  }
  return T{1};
}

/** The largest gap between the relaxation of a ReLU of unknown phase and the ReLU, with `[l, u]` the bounds
 * of its pre-activation: `-l u / (u - l)` (at `p = 0`, between the chord and the ReLU), `u` when `l` is
 * not finite, `0` when the phase is known (`l >= 0` or `u <= 0`). */
template <class T>
CUDA INLINE T relu_gap(T l, T u) {
  if(!(l < T{0}) || !(u > T{0})) { return T{0}; }
  if(!finite(l)) { return u; }
  if(!finite(u)) { return -l; }
  return static_cast<T>(-static_cast<double>(l) * static_cast<double>(u) / (static_cast<double>(u) - static_cast<double>(l)));
}

/** The bounds `[plb_i, pub_i]` of the pre-activation of every neuron `i` of a layer with an activation, by
 * the threads of `group` (the other entries are not written): the interval of the forward deduction from
 * the domains of its sources (`backward::forward_pre`), met with what the store says (`p <= y` and `p = y`
 * when `y > 0` for a ReLU; the logit of `y` for a sigmoid) and with the bounds `[xlb1_i, xub1_i]` and
 * `[xlb2_i, xub2_i]` when given (the bounds computed by the symbolic pass or the backward contractor of
 * the node, both sound on it). */
template <class T, class Group, class Net, class Bounds>
CUDA void preactivation(const Group& group, const Net& net, const Bounds& store, T* plb, T* pub,
  const T* xlb1 = nullptr, const T* xub1 = nullptr, const T* xlb2 = nullptr, const T* xub2 = nullptr)
{
  for(int l = 1; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    if(L.has_relu == 0 && L.sigmoid == 0) { continue; }
    for(int j = group.rank(); j < L.size; j += group.size()) {
      const int i = L.base + j;
      T lo, hi;
      backward::forward_pre<T>(net, store, l, j, lo, hi);
      if(xlb1) { lo = battery::max(lo, xlb1[i]); hi = battery::min(hi, xub1[i]); }
      if(xlb2) { lo = battery::max(lo, xlb2[i]); hi = battery::min(hi, xub2[i]); }
      const T ylb = static_cast<T>(store.lb(i)), yub = static_cast<T>(store.ub(i));
      if(L.sigmoid != 0) {
        lo = battery::max(lo, sigmoid::logit_lb<T>(ylb));
        hi = battery::min(hi, sigmoid::logit_ub<T>(yub));
      }
      else {
        if(ylb > T{0}) { lo = battery::max(lo, ylb); }
        hi = battery::min(hi, yub);
      }
      plb[i] = lo;
      pub[i] = hi;
    }
  }
  group.sync();
}

/** The reverse pass of one row, by the threads of `group`: `glo`, `ghi` (one entry per neuron) receive
 * the interval derivatives of the row with respect to every neuron that it depends on; the entries of the
 * inputs are the row of the Jacobian. */
template <class T, class Group, class Net, class Bounds>
CUDA void reverse_row(const Group& group, const Net& net, const Bounds& store, const Row& row, T* glo, T* ghi,
  T* gy = nullptr, const T* plb = nullptr, const T* pub = nullptr)
{
  const int N = net.num_neurons;
  for(int i = group.rank(); i < N; i += group.size()) { glo[i] = T{0}; ghi[i] = T{0}; }
  group.sync();
  if(group.rank() == 0) {
    glo[row.x] += static_cast<T>(row.sx); ghi[row.x] += static_cast<T>(row.sx);
    if(row.y >= 0) { glo[row.y] += static_cast<T>(row.sy); ghi[row.y] += static_cast<T>(row.sy); }
  }
  group.sync();
  for(int l = net.num_layers() - 1; l >= 1; --l) {
    const auto& L = net.layers[l];
    /** 1. Through the activation: the derivative with respect to the pre-activation. */
    for(int j = group.rank(); j < L.size; j += group.size()) {
      const int n = L.base + j;
      if(glo[n] == T{0} && ghi[n] == T{0}) { continue; }
      if(gy) {
        const T m = battery::max(glo[n] < T{0} ? -glo[n] : glo[n], ghi[n] < T{0} ? -ghi[n] : ghi[n]);
        gy[n] = battery::max(gy[n], m);
      }
      if(plb && (L.has_relu != 0 || L.sigmoid != 0)) {
        const T c = glo[n];
        const T v = c * relax_slope<T>(L, c, static_cast<T>(store.lb(n)), static_cast<T>(store.ub(n)), plb[n], pub[n]);
        glo[n] = v; ghi[n] = v;
        continue;
      }
      T dlo, dhi;
      activation_derivative<T>(L, static_cast<T>(store.lb(n)), static_cast<T>(store.ub(n)), dlo, dhi);
      T lo, hi;
      mul<T>(glo[n], ghi[n], dlo, dhi, lo, hi);
      glo[n] = lo; ghi[n] = hi;
    }
    group.sync();
    /** 2. Through the operation, to the operands. Each operand neuron is written by one thread. */
    const auto& S = net.layers[L.src0];
    if(L.kind == symbolic::KIND_AFFINE) {
      for(int c = group.rank(); c < S.size; c += group.size()) {
        T alo = T{0}, ahi = T{0};
        const int wb = L.weight_base + c * L.size;
        for(int j = 0; j < L.size; ++j) {
          const T gl = glo[L.base + j], gh = ghi[L.base + j];
          if(gl == T{0} && gh == T{0}) { continue; }
          const T w = static_cast<T>(net.weights[wb + j]);
          alo += (w >= T{0}) ? w * gl : w * gh;
          ahi += (w >= T{0}) ? w * gh : w * gl;
        }
        glo[S.base + c] += alo; ghi[S.base + c] += ahi;
      }
    }
    else if(L.kind == symbolic::KIND_SPARSE) {
      for(int c = group.rank(); c < S.size; c += group.size()) {
        T alo = T{0}, ahi = T{0};
        for(int e = net.sp_tcol[L.csc_base + c]; e < net.sp_tcol[L.csc_base + c + 1]; ++e) {
          const int j = net.sp_trow[e];
          const T gl = glo[L.base + j], gh = ghi[L.base + j];
          if(gl == T{0} && gh == T{0}) { continue; }
          const T w = static_cast<T>(net.sp_tval[e]);
          alo += (w >= T{0}) ? w * gl : w * gh;
          ahi += (w >= T{0}) ? w * gh : w * gl;
        }
        glo[S.base + c] += alo; ghi[S.base + c] += ahi;
      }
    }
    else if(L.kind == symbolic::KIND_ADD) {
      const auto& S1 = net.layers[L.src1];
      for(int j = group.rank(); j < L.size; j += group.size()) {
        const T gl = glo[L.base + j], gh = ghi[L.base + j];
        glo[S.base + j] += gl; ghi[S.base + j] += gh;
        glo[S1.base + j] += gl; ghi[S1.base + j] += gh;
      }
    }
    else if(L.kind == symbolic::KIND_SCALE) {
      for(int j = group.rank(); j < L.size; j += group.size()) {
        const T s = static_cast<T>(net.weights[L.weight_base + j]);
        const T gl = glo[L.base + j], gh = ghi[L.base + j];
        glo[S.base + j] += (s >= T{0}) ? s * gl : s * gh;
        ghi[S.base + j] += (s >= T{0}) ? s * gh : s * gl;
      }
    }
    group.sync();
  }
}

/** The smear of every input over the rows `rows[0..num_rows)`: `smear[k] = max_r max(|glo_rk|, |ghi_rk|)
 * * (ub(x_k) - lb(x_k))`, by the threads of `group`. `glo`, `ghi`: scratch of one entry per neuron.
 * With `gy` (one entry per neuron), `gy[n] = max_r |d row_r / d y_n|` for every non-input neuron (see the
 * top of the file); with `plb`, `pub` (`preactivation`), the derivatives are the relaxed ones.
 * \return `false` (in every thread) if a derivative or a smear is not finite. */
template <class T, class Group, class Net, class Bounds, class Rows>
CUDA bool smear(const Group& group, const Net& net, const Bounds& store, const Rows& rows, int num_rows,
  T* glo, T* ghi, T* smear_out, int* bad, T* gy = nullptr, const T* plb = nullptr, const T* pub = nullptr)
{
  const int n0 = net.num_inputs();
  for(int k = group.rank(); k < n0; k += group.size()) { smear_out[k] = T{0}; }
  if(gy) {
    for(int i = group.rank(); i < net.num_neurons; i += group.size()) { gy[i] = T{0}; }
  }
  if(group.rank() == 0) { *bad = 0; }
  group.sync();
  for(int r = 0; r < num_rows; ++r) {
    reverse_row<T>(group, net, store, rows[r], glo, ghi, gy, plb, pub);
    for(int k = group.rank(); k < n0; k += group.size()) {
      const T m = battery::max(glo[k] < T{0} ? -glo[k] : glo[k], ghi[k] < T{0} ? -ghi[k] : ghi[k]);
      if(!finite(m)) { *bad = 1; }
      else { smear_out[k] = battery::max(smear_out[k], m); }
    }
    group.sync();
  }
  for(int k = group.rank(); k < n0; k += group.size()) {
    const T w = static_cast<T>(store.ub(k)) - static_cast<T>(store.lb(k));
    const T s = smear_out[k] * w;
    if(!finite(s)) { *bad = 1; }
    smear_out[k] = s;
  }
  group.sync();
  const bool ok = *bad == 0;
  group.sync();
  return ok;
}

} // namespace jacobian
} // namespace jet

#endif
