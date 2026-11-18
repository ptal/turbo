// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_CANDIDATES_HPP
#define TURBO_JET_CANDIDATES_HPP

#include "battery/utility.hpp"
#include "jet_symbolic.hpp"
#include "jet_sigmoid.hpp"

/** The candidate points of `-jet_candidates` (`-arch jet`): a few points of the box of a node that are
 * tested as counterexamples, before the search goes on, without being nodes of the search tree.
 *
 * The candidates of a node, in this order (`num_candidates`, `applicable`, `value`):
 *   0. the middle of the box;
 *   then, only when the node ran a symbolic round (its forms, `jet_symbolic.hpp`, are those of the
 *   node), one corner of the box per constraint of the counterexample region of the disjunct:
 *   1 .. n_leq: the comparison `x - y <= k`: the corner minimizing `Low_x - Up_y` (the linear
 *      constraint of the clipping);
 *   then for each output neuron `o`, two ids: `o >= c` (a finite lower bound of the region): the
 *      corner maximizing `Up_o`; `o <= c` (a finite upper bound): the corner minimizing `Low_o`.
 * A corner takes, for every input, the end of its range given by the sign of the coefficient (the
 * middle of its interval coefficient) of the linear function to minimize or maximize, and the middle
 * of the range when the coefficient is 0 (or NaN). The forms only choose the points: whether a point
 * is a counterexample is decided by the strict test below, so stale or loose forms cost precision,
 * never soundness.
 *
 * The range of an input is the meet of its bounds in the node and of the *inner* box of the disjunct
 * (`apply_disjunct_inner`): the vnnlib bounds rounded inward, so that a candidate satisfies the
 * pre-condition of the real-valued problem, not only its outward-rounded box.
 *
 * The strict test of a point (`evaluate`, `neuron_ok`, `leq_ok`): the network is evaluated forward on
 * the point in `double`, rounded outward (not from the bounds of the node, which assume the property;
 * and not in the `float` of the store: a forward sweep in `float` on a point of 3072 inputs gives
 * output intervals of width about 1 on the SVHN models of `cora_2024`, which no counterexample can pass,
 * against about 1e-12 in `double`), and the point is a counterexample when every constraint of the region holds
 * on the whole interval of its neurons: `lb(o) >= c` for `o >= c`, `ub(o) <= c` for `o <= c` (with `c`
 * rounded inward), and `ub(x) - lb(y)` rounded up `<= k` (with `k` rounded down). This is an
 * underapproximation: a point that passes is a counterexample of the real-valued network, and a point
 * on the boundary of the region (the intervals overlap it) is rejected. The same test decides the
 * leaves of the search (`-jet_leaf_check strict`, the default; `lenient` takes any leaf whose
 * propagation is not empty). Sigmoid layers are evaluated with the enclosures of `jet_sigmoid.hpp`.
 *
 * The functions are generic, for the kernel (`check_candidates` of `jet_dive_and_solve.hpp`) and the
 * host tests: `Net` is a `FastNNReluNetwork` (or its mirror: `layers`, `weights`, `biases`,
 * `output_layer`, and `sp_row`, `sp_col`, `sp_val` for the sparse layers), `Prop` a `FastProperty` (`size()`, `leqs`), `Inner` gives `lb(i)` and `ub(i)`, `Group`
 * `rank()`, `size()` and `sync()` (as `symbolic::pass`), and `Coef(row,
 * upper, k)` the middle of the coefficient of the input `k` of the lower (or upper) form of a row.
 */
namespace jet {
namespace candidates {

/** The number of candidate ids of a node (some are not `applicable`). */
CUDA INLINE int num_candidates(int num_leqs, int num_outputs) { return 1 + num_leqs + 2 * num_outputs; }

/** The row of the forms of the output neuron `i`, or -1 if `i` is not an output neuron. */
template <class Net>
CUDA INLINE int output_row(const Net& net, const int* form_row, int i) {
  const auto& O = net.layers[net.output_layer];
  if(i < O.base || i >= O.base + O.size) { return -1; }
  return form_row[net.output_layer] + (i - O.base);
}

/** Whether the candidate `id` exists: the middle always; a corner when the forms are those of the node
 * (`use_forms`) and its constraint exists (its neurons are output neurons, its bound is finite). */
template <class T, class Net, class Prop, class Inner>
CUDA bool applicable(int id, const Net& net, const Prop& prop, const Inner& inner, const int* form_row, bool use_forms) {
  if(id == 0) { return true; }
  if(!use_forms || form_row == nullptr) { return false; }
  const int num_leqs = prop.size();
  if(id <= num_leqs) {
    const auto& leq = prop.leqs[id - 1];
    return output_row(net, form_row, leq.x) >= 0 && output_row(net, form_row, leq.y) >= 0;
  }
  const auto& O = net.layers[net.output_layer];
  const int r = id - 1 - num_leqs;
  const int o = O.base + r / 2;
  if(r / 2 >= O.size) { return false; }
  return (r % 2 == 0) ? inner.lb(o) > battery::limits<T>::neg_inf() : inner.ub(o) < battery::limits<T>::inf();
}

/** The end of `[lo, hi]` that minimizes `c * x` (the middle when `c` is 0 or NaN). */
template <class T>
CUDA INLINE T minimizer(T c, T lo, T hi) {
  if(c > T{0}) { return lo; }
  if(c < T{0}) { return hi; }
  return battery::midpoint(lo, hi);
}

/** The value of the input `k`, of range `[lo, hi]`, in the candidate `id` (`applicable`). */
template <class T, class Net, class Prop, class Coef>
CUDA T value(int id, int k, T lo, T hi, const Net& net, const Prop& prop, const int* form_row, const Coef& coef) {
  if(id == 0) { return battery::midpoint(lo, hi); }
  const int num_leqs = prop.size();
  if(id <= num_leqs) {  // minimize Low_x - Up_y.
    const auto& leq = prop.leqs[id - 1];
    const T c = coef(output_row(net, form_row, leq.x), false, k) - coef(output_row(net, form_row, leq.y), true, k);
    return minimizer<T>(c, lo, hi);
  }
  const int r = id - 1 - num_leqs;
  const int row = form_row[net.output_layer] + r / 2;
  if(r % 2 == 0) {  // o >= c: maximize Up_o, i.e. minimize -Up_o.
    return minimizer<T>(-coef(row, true, k), lo, hi);
  }
  return minimizer<T>(coef(row, false, k), lo, hi);  // o <= c: minimize Low_o.
}

/** `[a, b]` = `w * [lo, hi]` rounded outward (`w` finite). */
CUDA INLINE void product(double w, double lo, double hi, double& a, double& b) {
  if(w >= 0.0) { a = battery::mul_down<double>(w, lo); b = battery::mul_up<double>(w, hi); }
  else { a = battery::mul_down<double>(w, hi); b = battery::mul_up<double>(w, lo); }
}

/** The evaluation of the network on a point, in `double` rounded outward: `[lo[i], hi[i]]` encloses
 * the value of every neuron `i` of the real-valued network, the inputs being set by the caller
 * (`lo[k] = hi[k]` = the point). Layer by layer in topological order, the neurons of a layer by the
 * threads of `group` (a barrier between two layers). */
template <class Group, class Net>
CUDA void evaluate(Group& group, const Net& net, double* lo, double* hi) {
  for(int l = 1; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    for(int j = group.rank(); j < L.size; j += group.size()) {
      double a, b;
      if(L.kind == symbolic::KIND_ADD) {
        const int s0 = net.layers[L.src0].base + j, s1 = net.layers[L.src1].base + j;
        a = battery::add_down<double>(lo[s0], lo[s1]);
        b = battery::add_up<double>(hi[s0], hi[s1]);
      }
      else if(L.kind == symbolic::KIND_SCALE) {
        const int src = net.layers[L.src0].base + j;
        double pa, pb;
        product(static_cast<double>(net.weights[L.weight_base + j]), lo[src], hi[src], pa, pb);
        const double t = static_cast<double>(net.biases[L.bias_base + j]);
        a = battery::add_down<double>(pa, t);
        b = battery::add_up<double>(pb, t);
      }
      else if(L.kind == symbolic::KIND_SPARSE) {
        /** The stored weights of the row `j`, in increasing order of their sources (the zero weights of
         * a dense layer are skipped below too). */
        a = b = static_cast<double>(net.biases[L.bias_base + j]);
        const auto& S = net.layers[L.src0];
        int begin, end;
        symbolic::sparse_row(net, L, j, begin, end);
        for(int e = begin; e < end; ++e) {
          const double w = static_cast<double>(net.sp_val[e]);
          if(w == 0.0) { continue; }
          const int src = S.base + net.sp_col[e];
          double pa, pb;
          product(w, lo[src], hi[src], pa, pb);
          a = battery::add_down<double>(a, pa);
          b = battery::add_up<double>(b, pb);
        }
      }
      else {
        a = b = static_cast<double>(net.biases[L.bias_base + j]);
        const auto& S = net.layers[L.src0];
        for(int c = 0; c < S.size; ++c) {
          /** Weights are column-major: the weight of `(c, j)` is at `weight_base + c * size + j`. */
          const double w = static_cast<double>(net.weights[L.weight_base + c * L.size + j]);
          if(w == 0.0) { continue; }
          double pa, pb;
          product(w, lo[S.base + c], hi[S.base + c], pa, pb);
          a = battery::add_down<double>(a, pa);
          b = battery::add_up<double>(b, pb);
        }
      }
      if(L.has_relu != 0) { a = battery::max(a, 0.0); b = battery::max(b, 0.0); }
      else if(L.sigmoid != 0) { a = sigmoid::lb_d(a); b = sigmoid::ub_d(b); }
      lo[L.base + j] = a;
      hi[L.base + j] = b;
    }
    group.sync();
  }
}

/** The strict test of the neuron `i` of the evaluated point: its interval is in the inner bounds of the
 * region (no condition when they are infinite). False on NaN. */
template <class Inner>
CUDA INLINE bool neuron_ok(int i, const double* lo, const double* hi, const Inner& inner) {
  return lo[i] >= static_cast<double>(inner.lb(i)) && hi[i] <= static_cast<double>(inner.ub(i));
}

/** The strict test of the comparison `x - y <= k` (`k` rounded down) on the evaluated point. */
template <class Leq>
CUDA INLINE bool leq_ok(const Leq& leq, const double* lo, const double* hi) {
  return battery::sub_up<double>(hi[leq.x], lo[leq.y]) <= static_cast<double>(leq.k);
}

/** The middle of the interval coefficient of the input `k` of the lower (or upper) form of the row `r`,
 * in the storage of `symbolic::pass` (`layout`, `rows`): 0 when it is NaN. */
template <class T>
CUDA INLINE T coef_mid(const T* data, int layout, int n0, int rows, int r, bool upper, int k) {
  T a, b;
  if(layout == symbolic::LAYOUT_NEURON) {
    symbolic::NeuronMajor<T> f{const_cast<T*>(data), n0, static_cast<size_t>(rows)};
    const auto form = upper ? f.upper(r) : f.lower(r);
    a = form[symbolic::coef_min_slot(k)]; b = form[symbolic::coef_max_slot(n0, k)];
  }
  else {
    symbolic::SlotMajor<T> f{const_cast<T*>(data), n0, static_cast<size_t>(rows)};
    const auto form = upper ? f.upper(r) : f.lower(r);
    a = form[symbolic::coef_min_slot(k)]; b = form[symbolic::coef_max_slot(n0, k)];
  }
  const T m = a / T{2} + b / T{2};
  return (m == m) ? m : T{0};
}

} // namespace candidates
} // namespace jet

#endif
