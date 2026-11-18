// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_BACKWARD_HPP
#define TURBO_JET_BACKWARD_HPP

#include "battery/utility.hpp"
#include "jet_sigmoid.hpp"
#include "jet_symbolic.hpp"

/** The backward contractor of `-arch jet` (`-jet_backward`): back-substitution of linear rows through
 * the ReLU and sigmoid relaxations of the network (CROWN, DeepPoly), as one contractor of the fixpoint
 * of a node, next to the interval deductions and the forward symbolic pass (`jet_symbolic.hpp`).
 *
 * In the terms of continuous constraint programming, it is a linear-relaxation contractor of the
 * global constraint "the network" (as `CtcPolytopeHull(LinearizerXTaylor)` in Ibex, without the LP):
 * it reads the domains of the store, relaxes the activations on them, combines the relaxations into a
 * lower bound of a linear row over the inputs, and narrows the domains (it only meets). It keeps no
 * symbolic form: a row lives in a scratch buffer while it is back-substituted, and is dropped after.
 *
 * A row is a lower bound of a linear expression E of the neurons, valid on the region of the node
 * (the box of the store, the constraints of the property, the decisions of the search):
 *   * `ROW_NEURON_LB`: E = p_i, the pre-activation of the neuron i (y_i itself without activation);
 *   * `ROW_NEURON_UB`: E = -p_i (its lower bound is minus an upper bound of p_i);
 *   * `ROW_LEQ`: E = x - y, for a comparison x - y <= k of the property.
 * Its coefficients are over the neurons (`cmin`, `cmax`: an interval coefficient per neuron, see
 * soundness below) and a constant (`cst`, a lower bound). The back-substitution visits the layers in
 * reverse topological order, from the highest layer of the rows of a group: for the layer l,
 *   1. activation (one thread per row): the coefficients over the outputs y of l become coefficients
 *      over its pre-activations p, by the relaxation of the activation on the bounds of p:
 *        - ReLU, `p <= 0`: y = 0; `p >= 0`: y = p; unstable: `y >= p` if `u > -l`, else `y >= 0`, for
 *          a coefficient >= 0, and the chord `y <= s (p - l)` (`s >= u / (u - l)`) for one < 0;
 *        - sigmoid: the two lines of `sigmoid::relax`, `y >= aL p + bL` and `y <= aU p + bU`;
 *      then the rows whose target is a pre-activation of l are injected (+1 or -1), and the bias of l
 *      is added to the constant;
 *   2. push (all the threads, one work item per source and row): the coefficients over p become
 *      coefficients over the sources of l (`AFFINE`: through the weights, `ADD`: to both operands,
 *      `SCALE`: times the scale, its shift being the bias of step 1);
 *   3. concretization (one thread per row): the expression is then over the outputs of the layers
 *      read so far and not yet visited ("live" layers); its minimum over their domains in the store is
 *      a lower bound of E, at this depth. The best lower bound over the depths is kept: every depth is
 *      sound (the domains of the store are), and the deepest one is CROWN.
 *      A row is done when its bound proves what it is for (a comparison or an output out of its
 *      domain, the phase of a neuron with `et`, GPUPoly-like early termination); a done row costs
 *      nothing more.
 * When only the inputs are live, the row is a lower bound `sum_k a_k x_k + cst` of E over the inputs.
 *
 * Terminology: a "group" is a set of rows of ONE node back-substituted together by the threads of a
 * block (`run_rows`, `-jet_bwd_rows`), so that a weight read from memory serves all its rows. It is not
 * a batch of nodes (the domains processed together by alpha-beta-CROWN): every block works on its own
 * node.
 *
 * What a pass does with the rows (`pass`):
 *   0. the bounds of the pre-activations p of the neurons with an activation (`p_lb`, `p_ub`): the
 *      store keeps y only, and `y = max(p, 0)` loses a negative lower bound of p, which the ReLU
 *      relaxation needs. They are the interval of the forward deduction from the store, met with the
 *      bounds of the root of the disjunct (`root_p`, valid in every node of its tree), with those of
 *      the previous round of the same node (`reuse`), and with what the store says
 *      (`p <= ub(y)` and `p >= lb(y)` when `lb(y) > 0`; the logit of the bounds for a sigmoid);
 *   1. `neurons`: the hidden layers with an activation, in topological order (CROWN: a layer uses the
 *      bounds just computed for the layers below it), the two rows of every unstable neuron (a ReLU
 *      with `l < 0 < u`, a sigmoid whose p is not a point), in groups of `rows` rows;
 *   2. `spec`: the two rows of every output neuron, and the row of every comparison of the property;
 *   then, after each group, by one thread: the bounds of p are met, the store is met (`y` from the
 *   bounds of p through the activation; an output directly), a comparison with a lower bound above
 *   `k` fails the node, and, with `clip`, every row that reached the inputs clips them
 *   (Clip-and-Verify, `symbolic::clip_inputs`) with its constraint
 *        sum_k a_k x_k + cst <= E <= the upper bound of E in the store (`k` for a comparison).
 * The store and the inputs only shrink, so the rows of a later group are valid on the region too.
 *
 * Soundness in floating point. Every coefficient is an interval [cmin, cmax] that contains the real
 * coefficient of a valid real inequality, and `cst` is a lower bound of its real constant: the products
 * and sums are rounded outward (`mul_down`/`mul_up`, `add_down`/`add_up`, as `symbolic::accumulate`),
 * and the concretizations take the minimum over the interval coefficient and the domain
 * (`symbolic::interval_product_min`), rounded down. Through an activation, whose output y is >= 0
 * (ReLU) or > 0 (sigmoid), the interval [cmin, cmax] of y is first collapsed to cmin, since
 * `a y >= cmin y` for every a in it; the relaxation lines have float coefficients (the chord slope is
 * rounded up, which keeps `y <= s (p - l)` valid for `p >= l`; the lines of `sigmoid::relax` are valid
 * as written), so the products by cmin are again intervals. A coefficient or a bound that is not
 * finite where it is multiplied makes the row useless (`STATE_BAD`), never wrong. The bounds of p are
 * sound bounds of p on the region, so the relaxations hold on it.
 *
 * Parallelism and determinism. The work items of the push write distinct scratch entries; the store is
 * written by the thread 0 only (after each group of rows: meets, failures, clipping), so a pass gives the same
 * result with any number of threads. `Group` is as in `symbolic::pass` (`rank`, `size`, `sync`). */
namespace jet {
namespace backward {

constexpr int ROW_NEURON_LB = 0;
constexpr int ROW_NEURON_UB = 1;
constexpr int ROW_LEQ = 2;

constexpr int STATE_BAD = -1;
constexpr int STATE_DONE = 0;
constexpr int STATE_LIVE = 1;

/** The slots of `Scratch::misc`. */
constexpr int MISC_COUNT = 0;    // the number of unstable neurons of the current layer
constexpr int MISC_CHANGED = 1;  // the store changed
constexpr int MISC_FAILED = 2;   // the store is empty
constexpr int MISC_LIVE = 3;     // some row of the group is still live
constexpr int MISC_SIZE = 4;

struct Options {
  /** The rows of the unstable neurons of the hidden layers with an activation. */
  bool neurons = true;
  /** The rows of the output neurons and of the comparisons of the property. */
  bool spec = true;
  /** Clip the inputs with the rows that reached them. */
  bool clip = false;
  /** Early termination of a neuron row once it decides the phase of its neuron. */
  bool et = false;
  /** Keep the bounds of the pre-activations computed by the previous pass on the same node. */
  bool reuse = false;
  /** The number of rows back-substituted together (a group of rows of one node): even (the two rows of a
   * neuron are in the same group), >= 2. */
  int rows = 32;
};

/** Statistics of the passes: rows back-substituted, and multiply-adds of the pushes (an estimate). */
struct Counters {
  unsigned long long rows = 0;
  unsigned long long macs = 0;
};

/** The scratch of a pass, private to a block. `cmin`, `cmax`: `num_neurons * rows` values, the entry of
 * the neuron i and the row r at `i * rows + r` (consecutive rows of a neuron are contiguous, so the work
 * items of a push, consecutive rows of the same source, read the same weight and contiguous entries). */
template <class T>
struct Scratch {
  T* cmin;
  T* cmax;
  T* cst;
  T* best;
  T* thr;
  T* p_lb;
  T* p_ub;
  int* row_kind;
  int* row_id;
  int* row_layer;
  int* row_state;
  int* layer_live;
  int* list;
  int* misc;
};

/** The number of values and of integers of the scratch. */
CUDA INLINE size_t scratch_values(int num_neurons, int rows) {
  return size_t{2} * static_cast<size_t>(num_neurons) * static_cast<size_t>(rows)
    + size_t{3} * static_cast<size_t>(rows) + size_t{2} * static_cast<size_t>(num_neurons);
}
CUDA INLINE size_t scratch_ints(int num_layers, int max_width, int rows) {
  return size_t{4} * static_cast<size_t>(rows) + static_cast<size_t>(num_layers)
    + static_cast<size_t>(max_width) + static_cast<size_t>(MISC_SIZE);
}

/** The scratch over the storage `values` (`scratch_values`) and `ints` (`scratch_ints`). */
template <class T>
CUDA INLINE Scratch<T> make_scratch(T* values, int* ints, int num_neurons, int num_layers, int max_width, int rows) {
  Scratch<T> s;
  const size_t nr = static_cast<size_t>(num_neurons) * static_cast<size_t>(rows);
  s.cmin = values;
  s.cmax = values + nr;
  s.cst = values + 2 * nr;
  s.best = s.cst + rows;
  s.thr = s.best + rows;
  s.p_lb = s.thr + rows;
  s.p_ub = s.p_lb + num_neurons;
  s.row_kind = ints;
  s.row_id = ints + rows;
  s.row_layer = ints + 2 * rows;
  s.row_state = ints + 3 * rows;
  s.layer_live = ints + 4 * rows;
  s.list = s.layer_live + num_layers;
  s.misc = s.list + max_width;
  return s;
}

/** The number of rows of a group from `-jet_bwd_rows`: even and at least 2. */
CUDA INLINE int rows_per_group(int requested) {
  const int r = requested < 2 ? 2 : requested;
  return (r % 2 == 0) ? r : r + 1;
}

template <class Layer>
CUDA INLINE bool has_activation(const Layer& L) { return L.has_relu != 0 || L.sigmoid != 0; }

/** The layer of the neuron `i`. */
template <class Net>
CUDA INLINE int layer_of(const Net& net, int i) {
  for(int l = 0; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    if(i >= L.base && i < L.base + L.size) { return l; }
  }
  return -1;
}

/** `a := max(a, b)` and `a := min(a, b)` where a NaN `b` changes nothing. */
template <class T>
CUDA INLINE void raise(T& a, T b) { if(b > a) { a = b; } }
template <class T>
CUDA INLINE void lower(T& a, T b) { if(b < a) { a = b; } }

/** The bounds of the pre-activation of the neuron `i` of the layer `l`: the bounds `p_lb`, `p_ub` met with
 * what the store says of it; without activation, its own domain in the store. */
template <class T, class Net, class Access>
CUDA INLINE void pre_bounds(const Net& net, const Access& neurons, const Scratch<T>& s, int l, int i, T& lo, T& hi) {
  const auto& L = net.layers[l];
  if(L.sigmoid != 0) {
    lo = s.p_lb[i];
    hi = s.p_ub[i];
    raise(lo, sigmoid::logit_lb<T>(neurons.lb(i)));
    lower(hi, sigmoid::logit_ub<T>(neurons.ub(i)));
  }
  else if(L.has_relu != 0) {
    lo = s.p_lb[i];
    hi = s.p_ub[i];
    /** `p <= max(p, 0) = y` always, and `p = y` when `y > 0`. */
    if(neurons.lb(i) > T{0}) { raise(lo, neurons.lb(i)); }
    lower(hi, neurons.ub(i));
  }
  else {
    lo = neurons.lb(i);
    hi = neurons.ub(i);
  }
}

/** The interval of the forward deduction of the pre-activation of the neuron `j` of the layer `l` from
 * the domains of its sources in the store, rounded outward. */
template <class T, class Net, class Access>
CUDA INLINE void forward_pre(const Net& net, const Access& neurons, int l, int j, T& lo, T& hi) {
  using namespace symbolic;
  const auto& L = net.layers[l];
  if(L.kind == KIND_ADD) {
    const int a = net.layers[L.src0].base + j;
    const int b = net.layers[L.src1].base + j;
    lo = battery::add_down<T>(neurons.lb(a), neurons.lb(b));
    hi = battery::add_up<T>(neurons.ub(a), neurons.ub(b));
  }
  else if(L.kind == KIND_SCALE) {
    const int a = net.layers[L.src0].base + j;
    const T w = static_cast<T>(net.weights[L.weight_base + j]);
    const T t = static_cast<T>(net.biases[L.bias_base + j]);
    lo = battery::add_down<T>(interval_product_min<T>(w, w, neurons.lb(a), neurons.ub(a)), t);
    hi = battery::add_up<T>(interval_product_max<T>(w, w, neurons.lb(a), neurons.ub(a)), t);
  }
  else if(L.kind == KIND_SPARSE) {
    const auto& S = net.layers[L.src0];
    lo = static_cast<T>(net.biases[L.bias_base + j]);
    hi = lo;
    int begin, end;
    sparse_row(net, L, j, begin, end);
    for(int e = begin; e < end; ++e) {
      const T w = static_cast<T>(net.sp_val[e]);
      if(w == T{0}) { continue; }
      const int c = S.base + net.sp_col[e];
      lo = battery::add_down<T>(lo, interval_product_min<T>(w, w, neurons.lb(c), neurons.ub(c)));
      hi = battery::add_up<T>(hi, interval_product_max<T>(w, w, neurons.lb(c), neurons.ub(c)));
    }
  }
  else {
    const auto& S = net.layers[L.src0];
    lo = static_cast<T>(net.biases[L.bias_base + j]);
    hi = lo;
    for(int c = 0; c < S.size; ++c) {
      const T w = static_cast<T>(net.weights[L.weight_base + c * L.size + j]);
      if(w == T{0}) { continue; }
      lo = battery::add_down<T>(lo, interval_product_min<T>(w, w, neurons.lb(S.base + c), neurons.ub(S.base + c)));
      hi = battery::add_up<T>(hi, interval_product_max<T>(w, w, neurons.lb(S.base + c), neurons.ub(S.base + c)));
    }
  }
}

/** The coefficient [cmin, cmax] of the output y of an activation, as a coefficient [nmin, nmax] of its
 * pre-activation p in [lo, hi], plus `cadd` (a lower bound) to add to the constant, by the relaxation
 * (see the top of the file). `bad` is set when the relaxation gives no bound. */
template <class T>
CUDA INLINE void relax_coef(bool sig, T lo, T hi, T cmin, T cmax, T& nmin, T& nmax, T& cadd, bool& bad) {
  using namespace symbolic;
  nmin = T{0};
  nmax = T{0};
  cadd = T{0};
  if(cmin == T{0} && cmax == T{0}) { return; }
  /** `a y >= cmin y` for every `a >= cmin`, since `y >= 0`. */
  const T q = cmin;
  if(!is_finite(q)) { bad = true; return; }  // also NaN
  if(q == T{0}) { return; }
  if(!sig) {
    if(hi <= T{0}) { return; }                     // y = 0
    if(lo >= T{0}) { nmin = q; nmax = q; return; }  // y = p
    if(q > T{0}) {
      /** `y >= p` (always true) when `hi > -lo`, else `y >= 0` (as DeepPoly, CROWN). */
      if(hi > -lo) { nmin = q; nmax = q; }
      return;
    }
    if(is_finite(lo) && is_finite(hi)) {
      /** `y <= s (p - lo)` with `s >= hi / (hi - lo)`, so `q y >= (q s) p + (q s) (-lo)`. */
      const T slope = battery::div_up<T>(hi, battery::sub_down<T>(hi, lo));
      nmin = battery::mul_down<T>(q, slope);
      nmax = battery::mul_up<T>(q, slope);
      cadd = interval_product_min<T>(nmin, nmax, -lo, -lo);
    }
    else if(is_finite(hi)) {
      /** No chord through an infinite bound: `y <= hi`. */
      cadd = battery::mul_down<T>(q, hi);
    }
    else {
      bad = true;
    }
    if(!(cadd > battery::limits<T>::neg_inf())) { bad = true; }
    return;
  }
  /** Sigmoid: `q y >= q (aL p + bL)` for `q > 0`, `q y >= q (aU p + bU)` for `q < 0`. */
  T aL, bL, aU, bU;
  sigmoid::relax<T>(lo, hi, aL, bL, aU, bU);
  const T a = (q > T{0}) ? aL : aU;
  const T b = (q > T{0}) ? bL : bU;
  if(a != T{0}) {
    nmin = battery::mul_down<T>(q, a);
    nmax = battery::mul_up<T>(q, a);
  }
  cadd = battery::mul_down<T>(q, b);
  if(!is_finite(cadd) || !(nmin == nmin) || !(nmax == nmax)) { bad = true; }
}

/** `[cmin, cmax] += w * [amin, amax]` (a coefficient of a source from one of its reader), outward. */
template <class T>
CUDA INLINE void add_scaled_coef(T& cmin, T& cmax, T w, T amin, T amax) {
  if(w == T{0} || (amin == T{0} && amax == T{0})) { return; }
  if(w > T{0}) {
    cmin = battery::add_down<T>(cmin, battery::mul_down<T>(w, amin));
    cmax = battery::add_up<T>(cmax, battery::mul_up<T>(w, amax));
  }
  else {
    cmin = battery::add_down<T>(cmin, battery::mul_down<T>(w, amax));
    cmax = battery::add_up<T>(cmax, battery::mul_up<T>(w, amin));
  }
}

/** The back-substitution of the rows `0..m-1` of the scratch (`row_kind`, `row_id`, `row_layer`, `thr`
 * set by the thread 0), by all the threads of `group`. On return, `best[r]` is the best lower bound of the
 * row over the depths (`-inf` if none), and a row still `STATE_LIVE` reached the inputs: its
 * coefficients over the inputs and `cst[r]` are a lower bound of its expression over them. */
template <class T, class Group, class Net, class Prop, class Access>
CUDA void run_rows(Group& group, const Net& net, const Prop& prop, const Access& neurons, const Scratch<T>& s,
  int m, int R, Counters* counters)
{
  using namespace symbolic;
  const int rank = group.rank();
  const int gsize = group.size();
  /** The highest layer of the rows. */
  int top = 0;
  for(int r = 0; r < m; ++r) {
    if(s.row_kind[r] == ROW_LEQ) {
      const auto& leq = prop.leqs[s.row_id[r]];
      top = battery::max(top, battery::max(layer_of(net, leq.x), layer_of(net, leq.y)));
    }
    else {
      top = battery::max(top, s.row_layer[r]);
    }
  }
  const int end = net.layers[top].base + net.layers[top].size;
  /** Zero the coefficients of the neurons up to the highest layer. */
  for(int item = rank; item < end * m; item += gsize) {
    const int i = item / m;
    const int r = item % m;
    s.cmin[static_cast<size_t>(i) * R + r] = T{0};
    s.cmax[static_cast<size_t>(i) * R + r] = T{0};
  }
  for(int l = rank; l < net.num_layers(); l += gsize) { s.layer_live[l] = 0; }
  group.sync();
  if(rank == 0) {
    for(int r = 0; r < m; ++r) {
      s.cst[r] = T{0};
      s.best[r] = battery::limits<T>::neg_inf();
      s.row_state[r] = STATE_LIVE;
      if(s.row_kind[r] == ROW_LEQ) {
        /** `x - y`: coefficients of the outputs of the two neurons, injected before any layer. */
        const auto& leq = prop.leqs[s.row_id[r]];
        const size_t ix = static_cast<size_t>(leq.x) * R + r;
        const size_t iy = static_cast<size_t>(leq.y) * R + r;
        s.cmin[ix] = battery::add_down<T>(s.cmin[ix], T{1});
        s.cmax[ix] = battery::add_up<T>(s.cmax[ix], T{1});
        s.cmin[iy] = battery::add_down<T>(s.cmin[iy], T{-1});
        s.cmax[iy] = battery::add_up<T>(s.cmax[iy], T{-1});
        s.layer_live[layer_of(net, leq.x)] = 1;
        s.layer_live[layer_of(net, leq.y)] = 1;
      }
      else {
        s.layer_live[s.row_layer[r]] = 1;
      }
    }
    s.misc[MISC_LIVE] = 1;
  }
  group.sync();
  for(int l = top; l >= 1; --l) {
    if(s.layer_live[l] == 0) { continue; }
    const auto& L = net.layers[l];
    const bool act = has_activation(L);
    const bool sig = L.sigmoid != 0;
    /** 1. Activation, injection of the rows of this layer, bias: one thread per row. */
    for(int r = rank; r < m; r += gsize) {
      if(s.row_state[r] != STATE_LIVE) { continue; }
      bool bad = false;
      T cst = s.cst[r];
      const bool injected = s.row_kind[r] != ROW_LEQ && s.row_layer[r] == l;
      for(int j = 0; j < L.size; ++j) {
        const int i = L.base + j;
        const size_t e = static_cast<size_t>(i) * R + r;
        T amin = s.cmin[e];
        T amax = s.cmax[e];
        if(act) {
          T lo, hi;
          pre_bounds(net, neurons, s, l, i, lo, hi);
          T nmin, nmax, cadd;
          relax_coef<T>(sig, lo, hi, amin, amax, nmin, nmax, cadd, bad);
          amin = nmin;
          amax = nmax;
          if(cadd != T{0}) { cst = battery::add_down<T>(cst, cadd); }
        }
        if(injected && i == s.row_id[r]) {
          const T sign = (s.row_kind[r] == ROW_NEURON_LB) ? T{1} : T{-1};
          amin = battery::add_down<T>(amin, sign);
          amax = battery::add_up<T>(amax, sign);
        }
        if(L.kind == KIND_AFFINE || L.kind == KIND_SCALE || L.kind == KIND_SPARSE) {
          const T b = static_cast<T>(net.biases[L.bias_base + j]);
          if(b != T{0}) { cst = battery::add_down<T>(cst, interval_product_min<T>(amin, amax, b, b)); }
        }
        s.cmin[e] = amin;
        s.cmax[e] = amax;
      }
      if(bad || !(cst > battery::limits<T>::neg_inf()) || !(cst < battery::limits<T>::inf())) {
        s.row_state[r] = STATE_BAD;
      }
      s.cst[r] = cst;
    }
    group.sync();
    /** 2. Push to the sources: one work item per (source, row), consecutive rows in consecutive threads. */
    if(L.kind == KIND_AFFINE) {
      const auto& S = net.layers[L.src0];
      for(int item = rank; item < S.size * m; item += gsize) {
        const int c = item / m;
        const int r = item % m;
        if(s.row_state[r] != STATE_LIVE) { continue; }
        T acc_min = T{0};
        T acc_max = T{0};
        for(int j = 0; j < L.size; ++j) {
          const size_t e = static_cast<size_t>(L.base + j) * R + r;
          add_scaled_coef<T>(acc_min, acc_max, static_cast<T>(net.weights[L.weight_base + c * L.size + j]), s.cmin[e], s.cmax[e]);
        }
        const size_t d = static_cast<size_t>(S.base + c) * R + r;
        if(acc_min != T{0} || acc_max != T{0}) {
          s.cmin[d] = battery::add_down<T>(s.cmin[d], acc_min);
          s.cmax[d] = battery::add_up<T>(s.cmax[d], acc_max);
        }
      }
    }
    else if(L.kind == KIND_SPARSE) {
      /** As the dense layer, with the stored weights of the column `c` (`sp_tcol`), in increasing order
       * of their targets: the zero weights of the dense layer add zero to the accumulators. */
      const auto& S = net.layers[L.src0];
      for(int item = rank; item < S.size * m; item += gsize) {
        const int c = item / m;
        const int r = item % m;
        if(s.row_state[r] != STATE_LIVE) { continue; }
        T acc_min = T{0};
        T acc_max = T{0};
        int begin, end;
        sparse_column(net, L, c, begin, end);
        for(int e = begin; e < end; ++e) {
          const size_t ee = static_cast<size_t>(L.base + net.sp_trow[e]) * R + r;
          add_scaled_coef<T>(acc_min, acc_max, static_cast<T>(net.sp_tval[e]), s.cmin[ee], s.cmax[ee]);
        }
        const size_t d = static_cast<size_t>(S.base + c) * R + r;
        if(acc_min != T{0} || acc_max != T{0}) {
          s.cmin[d] = battery::add_down<T>(s.cmin[d], acc_min);
          s.cmax[d] = battery::add_up<T>(s.cmax[d], acc_max);
        }
      }
    }
    else {
      for(int item = rank; item < L.size * m; item += gsize) {
        const int j = item / m;
        const int r = item % m;
        if(s.row_state[r] != STATE_LIVE) { continue; }
        const size_t e = static_cast<size_t>(L.base + j) * R + r;
        const T amin = s.cmin[e];
        const T amax = s.cmax[e];
        const size_t d0 = static_cast<size_t>(net.layers[L.src0].base + j) * R + r;
        if(L.kind == KIND_SCALE) {
          add_scaled_coef<T>(s.cmin[d0], s.cmax[d0], static_cast<T>(net.weights[L.weight_base + j]), amin, amax);
        }
        else {  // KIND_ADD: both operands.
          add_scaled_coef<T>(s.cmin[d0], s.cmax[d0], T{1}, amin, amax);
          const size_t d1 = static_cast<size_t>(net.layers[L.src1].base + j) * R + r;
          add_scaled_coef<T>(s.cmin[d1], s.cmax[d1], T{1}, amin, amax);
        }
      }
    }
    group.sync();
    if(rank == 0) {
      s.layer_live[l] = 0;
      s.layer_live[L.src0] = 1;
      if(L.kind == KIND_ADD) { s.layer_live[L.src1] = 1; }
      if(counters != nullptr) {
        int live = 0;
        for(int r = 0; r < m; ++r) { live += (s.row_state[r] == STATE_LIVE) ? 1 : 0; }
        if(L.kind == KIND_SPARSE) {
          counters->macs += static_cast<unsigned long long>(live) * static_cast<unsigned long long>(sparse_nnz(net, L));
        }
        else {
          const unsigned long long fan = (L.kind == KIND_AFFINE) ? static_cast<unsigned long long>(net.layers[L.src0].size) : 1ull;
          counters->macs += static_cast<unsigned long long>(live) * static_cast<unsigned long long>(L.size) * fan;
        }
      }
    }
    group.sync();
    /** 3. Concretization over the live layers: one thread per row. */
    for(int r = rank; r < m; r += gsize) {
      if(s.row_state[r] != STATE_LIVE) { continue; }
      T v = s.cst[r];
      for(int ll = 0; ll < l; ++ll) {
        if(s.layer_live[ll] == 0) { continue; }
        const auto& LL = net.layers[ll];
        for(int j = 0; j < LL.size; ++j) {
          const size_t e = static_cast<size_t>(LL.base + j) * R + r;
          v = battery::add_down<T>(v, interval_product_min<T>(s.cmin[e], s.cmax[e], neurons.lb(LL.base + j), neurons.ub(LL.base + j)));
        }
      }
      if(v > s.best[r] && v < battery::limits<T>::inf()) { s.best[r] = v; }
      if(s.best[r] > s.thr[r]) { s.row_state[r] = STATE_DONE; }
    }
    group.sync();
    if(rank == 0) {
      int live = 0;
      for(int r = 0; r < m; ++r) { live += (s.row_state[r] == STATE_LIVE) ? 1 : 0; }
      s.misc[MISC_LIVE] = live;
    }
    group.sync();
    if(s.misc[MISC_LIVE] == 0) { break; }
  }
  /** A live row that did not reach the inputs (a layer was not live, a row of an input) did not end: it
   * is not over the inputs only, so it is not used for clipping. */
  if(rank == 0) {
    for(int l = 1; l < net.num_layers(); ++l) {
      if(s.layer_live[l] != 0) {
        for(int r = 0; r < m; ++r) {
          if(s.row_state[r] == STATE_LIVE) { s.row_state[r] = STATE_DONE; }
        }
        break;
      }
    }
    if(counters != nullptr) { counters->rows += static_cast<unsigned long long>(m); }
  }
  group.sync();
}

/** The upper bound of the expression of the row `r` in the store, for its threshold and its clipping:
 * the upper bound of p for a `ROW_NEURON_LB`, minus its lower bound for a `ROW_NEURON_UB`, `k` for a
 * comparison. */
template <class T, class Net, class Prop, class Access>
CUDA INLINE T row_rhs(const Net& net, const Prop& prop, const Access& neurons, const Scratch<T>& s, int r) {
  if(s.row_kind[r] == ROW_LEQ) { return static_cast<T>(prop.leqs[s.row_id[r]].k); }
  T lo, hi;
  pre_bounds(net, neurons, s, s.row_layer[r], s.row_id[r], lo, hi);
  return (s.row_kind[r] == ROW_NEURON_LB) ? hi : -lo;
}

/** After a group of rows (`run_rows`), by the thread 0: the bounds of the rows into the bounds of p and the store, the
 * failures, and the clipping. */
template <class T, class Net, class Prop, class Access>
CUDA void apply_rows(const Net& net, const Prop& prop, Access& neurons, const Scratch<T>& s, int m, int R,
  const Options& opts)
{
  using namespace symbolic;
  const int n0 = net.num_inputs();
  bool changed = false;
  bool failed = false;
  for(int r = 0; r < m && !failed; ++r) {
    const T v = s.best[r];
    if(!(v > battery::limits<T>::neg_inf())) { continue; }
    if(s.row_kind[r] == ROW_LEQ) {
      const auto& leq = prop.leqs[s.row_id[r]];
      if(v > static_cast<T>(leq.k)) {
        neurons.fail(leq.x);
        failed = true;
      }
      continue;
    }
    const int i = s.row_id[r];
    const int l = s.row_layer[r];
    if(has_activation(net.layers[l])) {
      if(s.row_kind[r] == ROW_NEURON_LB) { raise(s.p_lb[i], v); }
      else { lower(s.p_ub[i], -v); }
    }
    else {
      if(s.row_kind[r] == ROW_NEURON_LB) { changed |= neurons.meet(i, v, battery::limits<T>::inf()); }
      else { changed |= neurons.meet(i, battery::limits<T>::neg_inf(), -v); }
      if(neurons.lb(i) > neurons.ub(i)) { failed = true; }
    }
  }
  /** The outputs of the neurons with an activation, from the bounds of their pre-activation. */
  for(int r = 0; r < m && !failed; ++r) {
    if(s.row_kind[r] == ROW_LEQ) { continue; }
    const int i = s.row_id[r];
    const int l = s.row_layer[r];
    const auto& L = net.layers[l];
    if(!has_activation(L)) { continue; }
    T lo, hi;
    pre_bounds(net, neurons, s, l, i, lo, hi);
    if(lo > hi) {
      neurons.fail(i);
      failed = true;
      break;
    }
    if(L.sigmoid != 0) {
      changed |= neurons.meet(i, sigmoid::lb<T>(lo), sigmoid::ub<T>(hi));
    }
    else {
      changed |= neurons.meet(i, battery::max(lo, T{0}), battery::max(hi, T{0}));
    }
    if(neurons.lb(i) > neurons.ub(i)) { failed = true; }
  }
  /** Clipping with the rows that reached the inputs. */
  if(opts.clip && !failed) {
    for(int r = 0; r < m && !failed; ++r) {
      if(s.row_state[r] != STATE_LIVE) { continue; }
      const T rhs = row_rhs<T>(net, prop, neurons, s, r);
      changed |= clip_inputs<T>([&](int k, T& coef_min, T& coef_max) {
          coef_min = s.cmin[static_cast<size_t>(k) * R + r];
          coef_max = s.cmax[static_cast<size_t>(k) * R + r];
        }, s.cst[r], rhs, n0, neurons);
      for(int k = 0; k < n0; ++k) {
        if(!(neurons.lb(k) <= neurons.ub(k))) { failed = true; break; }
      }
    }
  }
  if(changed) { s.misc[MISC_CHANGED] = 1; }
  if(failed) {
    s.misc[MISC_FAILED] = 1;
    s.misc[MISC_CHANGED] = 1;
  }
}

/** One pass of the backward contractor on the store of `neurons`, by all the threads of `group` (see
 * the top of the file). `root_p`: the bounds of the pre-activations at the root of the disjunct
 * (`2 * num_neurons` values, the lower bounds then the upper ones), or `nullptr`. On return, the bounds
 * of the pre-activations are in `s.p_lb` and `s.p_ub` (for the root of a disjunct to save them).
 * \return in every thread: whether the store changed (also when it became empty). */
template <class T, class Group, class Net, class Prop, class Access>
CUDA bool pass(Group& group, const Net& net, const Prop& prop, Access& neurons, const Scratch<T>& s,
  const T* root_p, const Options& opts, Counters* counters = nullptr)
{
  using namespace symbolic;
  const int rank = group.rank();
  const int gsize = group.size();
  const int N = net.num_neurons;
  const int n0 = net.num_inputs();
  const int R = rows_per_group(opts.rows);
  if(rank == 0) {
    s.misc[MISC_CHANGED] = 0;
    s.misc[MISC_FAILED] = 0;
  }
  /** 0. The bounds of the pre-activations. */
  for(int l = 1; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    if(!has_activation(L)) { continue; }
    for(int j = rank; j < L.size; j += gsize) {
      const int i = L.base + j;
      if(!opts.reuse) {
        s.p_lb[i] = root_p ? root_p[i] : battery::limits<T>::neg_inf();
        s.p_ub[i] = root_p ? root_p[N + i] : battery::limits<T>::inf();
      }
      T lo, hi;
      forward_pre<T>(net, neurons, l, j, lo, hi);
      raise(s.p_lb[i], lo);
      lower(s.p_ub[i], hi);
    }
  }
  group.sync();
  if(rank == 0) {
    for(int l = 1; l < net.num_layers() && s.misc[MISC_FAILED] == 0; ++l) {
      if(!has_activation(net.layers[l])) { continue; }
      for(int j = 0; j < net.layers[l].size; ++j) {
        T lo, hi;
        pre_bounds(net, neurons, s, l, net.layers[l].base + j, lo, hi);
        if(lo > hi) {
          neurons.fail(net.layers[l].base + j);
          s.misc[MISC_FAILED] = 1;
          s.misc[MISC_CHANGED] = 1;
          break;
        }
      }
    }
  }
  group.sync();
  if(s.misc[MISC_FAILED] != 0) { return true; }
  /** 1. The unstable neurons of the hidden layers with an activation, layer by layer. */
  if(opts.neurons) {
    for(int l = 1; l < net.num_layers(); ++l) {
      const auto& L = net.layers[l];
      if(!has_activation(L) || l == net.output_layer) { continue; }
      if(rank == 0) {
        int count = 0;
        for(int j = 0; j < L.size; ++j) {
          T lo, hi;
          pre_bounds(net, neurons, s, l, L.base + j, lo, hi);
          const bool unstable = (L.sigmoid != 0) ? (lo < hi) : (lo < T{0} && hi > T{0});
          if(unstable) { s.list[count++] = L.base + j; }
        }
        s.misc[MISC_COUNT] = count;
      }
      group.sync();
      const int num_rows = 2 * s.misc[MISC_COUNT];
      /** Every thread has read the count before the thread 0 writes the next one. */
      group.sync();
      for(int start = 0; start < num_rows; start += R) {
        const int m = battery::min(R, num_rows - start);
        if(rank == 0) {
          for(int r = 0; r < m; ++r) {
            const int q = start + r;
            s.row_kind[r] = (q % 2 == 0) ? ROW_NEURON_LB : ROW_NEURON_UB;
            s.row_id[r] = s.list[q / 2];
            s.row_layer[r] = l;
            /** With `et`, a row is done when it decides the phase of its neuron; otherwise when it proves
             * that the store is empty (above the upper bound of its expression). */
            s.thr[r] = opts.et ? T{0} : row_rhs<T>(net, prop, neurons, s, r);
          }
        }
        group.sync();
        run_rows<T>(group, net, prop, neurons, s, m, R, counters);
        if(rank == 0) { apply_rows<T>(net, prop, neurons, s, m, R, opts); }
        group.sync();
        if(s.misc[MISC_FAILED] != 0) { return true; }
      }
    }
  }
  /** 2. The output neurons (two rows each) and the comparisons of the property. */
  if(opts.spec) {
    const auto& O = net.layers[net.output_layer];
    const int num_rows = 2 * O.size + prop.size();
    for(int start = 0; start < num_rows; start += R) {
      const int stop = battery::min(num_rows, start + R);
      if(rank == 0) {
        int m = 0;
        for(int q = start; q < stop; ++q) {
          if(q < 2 * O.size) {
            s.row_kind[m] = (q % 2 == 0) ? ROW_NEURON_LB : ROW_NEURON_UB;
            s.row_id[m] = O.base + q / 2;
            s.row_layer[m] = net.output_layer;
          }
          else {
            const int p = q - 2 * O.size;
            const auto& leq = prop.leqs[p];
            /** The comparisons on the inputs are the business of the interval deduction. */
            if(leq.x < n0 || leq.y < n0) { continue; }
            s.row_kind[m] = ROW_LEQ;
            s.row_id[m] = p;
            s.row_layer[m] = -1;
          }
          s.thr[m] = row_rhs<T>(net, prop, neurons, s, m);
          ++m;
        }
        s.misc[MISC_COUNT] = m;
      }
      group.sync();
      const int m = s.misc[MISC_COUNT];
      group.sync();
      if(m > 0) {
        run_rows<T>(group, net, prop, neurons, s, m, R, counters);
        if(rank == 0) { apply_rows<T>(net, prop, neurons, s, m, R, opts); }
        group.sync();
        if(s.misc[MISC_FAILED] != 0) { return true; }
      }
    }
  }
  group.sync();
  const bool changed = s.misc[MISC_CHANGED] != 0;
  group.sync();
  return changed;
}

} // namespace backward
} // namespace jet

#endif
