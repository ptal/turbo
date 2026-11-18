// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_SYMBOLIC_HPP
#define TURBO_JET_SYMBOLIC_HPP

#include "battery/utility.hpp"
#include "jet_sigmoid.hpp"
#include <algorithm>
#include <vector>

/** Symbolic bounds of JET (design 2), enabled with `-jet_symbolic`.
 *
 * Every non-input neuron `y` gets two affine *forms* over the `n0` input variables, the lower form
 * `Low` and the upper form `Up`, each with an interval coefficient `[coef_min_k, coef_max_k]` per
 * input `k` and an interval constant `[const_min, const_max]`:
 *
 *   Low(x) = sum_k [coef_min_k, coef_max_k] * x_k + [const_min, const_max],
 *            with  y >= min_{a_k in [coef_min_k, coef_max_k]} sum_k a_k x_k + const_min,
 *   Up(x)  = sum_k [coef_min_k, coef_max_k] * x_k + [const_min, const_max],
 *            with  y <= max_{a_k in [coef_min_k, coef_max_k]} sum_k a_k x_k + const_max,
 *
 * for every solution `x` of the current box. The coefficients are intervals and every operation
 * is rounded outward, so the floating-point errors made while composing the forms are already
 * accounted for and the forms are sound for the real-valued network.
 *
 * One pass (`pass`) does, layer by layer in topological order (steps 1 and 2-4 are two steps of
 * the threads, separated by a barrier, see `pass`):
 *   1. compose the forms of the pre-activation from the forms of the source neurons (an input
 *      neuron `k` has the exact form `x_k`, which is not stored);
 *   2. concretize them over the input box to get bounds of the pre-activation;
 *   3. relax the ReLU as DeepPoly: the chord as upper bound (its slope rounded up), and either
 *      `y >= p` or `y >= 0` as lower bound (`y >= p` when `p_ub > -p_lb`);
 *      a sigmoid by the two lines of `sigmoid::relax` (`jet_sigmoid.hpp`), whose slopes are >= 0;
 *   4. meet the bounds of the neuron into the store.
 * With `clip_hidden` (`-jet_sym_clip_hidden`), it also clips the input box, during step 3, with the
 * forms of every hidden neuron whose bounds come from elsewhere than its forms and are tighter (the
 * interval deduction, the phase of a ReLU split): see `clip_hidden`.
 * Then (unless `-forward_only`, see `pass`) it clips the input box with every bounded output neuron
 * and every comparison of the property, with the closed form of the relaxed clipping of Clip-and-Verify (Theorem 3.2,
 * `_clip_main_fn` of `complete_verifier/input_split/clip.py` in alpha-beta-CROWN) applied to these
 * short constraints over the inputs:  Low_o(x) <= ub(o),  -Up_o(x) <= -lb(o),  Low_x(x) - Up_y(x) <= k.
 * Unlike Clip-and-Verify, the forms are composed forward (above), not by back-substitution, and the
 * coefficients are intervals. As in the backward step of JET (`FastNNRelu::deduce_neuron`), the
 * sum of the terms of a constraint is computed once, and the sum of the other terms of an input is
 * recovered by subtracting its own term on the same side (see `clip_inputs`).
 *
 * Naming. Three pairs of words are used, each for one thing only:
 *   * `lower` / `upper`: the lower form `Low` and the upper form `Up` of a neuron;
 *   * `min` / `max`: the two ends of an interval of the symbolic arithmetic: the coefficient
 *     `[coef_min, coef_max]` of an input, the constant `[const_min, const_max]`, a term
 *     `[term_min, term_max]`, the product of two intervals (`interval_product_min`,
 *     `interval_product_max`);
 *   * `lb` / `ub`: the bounds of a variable or of a value: the store (`neurons.lb(i)`), the
 *     pre-activation `p` (`p_lb`, `p_ub`), a form over the box (`concretize_lb`, `concretize_ub`).
 * `target` is the neuron being computed (as in `deduce_neuron`), `src` a neuron it reads, `S` the
 * layer of the sources of an affine layer and `c` the index of a source in `S`.
 *
 * Infinities and NaN. The weights and biases are assumed finite (they come from the ONNX file), and
 * the bounds of the inputs are finite (vnnlib), but a form can still overflow to an infinity, and an
 * infinity can then give NaN (`0 * inf`, `inf - inf`, `inf / inf`). The pass never lets NaN, or an
 * infinity where it means nothing, reach a decision:
 *   * a product of interval ends with a zero factor is 0 (`safe_mul_down`, `safe_mul_up`), as in
 *     interval arithmetic, and a NaN coefficient gives an infinite product (`interval_product_min`,
 *     `interval_product_max`);
 *   * a form whose bound over the box is NaN, or infinite on the wrong side, gives no bound
 *     (`concretize_lb` returns -inf, `concretize_ub` returns +inf);
 *   * the chord of a ReLU needs finite bounds of `p`: otherwise the upper form of `y` becomes the
 *     constant `p_ub`; and a NaN bound of `p` would be taken as an unstable ReLU, never as an
 *     active one;
 *   * a constraint with an infinite or NaN right-hand side or constant clips nothing, an input
 *     whose coefficient is infinite, NaN or of unknown sign is not clipped, and the terms that are
 *     not finite are counted instead of summed (see `clip_inputs`).
 * The meets of the store ignore a NaN bound too, and an infinite bound means "no information".
 *
 * Memory: `rows` rows of two forms, `rows * (4 * n0 + 4)` values, plus one flag per row (both forms
 * of the row are constant). With the plan of the network (`plan_form_rows`, `FastNNReluNetwork::
 * form_row`), a layer has rows only while it is read: the layers of a chain share two layers of rows
 * (plus the output layer), and a scale of the inputs (`ROW_VIRTUAL`) has none. Without plan, every
 * non-input neuron `i` has the row `i - n0`.
 * A form has `form_size(n0) = 2 * n0 + 2` *slots*: `coef_min_k` at `coef_min_slot(k) = k`,
 * `coef_max_k` at `coef_max_slot(n0, k) = n0 + k`, `const_min` at `const_min_slot(n0) = 2 n0` and
 * `const_max` at `const_max_slot(n0) = 2 n0 + 1`. Only `const_min` of a lower form and `const_max` of
 * an upper form are ever read for a bound; the other end of the constant is kept so that the
 * coefficients and the constant are computed by the same code.
 * Two layouts store the same values, chosen with `-jet_sym_layout` (`layout` of `pass`):
 *   * `NeuronMajor` (`LAYOUT_NEURON`): the slots of a row are contiguous, the lower form then
 *     the upper form: slot `s` of the lower form of the row `r` is at `r * (4 n0 + 4) + s`.
 *   * `SlotMajor` (`LAYOUT_SLOT`, default): a slot of all the rows is contiguous: slot `s` of
 *     the lower form of the row `r` is at `s * rows + r`, of the upper form at
 *     `(form_size(n0) + s) * rows + r`.
 *     In the composition, consecutive threads take consecutive targets of a layer for the same slots,
 *     so with this layout they read the same values of a source and write consecutive addresses
 *     (coalesced accesses).
 * The composition of the forms does not depend on the layout nor on the plan, so they all give the
 * same forms and, on one thread, the same store. With several threads, the clipping of the inputs
 * reads bounds that other threads may be tightening at the same time; bounds only shrink, so every
 * value read is sound, but how much is clipped in one pass can depend on the timing.
 *
 * The functions are generic so that they can be tested on the host:
 *   * `Group` provides `int rank()`, `int size()` and `void sync()` (a block barrier on the GPU);
 *   * `Net` is a `FastNNReluNetwork` (fields `layers`, `weights`, `biases`, `output_layer`,
 *     `num_neurons`, methods `num_inputs()`, `num_layers()`);
 *   * `Prop` is a `FastProperty` (`size()`, `leqs[i].x`, `.y`, `.k`);
 *   * `Access` is `FastNNRelu::NeuronsAccess` (`lb`, `ub`, `meet`, `fail`).
 */
namespace jet {
namespace symbolic {

/** Must be the values of `jet::LayerKind` (checked by a `static_assert` in `jet_dive_and_solve.hpp`). */
constexpr int KIND_INPUT = 0;
constexpr int KIND_AFFINE = 1;
constexpr int KIND_ADD = 2;
constexpr int KIND_SCALE = 3;
/** A sparse affine layer \f$ W \cdot src0 + b \f$ (a `Conv` of the ONNX graph, lowered by the parsing):
 * only the nonzero weights are stored, by rows (CSR, `sp_row`, `sp_col`, `sp_val` of the network, the
 * rows of the layer at `weight_base`) and by columns (CSC, `sp_tcol`, `sp_trow`, `sp_tval`, at
 * `csc_base`, built by the network from the rows). A row lists its sources in increasing order, so that
 * a sum over a row adds the same terms in the same order as the dense layer with the same weights
 * (`tests/sparse_layer_test.cpp`). */
constexpr int KIND_SPARSE = 4;

/** Whether the layer `L` is affine, dense or sparse (`W src0 + b`). */
template <class Layer>
CUDA INLINE bool is_affine_kind(const Layer& L) { return L.kind == KIND_AFFINE || L.kind == KIND_SPARSE; }

/** The entries `[begin, end)` of the row `j` of the sparse layer `L` in `sp_col` and `sp_val`. */
template <class Net, class Layer>
CUDA INLINE void sparse_row(const Net& net, const Layer& L, int j, int& begin, int& end) {
  begin = net.sp_row[L.weight_base + j];
  end = net.sp_row[L.weight_base + j + 1];
}

/** The entries `[begin, end)` of the column `c` (the source `c` of `L.src0`) of the sparse layer `L` in
 * `sp_trow` and `sp_tval`. */
template <class Net, class Layer>
CUDA INLINE void sparse_column(const Net& net, const Layer& L, int c, int& begin, int& end) {
  begin = net.sp_tcol[L.csc_base + c];
  end = net.sp_tcol[L.csc_base + c + 1];
}

/** The number of stored weights of the sparse layer `L`. */
template <class Net, class Layer>
CUDA INLINE int sparse_nnz(const Net& net, const Layer& L) {
  return net.sp_row[L.weight_base + L.size] - net.sp_row[L.weight_base];
}

/** Builds the columns (CSC) of the sparse layers of a network from their rows: for each sparse layer,
 * `csc_base` is set and its columns are appended to `tcol`, `trow` and `tval`, each column listing its
 * targets in increasing order. Host only. `Layers`, `Ints`, `Floats` are vectors (`push_back`, `[]`). */
template <class Layers, class Ints, class Floats>
void build_sparse_columns(Layers& layers, const Ints& row, const Ints& col, const Floats& val,
  Ints& tcol, Ints& trow, Floats& tval)
{
  tcol.clear(); trow.clear(); tval.clear();
  for(int l = 0; l < static_cast<int>(layers.size()); ++l) {
    auto& L = layers[l];
    if(L.kind != KIND_SPARSE) { continue; }
    const int fan = layers[L.src0].size;
    L.csc_base = static_cast<int>(tcol.size());
    /** Counting sort of the entries by column; the rows are visited in order, so each column lists its
     * targets in increasing order. */
    std::vector<int> count(static_cast<size_t>(fan) + 1, 0);
    for(int j = 0; j < L.size; ++j) {
      for(int e = row[L.weight_base + j]; e < row[L.weight_base + j + 1]; ++e) { ++count[col[e] + 1]; }
    }
    for(int c = 0; c < fan; ++c) { count[c + 1] += count[c]; }
    const int base = static_cast<int>(trow.size());
    for(int c = 0; c <= fan; ++c) { tcol.push_back(base + count[c]); }
    const int nnz = count[fan];
    for(int e = 0; e < nnz; ++e) { trow.push_back(0); tval.push_back(0.0f); }
    std::vector<int> next(count.begin(), count.end() - 1);
    for(int j = 0; j < L.size; ++j) {
      for(int e = row[L.weight_base + j]; e < row[L.weight_base + j + 1]; ++e) {
        const int pos = base + next[col[e]]++;
        trow[pos] = j;
        tval[pos] = val[e];
      }
    }
  }
}

/** Values of `-jet_sym_layout`. */
constexpr int LAYOUT_NEURON = 0;
constexpr int LAYOUT_SLOT = 1;

CUDA INLINE constexpr int form_size(int n0) { return 2 * n0 + 2; }
CUDA INLINE constexpr int neuron_stride(int n0) { return 2 * form_size(n0); }
/** The storage of `rows` rows of forms: the two forms of each row, then one flag per row (whether
 * both forms of the row are constant, see `pass`). */
CUDA INLINE constexpr size_t storage_size_rows(int rows, int n0) {
  return static_cast<size_t>(rows) * static_cast<size_t>(neuron_stride(n0)) + static_cast<size_t>(rows);
}
/** The storage when every non-input neuron has its own row (no plan, see `plan_form_rows`). */
CUDA INLINE constexpr size_t storage_size(int num_neurons, int n0) {
  return storage_size_rows(num_neurons - n0, n0);
}

/** Values of `layer_row` (see `plan_form_rows`) that are not a row. */
constexpr int ROW_INPUT = -1;    /**< the input layer: the form of the input `k` is `x_k`, never stored. */
constexpr int ROW_VIRTUAL = -2;  /**< a scale of the inputs: the forms `s_j x_j + t_j` are not stored. */

/** Whether the layer `l` is a scale of the inputs that `pass` does not store (`ROW_VIRTUAL`): its forms
 * `s_j * x_j + t_j` are rebuilt on the fly by the layers that read it, which turns the dense
 * composition of the next affine layer (one full form per input, `cora_2024`) into one term per slot. */
template <class L>
CUDA INLINE bool is_virtual_scale(const L& layer, int l, int output_layer) {
  return layer.kind == KIND_SCALE && layer.src0 == 0 && layer.has_relu == 0 && layer.sigmoid == 0 && l != output_layer;
}

/** The rows of the forms, layer by layer (`layer_row[l]` is the row of the first neuron of the layer
 * `l`, whose neurons have consecutive rows), and their number. A layer is live from the moment it is
 * computed until the last layer that reads it (the output layer until the end of the pass, for the
 * clipping): two layers share rows only when the first is no longer live when the second is computed
 * (first fit). The input layer is `ROW_INPUT`, a scale of the inputs `ROW_VIRTUAL`. With a chain of
 * layers, two layers of rows (plus the output layer) are enough, instead of every neuron.
 * Host only. */
template <class Layers>
int plan_form_rows(const Layers& layers, int num_layers, int output_layer, std::vector<int>& layer_row) {
  layer_row.assign(static_cast<size_t>(num_layers), ROW_INPUT);
  std::vector<int> last_use(static_cast<size_t>(num_layers), 0);
  for(int l = 1; l < num_layers; ++l) {
    last_use[l] = std::max(last_use[l], l);
    /** No range-for over a braced list here: nvcc's front end (cudafe) rewrites it into code that gcc
     * rejects. */
    const int srcs[2] = {layers[l].src0, layers[l].src1};
    for(int k = 0; k < 2; ++k) {
      if(srcs[k] > 0) { last_use[srcs[k]] = std::max(last_use[srcs[k]], l); }
    }
  }
  if(output_layer > 0 && output_layer < num_layers) { last_use[output_layer] = num_layers; }
  int rows = 0;
  for(int l = 1; l < num_layers; ++l) {
    if(is_virtual_scale(layers[l], l, output_layer)) { layer_row[l] = ROW_VIRTUAL; continue; }
    const int size = layers[l].size;
    /** The candidates are 0 and the end of every live layer; the first that overlaps no live layer. */
    auto fits = [&](int start) {
      for(int p = 1; p < l; ++p) {
        if(layer_row[p] < 0 || last_use[p] < l) { continue; }
        if(start < layer_row[p] + layers[p].size && layer_row[p] < start + size) { return false; }
      }
      return true;
    };
    int best = -1;
    if(fits(0)) { best = 0; }
    for(int p = 1; p < l; ++p) {
      if(layer_row[p] < 0 || last_use[p] < l) { continue; }
      const int candidate = layer_row[p] + layers[p].size;
      if((best == -1 || candidate < best) && fits(candidate)) { best = candidate; }
    }
    layer_row[l] = best;
    rows = std::max(rows, best + size);
  }
  return rows;
}

/** The slots of a form: the coefficient `[coef_min_k, coef_max_k]` of the input `k`, and the constant
 * `[const_min, const_max]`. */
CUDA INLINE constexpr int coef_min_slot(int k) { return k; }
CUDA INLINE constexpr int coef_max_slot(int n0, int k) { return n0 + k; }
CUDA INLINE constexpr int const_min_slot(int n0) { return 2 * n0; }
CUDA INLINE constexpr int const_max_slot(int n0) { return 2 * n0 + 1; }

/** A form whose slots are contiguous: slot `s` is `ptr[s]`. */
template <class T>
struct ContiguousForm {
  T* ptr;
  CUDA INLINE T& operator[](int s) const { return ptr[s]; }
};

/** A form whose slots are `stride` values apart: slot `s` is `ptr[s * stride]`. */
template <class T>
struct StridedForm {
  T* ptr;
  size_t stride;
  CUDA INLINE T& operator[](int s) const { return ptr[static_cast<size_t>(s) * stride]; }
};

/** The forms of the row `r` are `lower(r)` and `upper(r)`; `flag(r)` says whether both are constant. */
template <class T>
struct NeuronMajor {
  using form = ContiguousForm<T>;
  static constexpr bool slot_major = false;
  T* data;
  int n0;
  size_t rows;
  CUDA INLINE form lower(int r) const {
    return form{data + static_cast<size_t>(r) * static_cast<size_t>(neuron_stride(n0))};
  }
  CUDA INLINE form upper(int r) const {
    return form{data + static_cast<size_t>(r) * static_cast<size_t>(neuron_stride(n0)) + form_size(n0)};
  }
  CUDA INLINE T& flag(int r) const { return data[rows * static_cast<size_t>(neuron_stride(n0)) + static_cast<size_t>(r)]; }
};

template <class T>
struct SlotMajor {
  using form = StridedForm<T>;
  static constexpr bool slot_major = true;
  T* data;
  int n0;
  size_t stride;  /**< the number of rows, which is the distance between two slots of a form. */
  CUDA INLINE form lower(int r) const {
    return form{data + static_cast<size_t>(r), stride};
  }
  CUDA INLINE form upper(int r) const {
    return form{data + static_cast<size_t>(form_size(n0)) * stride + static_cast<size_t>(r), stride};
  }
  CUDA INLINE T& flag(int r) const { return data[stride * static_cast<size_t>(neuron_stride(n0)) + static_cast<size_t>(r)]; }
};

/** The forms of a scale of the inputs (`ROW_VIRTUAL`), read-only: `s * x_c + t`, as the pass of
 * before r14 stored them (`set_const(t)` then `add_input(c, s)`). */
template <class T>
struct VirtualScaleForm {
  T s;
  T t;
  int c;
  int n0;
  CUDA INLINE T operator[](int slot) const {
    if(slot >= 2 * n0) { return t; }
    const int k = slot < n0 ? slot : slot - n0;
    return (k == c && s != T{0}) ? s : T{0};
  }
};

/** `x` is neither infinite nor NaN. */
template <class T>
CUDA INLINE bool is_finite(T x) {
  return x > battery::limits<T>::neg_inf() && x < battery::limits<T>::inf();
}

/** The constant form `[const_min, const_max]` (all the coefficients are 0). */
template <class T, class F>
CUDA INLINE void set_const(F form, int n0, T const_min, T const_max) {
  for(int k = 0; k < n0; ++k) {
    form[coef_min_slot(k)] = T{0};
    form[coef_max_slot(n0, k)] = T{0};
  }
  form[const_min_slot(n0)] = const_min;
  form[const_max_slot(n0)] = const_max;
}

/** `form += w * x_k`, the exact form of the input neuron `k`. */
template <class T, class F>
CUDA INLINE void add_input(F form, int n0, int k, T w) {
  form[coef_min_slot(k)] = battery::add_down<T>(form[coef_min_slot(k)], w);
  form[coef_max_slot(n0, k)] = battery::add_up<T>(form[coef_max_slot(n0, k)], w);
}

/** `form += w * src`, rounded outward. With `w < 0`, `src` must be the form of the opposite side. */
template <class T, class F, class G>
CUDA INLINE void add_scaled(F form, G src, int n0, T w) {
  for(int k = 0; k <= n0; ++k) {
    /** k < n0: the coefficient of the input k; k == n0: the constant. */
    const int slot_min = (k < n0) ? coef_min_slot(k) : const_min_slot(n0);
    const int slot_max = (k < n0) ? coef_max_slot(n0, k) : const_max_slot(n0);
    T term_min, term_max;
    if(w >= T{0}) {
      term_min = battery::mul_down<T>(w, src[slot_min]);
      term_max = battery::mul_up<T>(w, src[slot_max]);
    }
    else {
      term_min = battery::mul_down<T>(w, src[slot_max]);
      term_max = battery::mul_up<T>(w, src[slot_min]);
    }
    form[slot_min] = battery::add_down<T>(form[slot_min], term_min);
    form[slot_max] = battery::add_up<T>(form[slot_max], term_max);
  }
}

/** A product of two interval ends, rounded down / up, where a zero factor gives 0 even when the
 * other factor is infinite (an infinite end means "unbounded", and 0 times any real is 0). */
template <class T>
CUDA INLINE T safe_mul_down(T a, T b) {
  return (a == T{0} || b == T{0}) ? T{0} : battery::mul_down<T>(a, b);
}

template <class T>
CUDA INLINE T safe_mul_up(T a, T b) {
  return (a == T{0} || b == T{0}) ? T{0} : battery::mul_up<T>(a, b);
}

/** The smallest and the largest value of `a * x` for `a in [coef_min, coef_max]` and
 * `x in [x_lb, x_ub]` (the ends of the product of the two intervals), rounded outward. A NaN
 * coefficient means "unknown": the product is then unbounded (the min and max of the GPU ignore a NaN operand, so it
 * must be caught before). The bounds of a variable of the store are never NaN. */
template <class T>
CUDA INLINE T interval_product_min(T coef_min, T coef_max, T x_lb, T x_ub) {
  if(coef_min == T{0} && coef_max == T{0}) { return T{0}; }
  if(!(coef_min == coef_min) || !(coef_max == coef_max)) { return battery::limits<T>::neg_inf(); }
  return battery::min(battery::min(safe_mul_down<T>(coef_min, x_lb), safe_mul_down<T>(coef_min, x_ub)),
                      battery::min(safe_mul_down<T>(coef_max, x_lb), safe_mul_down<T>(coef_max, x_ub)));
}

template <class T>
CUDA INLINE T interval_product_max(T coef_min, T coef_max, T x_lb, T x_ub) {
  if(coef_min == T{0} && coef_max == T{0}) { return T{0}; }
  if(!(coef_min == coef_min) || !(coef_max == coef_max)) { return battery::limits<T>::inf(); }
  return battery::max(battery::max(safe_mul_up<T>(coef_min, x_lb), safe_mul_up<T>(coef_min, x_ub)),
                      battery::max(safe_mul_up<T>(coef_max, x_lb), safe_mul_up<T>(coef_max, x_ub)));
}

/** Lower bound of a lower form over the input box (the inputs are the neurons `0..n0-1`).
 * NaN (`-inf + inf`, a NaN constant) and `+inf` say nothing on a lower bound: `-inf` is returned. */
template <class T, class F, class Access>
CUDA INLINE T concretize_lb(F form, int n0, const Access& neurons) {
  T sum = form[const_min_slot(n0)];
  for(int k = 0; k < n0; ++k) {
    sum = battery::add_down<T>(sum, interval_product_min<T>(form[coef_min_slot(k)], form[coef_max_slot(n0, k)],
      neurons.lb(k), neurons.ub(k)));
  }
  return (sum < battery::limits<T>::inf()) ? sum : battery::limits<T>::neg_inf();
}

/** Upper bound of an upper form over the input box (NaN and `-inf` give `+inf`). */
template <class T, class F, class Access>
CUDA INLINE T concretize_ub(F form, int n0, const Access& neurons) {
  T sum = form[const_max_slot(n0)];
  for(int k = 0; k < n0; ++k) {
    sum = battery::add_up<T>(sum, interval_product_max<T>(form[coef_min_slot(k)], form[coef_max_slot(n0, k)],
      neurons.lb(k), neurons.ub(k)));
  }
  return (sum > battery::limits<T>::neg_inf()) ? sum : battery::limits<T>::inf();
}

/** In place: `form := a * form + b` with `a > 0` finite (the forms of a sigmoid from those of its
 * pre-activation, `sigmoid::relax`): the coefficients and the constant are scaled outward, and `b`
 * is added rounded down to `const_min` and up to `const_max`. A lower (upper) form stays a lower
 * (upper) form because `a >= 0`. */
template <class T, class F>
CUDA INLINE void scale_shift(F form, int n0, T a, T b) {
  for(int k = 0; k < n0; ++k) {
    form[coef_min_slot(k)] = battery::mul_down<T>(a, form[coef_min_slot(k)]);
    form[coef_max_slot(n0, k)] = battery::mul_up<T>(a, form[coef_max_slot(n0, k)]);
  }
  form[const_min_slot(n0)] = battery::add_down<T>(battery::mul_down<T>(a, form[const_min_slot(n0)]), b);
  form[const_max_slot(n0)] = battery::add_up<T>(battery::mul_up<T>(a, form[const_max_slot(n0)]), b);
}

/** In place: `form := slope * (form - p_lb)` (the upper form of an unstable ReLU), with `slope > 0`
 * and `p_lb` finite. */
template <class T, class F>
CUDA INLINE void chord(F form, int n0, T slope, T p_lb) {
  for(int k = 0; k < n0; ++k) {
    form[coef_min_slot(k)] = battery::mul_down<T>(slope, form[coef_min_slot(k)]);
    form[coef_max_slot(n0, k)] = battery::mul_up<T>(slope, form[coef_max_slot(n0, k)]);
  }
  form[const_min_slot(n0)] = battery::mul_down<T>(slope, battery::sub_down<T>(form[const_min_slot(n0)], p_lb));
  form[const_max_slot(n0)] = battery::mul_up<T>(slope, battery::sub_up<T>(form[const_max_slot(n0)], p_lb));
}

/** Clip the input `m`, of coefficient `[coef_min, coef_max]`, with
 *   min_{a in [coef_min, coef_max]} a * x_m <= budget - partial,
 * where `budget` is an upper bound of the sum of all the terms and `partial` a lower bound of the sum of the other
 * terms (both finite). This is the step of Theorem 3.2 of Clip-and-Verify for one input.
 * \return `true` if the bound of `x_m` changed. */
template <class T, class Access>
CUDA INLINE bool clip_input(T coef_min, T coef_max, T budget, T partial, int m, Access& neurons) {
  /** The sign of the coefficient must be known, and its ends finite (also false on NaN). */
  if(!is_finite(coef_min) || !is_finite(coef_max) || !(coef_min <= coef_max)) { return false; }
  if(!(coef_min > T{0}) && !(coef_max < T{0})) { return false; }
  const T residual = battery::sub_up<T>(budget, partial);  // finite or +inf: never NaN.
  if(!(residual < battery::limits<T>::inf())) { return false; }
  /** `min_{a in [coef_min, coef_max]} a * x_m` is `coef_min * x_m` or `coef_max * x_m` depending on
   * the sign of `x_m`, and the sign of `residual` tells which one bounds `x_m`. */
  if(coef_min > T{0}) {
    const T x_ub = (residual >= T{0}) ? battery::div_up<T>(residual, coef_min)
                                      : battery::div_up<T>(residual, coef_max);
    return neurons.meet(m, battery::limits<T>::neg_inf(), x_ub);
  }
  else {
    const T x_lb = (residual >= T{0}) ? battery::div_down<T>(residual, coef_max)
                                      : battery::div_down<T>(residual, coef_min);
    return neurons.meet(m, x_lb, battery::limits<T>::inf());
  }
}

/** Clip every input with the sound linear constraint  sum_k min_{a in A_k} a * x_k + const_min <= rhs,
 * where `coef(k, coef_min, coef_max)` gives the interval coefficient `A_k = [coef_min, coef_max]` of
 * the input `k`.
 *
 * As the backward step of JET (`deduce_neuron`, STEP 3), the sum of the terms is computed once, and
 * the sum of the *other* terms of an input `m` is recovered by subtracting its own term on the same
 * side: `partial = sum (-) term_m`, both lower bounds and the subtraction rounded down. This is
 * O(n0) per constraint instead of O(n0^2). It is sound because the term subtracted is recomputed from
 * the bounds of `x_m` read *after* the sum was formed: bounds only shrink, so it is at least the term
 * that was added, and `partial` stays a lower bound of the other terms (the invariant of
 * `jet_affine_v2.hpp`). When `x_m` is tightened, its new term replaces the old one in the sum, so the
 * next inputs are clipped with it (as when the sum is recomputed).
 *
 * A term that is not finite (an infinite coefficient or input bound; NaN cannot be summed) is not
 * added to the sum but counted: with two or more of them, every input has one among its other terms
 * and nothing can be clipped; with one, only its own input can be, from the sum of the finite terms.
 * \return `true` if an input changed. */
template <class T, class Access, class Coef>
CUDA INLINE bool clip_inputs(const Coef& coef, T const_min, T rhs, int n0, Access& neurons) {
  if(!is_finite(const_min) || !is_finite(rhs)) { return false; }
  const T budget = battery::sub_up<T>(rhs, const_min);  // an upper bound of  sum_k min_{a in A_k} a * x_k.
  /** STEP 1: the sum of the finite terms, rounded down, and the number of the other ones. */
  T sum = T{0};
  int num_infinite = 0;
  int infinite_k = -1;
  for(int k = 0; k < n0; ++k) {
    T coef_min, coef_max;
    coef(k, coef_min, coef_max);
    const T term = interval_product_min<T>(coef_min, coef_max, neurons.lb(k), neurons.ub(k));
    if(is_finite(term)) {
      sum = battery::add_down<T>(sum, term);
    }
    else {
      ++num_infinite;
      infinite_k = k;
    }
  }
  if(num_infinite >= 2) { return false; }
  if(num_infinite == 1) {
    T coef_min, coef_max;
    coef(infinite_k, coef_min, coef_max);
    return clip_input<T>(coef_min, coef_max, budget, sum, infinite_k, neurons);
  }
  /** STEP 2: every input, from the sum of the others. */
  bool changed = false;
  for(int m = 0; m < n0; ++m) {
    T coef_min, coef_max;
    coef(m, coef_min, coef_max);
    const T term = interval_product_min<T>(coef_min, coef_max, neurons.lb(m), neurons.ub(m));
    if(!is_finite(term)) { break; }  // the box became empty meanwhile: nothing more to clip.
    const T partial = battery::sub_down<T>(sum, term);
    if(clip_input<T>(coef_min, coef_max, budget, partial, m, neurons)) {
      changed = true;
      const T new_term = interval_product_min<T>(coef_min, coef_max, neurons.lb(m), neurons.ub(m));
      if(!is_finite(new_term)) { break; }
      sum = battery::add_down<T>(partial, new_term);
    }
  }
  return changed;
}

/** Clip the inputs with  Low(x) <= v_ub  and  -Up(x) <= -v_lb  (`-jet_sym_clip_hidden`), where `lower`
 * and `upper` are the forms of a value with bounds `[v_lb, v_ub]` that may come from elsewhere than
 * the forms (the pre-activation of a ReLU: the deduction of the store, a ReLU split).
 * A constraint is skipped when its bound is not tighter than the concretization of the forms
 * (`form_lb`, `form_ub`), since it then holds on the whole box (`max Low <= max Up <= form_ub`).
 * \return `true` if an input changed. */
template <class T, class F, class Access>
CUDA INLINE bool clip_hidden(F lower, F upper, T form_lb, T form_ub, T v_lb, T v_ub, int n0, Access& neurons) {
  bool changed = false;
  if(v_ub < form_ub) {
    changed |= clip_inputs<T>([&](int k, T& coef_min, T& coef_max) {
        coef_min = lower[coef_min_slot(k)];
        coef_max = lower[coef_max_slot(n0, k)];
      },
      lower[const_min_slot(n0)], v_ub, n0, neurons);
  }
  if(v_lb > form_lb) {
    changed |= clip_inputs<T>([&](int k, T& coef_min, T& coef_max) {
        coef_min = -upper[coef_max_slot(n0, k)];
        coef_max = -upper[coef_min_slot(k)];
      },
      -upper[const_max_slot(n0)], -v_lb, n0, neurons);
  }
  return changed;
}

/** A lower bound of  min_x Low_x(x) - Up_y(x)  over the input box, from the lower form `lower_x` of a
 * neuron and the upper form `upper_y` of another: `-inf` when it is NaN or `+inf` (no bound), as
 * `concretize_lb`. It is a lower bound of `x - y` for every point of the box. */
template <class T, class F, class Access>
CUDA INLINE T difference_lb(F lower_x, F upper_y, int n0, const Access& neurons) {
  T sum = battery::sub_down<T>(lower_x[const_min_slot(n0)], upper_y[const_max_slot(n0)]);
  for(int k = 0; k < n0; ++k) {
    const T coef_min = battery::sub_down<T>(lower_x[coef_min_slot(k)], upper_y[coef_max_slot(n0, k)]);
    const T coef_max = battery::sub_up<T>(lower_x[coef_max_slot(n0, k)], upper_y[coef_min_slot(k)]);
    sum = battery::add_down<T>(sum, interval_product_min<T>(coef_min, coef_max, neurons.lb(k), neurons.ub(k)));
  }
  return (sum < battery::limits<T>::inf()) ? sum : battery::limits<T>::neg_inf();
}

/** The four ends `lower[min], lower[max], upper[min], upper[max]` of one slot pair of a target being
 * composed (the slots `k` and `n0 + k` of an input `k < n0`, or the constant for `k == n0`). */
template <class T>
struct SlotAcc {
  T lower_min, lower_max, upper_min, upper_max;
};

/** `acc += w * src` for one slot pair, with the forms of the source `[lower_min, lower_max]` and
 * `[upper_min, upper_max]` at that slot pair: exactly the operations of `add_scaled` on that slot
 * pair (`w != 0`; the lower form of the target reads the lower form of the source when `w > 0`, the
 * upper one otherwise, and conversely). */
template <class T>
CUDA INLINE void accumulate(SlotAcc<T>& acc, T w, T src_lower_min, T src_lower_max, T src_upper_min, T src_upper_max) {
  if(w > T{0}) {
    acc.lower_min = battery::add_down<T>(acc.lower_min, battery::mul_down<T>(w, src_lower_min));
    acc.lower_max = battery::add_up<T>(acc.lower_max, battery::mul_up<T>(w, src_lower_max));
    acc.upper_min = battery::add_down<T>(acc.upper_min, battery::mul_down<T>(w, src_upper_min));
    acc.upper_max = battery::add_up<T>(acc.upper_max, battery::mul_up<T>(w, src_upper_max));
  }
  else {
    acc.lower_min = battery::add_down<T>(acc.lower_min, battery::mul_down<T>(w, src_upper_max));
    acc.lower_max = battery::add_up<T>(acc.lower_max, battery::mul_up<T>(w, src_upper_min));
    acc.upper_min = battery::add_down<T>(acc.upper_min, battery::mul_down<T>(w, src_lower_max));
    acc.upper_max = battery::add_up<T>(acc.upper_max, battery::mul_up<T>(w, src_lower_min));
  }
}

/** `acc += w * x_k` on the slot pair of the input `k` (`add_input` on both forms). */
template <class T>
CUDA INLINE void accumulate_input(SlotAcc<T>& acc, T w) {
  acc.lower_min = battery::add_down<T>(acc.lower_min, w);
  acc.lower_max = battery::add_up<T>(acc.lower_max, w);
  acc.upper_min = battery::add_down<T>(acc.upper_min, w);
  acc.upper_max = battery::add_up<T>(acc.upper_max, w);
}

/** The number of slot pairs of a target composed by one thread (registers: 4 values each). The loops
 * over them are unrolled, so that `acc[q]` stays in registers (a dynamic index would put the array in
 * local memory). */
constexpr int SLOT_TILE = 8;
/** Only where the compiler knows the pragma: the device code (`__CUDA_ARCH__`), and clang on the host
 * (not the host code of nvcc, which gcc compiles and would warn about). */
#if defined(__CUDA_ARCH__) || (defined(__clang__) && !defined(__CUDACC__))
  #define TURBO_UNROLL _Pragma("unroll")
#else
  #define TURBO_UNROLL
#endif

/** What the steps of `pass` need to find the forms of a neuron: its row (`layer_row`, or `i - n0`
 * without plan), whether its layer is a scale of the inputs that is not stored, and the slots of the
 * forms of a source. The members are force-inlined (`INLINE`), so that the accumulators of a work item
 * that `add_source` updates stay in registers. */
template <class T, class Layout, class Net>
struct FormAccess {
  const Net& net;
  const Layout& forms;
  const int* layer_row;
  int n0;

  CUDA INLINE int row_of(int l, int j) const {
    return layer_row ? layer_row[l] + j : net.layers[l].base - n0 + j;
  }
  CUDA INLINE bool is_virtual(int l) const { return layer_row != nullptr && l > 0 && layer_row[l] == ROW_VIRTUAL; }
  CUDA INLINE VirtualScaleForm<T> virtual_form(int l, int c) const {
    const auto& S = net.layers[l];
    return VirtualScaleForm<T>{static_cast<T>(net.weights[S.weight_base + c]), static_cast<T>(net.biases[S.bias_base + c]), c, n0};
  }
  /** Whether both forms of the source `c` of the layer `l` are constant (an input is not). */
  CUDA INLINE bool source_constant(int l, int c) const {
    if(l == 0) { return false; }
    if(is_virtual(l)) { return static_cast<T>(net.weights[net.layers[l].weight_base + c]) == T{0}; }
    return forms.flag(row_of(l, c)) != T{0};
  }
  /** The slot pair `k` of the forms of the source `c` of the (non-input) layer `l`. */
  CUDA INLINE void source_slots(int l, int c, int k, T& lower_min, T& lower_max, T& upper_min, T& upper_max) const {
    const int slot_min = (k < n0) ? coef_min_slot(k) : const_min_slot(n0);
    const int slot_max = (k < n0) ? coef_max_slot(n0, k) : const_max_slot(n0);
    if(is_virtual(l)) {
      const auto f = virtual_form(l, c);
      lower_min = f[slot_min]; lower_max = f[slot_max];
      upper_min = lower_min; upper_max = lower_max;
    }
    else {
      const int r = row_of(l, c);
      const auto lower = forms.lower(r);
      const auto upper = forms.upper(r);
      lower_min = lower[slot_min]; lower_max = lower[slot_max];
      upper_min = upper[slot_min]; upper_max = upper[slot_max];
    }
  }
  /** The row of a neuron of the output layer or of the property (`-1`: not stored). */
  CUDA INLINE int row_of_neuron(int i) const {
    for(int l = 1; l < net.num_layers(); ++l) {
      const auto& L = net.layers[l];
      if(i >= L.base && i < L.base + L.size) { return is_virtual(l) ? -1 : row_of(l, i - L.base); }
    }
    return -1;
  }
  /** `acc += w * src` on the slot pairs `[k_begin, k_begin + num_k)` of a target, for the source `c`
   * of the layer `ls`. */
  CUDA INLINE void add_source(SlotAcc<T> (&acc)[SLOT_TILE], int k_begin, int num_k, int ls, int c, T w) const {
    if(ls == 0) {  // the input `c`: only its own slot pair.
      TURBO_UNROLL
      for(int q = 0; q < SLOT_TILE; ++q) {
        if(q < num_k && k_begin + q == c) { accumulate_input<T>(acc[q], w); }
      }
      return;
    }
    if(is_virtual(ls)) {  // `s x_c + t`: the slot pair of the input `c`, and the constant.
      const bool zero_scale = virtual_form(ls, c).s == T{0};
      TURBO_UNROLL
      for(int q = 0; q < SLOT_TILE; ++q) {
        if(q >= num_k) { continue; }
        const int k = k_begin + q;
        if(k < n0 && (k != c || zero_scale)) { continue; }
        T lmin, lmax, umin, umax;
        source_slots(ls, c, k, lmin, lmax, umin, umax);
        accumulate<T>(acc[q], w, lmin, lmax, umin, umax);
      }
      return;
    }
    const bool constant = source_constant(ls, c);
    TURBO_UNROLL
    for(int q = 0; q < SLOT_TILE; ++q) {
      if(q >= num_k) { continue; }
      const int k = k_begin + q;
      if(constant && k < n0) { continue; }
      T lmin, lmax, umin, umax;
      source_slots(ls, c, k, lmin, lmax, umin, umax);
      accumulate<T>(acc[q], w, lmin, lmax, umin, umax);
    }
  }
};

/** One symbolic pass followed by input clipping, executed cooperatively by the threads of `group`.
 * `forms` is a layout (`NeuronMajor` or `SlotMajor`) over `storage_size_rows(rows, n0)` values,
 * `layer_row` the rows of the layers (`plan_form_rows`; `nullptr`: the row of the neuron `i` is
 * `i - n0`, and no layer is `ROW_VIRTUAL`).
 *
 * Each layer is done in two steps, separated by a barrier:
 *   A. the composition of the forms of the pre-activation (step 1 below). One work item is one target
 *      and `SLOT_TILE` slot pairs of its forms, accumulated in registers over the sources in their
 *      order, and written once: the operations on each slot are those of `add_scaled` / `add_input`
 *      in the same order, so the forms are the same values as when each neuron composed its whole
 *      forms in place (the reference of `tests/jet_symbolic_reference.hpp`), but all the threads
 *      work (targets x slot pairs, not only targets) and a target is not read and written again in
 *      memory for every source. With `SlotMajor`, consecutive threads take consecutive targets of
 *      the same slot pairs: they read the same values of a source (one transaction for the warp)
 *      and write consecutive addresses. Three kinds of sources cost less than a full form: an input
 *      (one slot), a scale of the inputs (`ROW_VIRTUAL`: one slot, and the constant), and a source
 *      whose two forms are constant (`flag`: only the constant; adding `w * 0` to a coefficient
 *      changes nothing).
 *   B. one thread per target: concretization, ReLU relaxation, meet, and the clipping with the
 *      hidden neurons (steps 2-4), then the flag of the row.
 * `hidden_clipping`: also clip the inputs with the hidden neurons (`clip_hidden`).
 * `pre` (`2 * net.num_neurons` values, or null): receives the bounds of the pre-activation `p` of every
 * neuron with an activation computed by the pass, `pre[i]` the lower one and `pre[num_neurons + i]` the
 * upper one (for the score of the ReLUs, `-relu_score gap` and `-abs_grad relax`).
 * `forward_only` (`-forward_only`): the forms only bound the neurons (steps 1-4); nothing is clipped,
 * neither the inputs nor with the hidden neurons (`hidden_clipping` is ignored), and the comparisons
 * `x - y <= k` of the property are only checked: the store fails if  min_x Low_x(x) - Up_y(x) > k
 * (`difference_lb`). The bounds of the output neurons against the store are checked by their meet.
 * \return `true` (in at least one thread) if the store changed. */
template <class T, class Layout, class Group, class Net, class Prop, class Access>
CUDA bool pass(Group& group, const Net& net, const Prop& prop, Access& neurons, const Layout& forms,
  const int* layer_row, bool hidden_clipping = false, bool forward_only = false, T* pre = nullptr)
{
  const int n0 = net.num_inputs();
  bool changed = false;
  if(forward_only) { hidden_clipping = false; }
  const int num_slot_pairs = n0 + 1;
  const int num_tiles = (num_slot_pairs + SLOT_TILE - 1) / SLOT_TILE;

  const FormAccess<T, Layout, Net> fa{net, forms, layer_row, n0};

  /** 1-4: forms, concretization, ReLU relaxation and meet, layer by layer. */
  for(int l = 1; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    const bool virtual_layer = fa.is_virtual(l);

    /** A. The forms of the pre-activation of the targets. */
    if(!virtual_layer) {
      const int num_items = L.size * num_tiles;
      for(int item = group.rank(); item < num_items; item += group.size()) {
        const int j = Layout::slot_major ? item % L.size : item / num_tiles;
        const int tile = Layout::slot_major ? item / L.size : item % num_tiles;
        const int k_begin = tile * SLOT_TILE;
        const int num_k = battery::min(SLOT_TILE, num_slot_pairs - k_begin);
        /** The constant of the target: the bias (affine), the shift (scale), or 0 (residual). */
        const T init_const = (L.kind == KIND_ADD) ? T{0} : static_cast<T>(net.biases[L.bias_base + j]);
        SlotAcc<T> acc[SLOT_TILE];
        TURBO_UNROLL
        for(int q = 0; q < SLOT_TILE; ++q) {
          const T v = (k_begin + q == n0) ? init_const : T{0};
          acc[q] = SlotAcc<T>{v, v, v, v};
        }
        if(L.kind == KIND_ADD) {
          fa.add_source(acc, k_begin, num_k, L.src0, j, T{1});
          fa.add_source(acc, k_begin, num_k, L.src1, j, T{1});
        }
        else if(L.kind == KIND_SCALE) {
          const T s = static_cast<T>(net.weights[L.weight_base + j]);
          if(s != T{0}) { fa.add_source(acc, k_begin, num_k, L.src0, j, s); }
        }
        else if(L.kind == KIND_SPARSE) {
          /** The stored weights of the row `j`, in increasing order of their sources: each slot pair
           * receives the same terms in the same order as in the dense case below. */
          int begin, end;
          sparse_row(net, L, j, begin, end);
          if(L.src0 == 0) {
            for(int e = begin; e < end; ++e) {
              const int q = net.sp_col[e] - k_begin;
              if(q < 0 || q >= num_k) { continue; }
              const T w = static_cast<T>(net.sp_val[e]);
              if(w != T{0}) { accumulate_input<T>(acc[q], w); }
            }
          }
          else if(fa.is_virtual(L.src0)) {
            const int qc = n0 - k_begin;  // the slot pair of the constant, if it is in the tile.
            for(int e = begin; e < end; ++e) {
              const int c = net.sp_col[e];
              const T w = static_cast<T>(net.sp_val[e]);
              if(w == T{0}) { continue; }
              const int q = c - k_begin;
              if(c < n0 && q >= 0 && q < num_k && fa.virtual_form(L.src0, c).s != T{0}) {
                T lmin, lmax, umin, umax;
                fa.source_slots(L.src0, c, c, lmin, lmax, umin, umax);
                accumulate<T>(acc[q], w, lmin, lmax, umin, umax);
              }
              if(qc >= 0 && qc < num_k) {
                const T t = fa.virtual_form(L.src0, c).t;
                accumulate<T>(acc[qc], w, t, t, t, t);
              }
            }
          }
          else {
            for(int e = begin; e < end; ++e) {
              const T w = static_cast<T>(net.sp_val[e]);
              if(w != T{0}) { fa.add_source(acc, k_begin, num_k, L.src0, net.sp_col[e], w); }
            }
          }
        }
        else {
          const auto& S = net.layers[L.src0];
          if(L.src0 == 0) {
            /** The sources are the inputs: the slot pair `k` only receives the weight of the input `k`. */
            TURBO_UNROLL
            for(int q = 0; q < SLOT_TILE; ++q) {
              if(q >= num_k) { continue; }
              const int k = k_begin + q;
              if(k >= n0) { continue; }
              const T w = static_cast<T>(net.weights[L.weight_base + k * L.size + j]);
              if(w != T{0}) { accumulate_input<T>(acc[q], w); }
            }
          }
          else if(fa.is_virtual(L.src0)) {
            /** The sources are a scale of the inputs: the slot pair `k` only receives the source `k`,
             * and the constant receives every source. */
            TURBO_UNROLL
            for(int q = 0; q < SLOT_TILE; ++q) {
              if(q >= num_k) { continue; }
              const int k = k_begin + q;
              if(k < n0) {
                const T w = static_cast<T>(net.weights[L.weight_base + k * L.size + j]);
                if(w != T{0} && fa.virtual_form(L.src0, k).s != T{0}) {
                  T lmin, lmax, umin, umax;
                  fa.source_slots(L.src0, k, k, lmin, lmax, umin, umax);
                  accumulate<T>(acc[q], w, lmin, lmax, umin, umax);
                }
              }
              else {
                for(int c = 0; c < S.size; ++c) {
                  const T w = static_cast<T>(net.weights[L.weight_base + c * L.size + j]);
                  if(w == T{0}) { continue; }
                  const T t = fa.virtual_form(L.src0, c).t;
                  accumulate<T>(acc[q], w, t, t, t, t);
                }
              }
            }
          }
          else {
            for(int c = 0; c < S.size; ++c) {
              /** Weights are column-major: the weight of `(c, j)` is at `weight_base + c * size + j`. */
              const T w = static_cast<T>(net.weights[L.weight_base + c * L.size + j]);
              if(w != T{0}) { fa.add_source(acc, k_begin, num_k, L.src0, c, w); }
            }
          }
        }
        const int r = fa.row_of(l, j);
        const auto lower = forms.lower(r);
        const auto upper = forms.upper(r);
        TURBO_UNROLL
        for(int q = 0; q < SLOT_TILE; ++q) {
          if(q >= num_k) { continue; }
          const int k = k_begin + q;
          const int slot_min = (k < n0) ? coef_min_slot(k) : const_min_slot(n0);
          const int slot_max = (k < n0) ? coef_max_slot(n0, k) : const_max_slot(n0);
          lower[slot_min] = acc[q].lower_min;
          lower[slot_max] = acc[q].lower_max;
          upper[slot_min] = acc[q].upper_min;
          upper[slot_max] = acc[q].upper_max;
        }
      }
      group.sync();
    }

    /** B. Concretization, ReLU relaxation, meet and clipping with the hidden neurons, one thread
     * per target, then whether both forms of the target are constant. */
    for(int j = group.rank(); j < L.size; j += group.size()) {
      const int target = L.base + j;
      if(virtual_layer) {
        /** `s x_c + t` (no ReLU: see `is_virtual_scale`), as `concretize_lb` / `concretize_ub` of the
         * stored forms, whose other terms are all 0. */
        const auto f = fa.virtual_form(l, j);
        T form_lb = battery::add_down<T>(f.t, interval_product_min<T>(f[coef_min_slot(j)], f[coef_max_slot(n0, j)], neurons.lb(j), neurons.ub(j)));
        T form_ub = battery::add_up<T>(f.t, interval_product_max<T>(f[coef_min_slot(j)], f[coef_max_slot(n0, j)], neurons.lb(j), neurons.ub(j)));
        form_lb = (form_lb < battery::limits<T>::inf()) ? form_lb : battery::limits<T>::neg_inf();
        form_ub = (form_ub > battery::limits<T>::neg_inf()) ? form_ub : battery::limits<T>::inf();
        if(hidden_clipping && l != net.output_layer) {
          changed |= clip_hidden<T>(f, f, form_lb, form_ub, neurons.lb(target), neurons.ub(target), n0, neurons);
        }
        changed |= neurons.meet(target, form_lb, form_ub);
        continue;
      }
      const int r = fa.row_of(l, j);
      const auto lower = forms.lower(r);
      const auto upper = forms.upper(r);
      /** Whether the composition above gave constant forms: every source with a weight is constant. */
      bool composed_constant = true;
      if(L.kind == KIND_ADD) {
        composed_constant = fa.source_constant(L.src0, j) && fa.source_constant(L.src1, j);
      }
      else if(L.kind == KIND_SCALE) {
        composed_constant = static_cast<T>(net.weights[L.weight_base + j]) == T{0} || fa.source_constant(L.src0, j);
      }
      else if(L.kind == KIND_SPARSE) {
        int begin, end;
        sparse_row(net, L, j, begin, end);
        for(int e = begin; e < end && composed_constant; ++e) {
          if(static_cast<T>(net.sp_val[e]) != T{0} && !fa.source_constant(L.src0, net.sp_col[e])) {
            composed_constant = false;
          }
        }
      }
      else {
        for(int c = 0; c < net.layers[L.src0].size && composed_constant; ++c) {
          if(static_cast<T>(net.weights[L.weight_base + c * L.size + j]) != T{0} && !fa.source_constant(L.src0, c)) {
            composed_constant = false;
          }
        }
      }
      bool lower_constant = composed_constant;
      bool upper_constant = composed_constant;
      const T form_lb = concretize_lb<T>(lower, n0, neurons);
      const T form_ub = concretize_ub<T>(upper, n0, neurons);
      if(L.sigmoid != 0) {
        /** Bounds of the pre-activation `p`: its forms, and the inverse of the bounds of `y = sigmoid(p)`. */
        const T p_lb = battery::max(form_lb, sigmoid::logit_lb<T>(neurons.lb(target)));
        const T p_ub = battery::min(form_ub, sigmoid::logit_ub<T>(neurons.ub(target)));
        if(pre) { pre[target] = p_lb; pre[net.num_neurons + target] = p_ub; }
        if(!(p_lb <= p_ub)) {
          if(p_lb > p_ub) { changed |= neurons.fail(target); }
          set_const(lower, n0, T{0}, T{0});
          set_const(upper, n0, T{1}, T{1});
          forms.flag(r) = T{1};
          continue;
        }
        if(hidden_clipping) {
          changed |= clip_hidden<T>(lower, upper, form_lb, form_ub, p_lb, p_ub, n0, neurons);
        }
        /** The forms of `p` become those of `y` by the two lines of `sigmoid::relax`. */
        T aL, bL, aU, bU;
        sigmoid::relax<T>(p_lb, p_ub, aL, bL, aU, bU);
        if(aL > T{0}) { scale_shift<T>(lower, n0, aL, bL); }
        else { set_const(lower, n0, bL, bL); lower_constant = true; }
        if(aU > T{0}) { scale_shift<T>(upper, n0, aU, bU); }
        else { set_const(upper, n0, bU, bU); upper_constant = true; }
        changed |= neurons.meet(target, sigmoid::lb<T>(p_lb), sigmoid::ub<T>(p_ub));
        forms.flag(r) = (lower_constant && upper_constant) ? T{1} : T{0};
        continue;
      }
      if(L.has_relu == 0) {
        if(hidden_clipping && l != net.output_layer) {  // the output neurons are clipped after the loop.
          changed |= clip_hidden<T>(lower, upper, form_lb, form_ub, neurons.lb(target), neurons.ub(target),
            n0, neurons);
        }
        changed |= neurons.meet(target, form_lb, form_ub);
        forms.flag(r) = (lower_constant && upper_constant) ? T{1} : T{0};
        continue;
      }
      /** Bounds of the pre-activation `p`: `p <= y` always, and `p = y` when `y > 0`. */
      T p_lb = form_lb;
      T p_ub = battery::min(form_ub, neurons.ub(target));
      if(neurons.lb(target) > T{0}) { p_lb = battery::max(p_lb, neurons.lb(target)); }
      if(pre) { pre[target] = p_lb; pre[net.num_neurons + target] = p_ub; }
      if(p_lb > p_ub) {
        changed |= neurons.fail(target);
        set_const(lower, n0, T{0}, T{0});
        set_const(upper, n0, T{0}, T{0});
        forms.flag(r) = T{1};
        continue;
      }
      /** The forms of `p`, before they become those of `y`. */
      if(hidden_clipping) {
        changed |= clip_hidden<T>(lower, upper, form_lb, form_ub, p_lb, p_ub, n0, neurons);
      }
      if(p_ub <= T{0}) {          // inactive: y = 0.
        set_const(lower, n0, T{0}, T{0});
        set_const(upper, n0, T{0}, T{0});
        lower_constant = upper_constant = true;
      }
      else if(p_lb >= T{0}) {     // active: y = p, the forms of p are those of y.
      }
      else {                      // unstable (a NaN bound would land here too, never in "active").
        if(is_finite(p_lb) && is_finite(p_ub)) {
          const T slope = battery::div_up<T>(p_ub, battery::sub_down<T>(p_ub, p_lb));  // >= p_ub / (p_ub - p_lb), > 0.
          chord(upper, n0, slope, p_lb);
        }
        else {                    // no chord through an infinite bound: y <= p_ub.
          set_const(upper, n0, T{0}, p_ub);
          upper_constant = true;
        }
        /** `y >= p` holds always; DeepPoly keeps it when `p_ub > -p_lb` (false on an infinite `p_lb`). */
        if(!(p_ub > -p_lb)) {
          set_const(lower, n0, T{0}, T{0});
          lower_constant = true;
        }
      }
      changed |= neurons.meet(target, battery::max(p_lb, T{0}), battery::max(p_ub, T{0}));
      forms.flag(r) = (lower_constant && upper_constant) ? T{1} : T{0};
    }
    group.sync();
  }

  /** `forward_only`: the comparisons of the property are checked, one per thread. */
  if(forward_only) {
    for(int q = group.rank(); q < prop.size(); q += group.size()) {
      const auto& leq = prop.leqs[q];
      if(leq.x < n0 || leq.y < n0) { continue; }  // comparisons on inputs are handled by the interval deduction.
      const int rx = fa.row_of_neuron(leq.x), ry = fa.row_of_neuron(leq.y);
      if(rx < 0 || ry < 0) { continue; }
      if(difference_lb<T>(forms.lower(rx), forms.upper(ry), n0, neurons) > static_cast<T>(leq.k)) {
        changed |= neurons.fail(leq.x);
      }
    }
    group.sync();
    return changed;
  }

  /** Input clipping with the bounded output neurons and the comparisons of the property, one
   * constraint per thread (`clip_inputs` walks the inputs). */
  const auto& out_layer = net.layers[net.output_layer];
  const int num_constraints = 2 * out_layer.size + prop.size();
  for(int q = group.rank(); q < num_constraints; q += group.size()) {
    if(q < out_layer.size) {                // Low_out(x) <= ub(out)
      const int out = out_layer.base + q;
      const auto lower = forms.lower(fa.row_of(net.output_layer, q));
      changed |= clip_inputs<T>([&](int k, T& coef_min, T& coef_max) {
          coef_min = lower[coef_min_slot(k)];
          coef_max = lower[coef_max_slot(n0, k)];
        },
        lower[const_min_slot(n0)], neurons.ub(out), n0, neurons);
    }
    else if(q < 2 * out_layer.size) {       // -Up_out(x) <= -lb(out)
      const int out = out_layer.base + q - out_layer.size;
      const auto upper = forms.upper(fa.row_of(net.output_layer, q - out_layer.size));
      changed |= clip_inputs<T>([&](int k, T& coef_min, T& coef_max) {
          coef_min = -upper[coef_max_slot(n0, k)];
          coef_max = -upper[coef_min_slot(k)];
        },
        -upper[const_max_slot(n0)], -neurons.lb(out), n0, neurons);
    }
    else {                                  // Low_x(x) - Up_y(x) <= k
      const auto& leq = prop.leqs[q - 2 * out_layer.size];
      if(leq.x < n0 || leq.y < n0) { continue; }  // comparisons on inputs are handled by the interval deduction.
      const int rx = fa.row_of_neuron(leq.x), ry = fa.row_of_neuron(leq.y);
      if(rx < 0 || ry < 0) { continue; }
      const auto lower_x = forms.lower(rx);
      const auto upper_y = forms.upper(ry);
      changed |= clip_inputs<T>([&](int k, T& coef_min, T& coef_max) {
          coef_min = battery::sub_down<T>(lower_x[coef_min_slot(k)], upper_y[coef_max_slot(n0, k)]);
          coef_max = battery::sub_up<T>(lower_x[coef_max_slot(n0, k)], upper_y[coef_min_slot(k)]);
        },
        battery::sub_down<T>(lower_x[const_min_slot(n0)], upper_y[const_max_slot(n0)]), static_cast<T>(leq.k),
        n0, neurons);
    }
  }
  group.sync();
  return changed;
}

/** `pass` over the raw storage `data` of `storage_size_rows(rows, n0)` values, with the rows of
 * `layer_row` (`plan_form_rows`, `rows` rows), or without plan (`layer_row == nullptr`:
 * `storage_size(net.num_neurons, n0)` values, one row per non-input neuron), in the layout `layout`
 * (`LAYOUT_NEURON` or `LAYOUT_SLOT`). The layout is the same for all the threads, so the branch does
 * not diverge. */
template <class T, class Group, class Net, class Prop, class Access>
CUDA bool pass(Group& group, const Net& net, const Prop& prop, Access& neurons, T* data, int layout,
  bool hidden_clipping = false, bool forward_only = false, const int* layer_row = nullptr, int rows = 0,
  T* pre = nullptr)
{
  const int n0 = net.num_inputs();
  const size_t num_rows = layer_row ? static_cast<size_t>(rows) : static_cast<size_t>(net.num_neurons - n0);
  if(layout == LAYOUT_NEURON) {
    return pass<T>(group, net, prop, neurons, NeuronMajor<T>{data, n0, num_rows}, layer_row, hidden_clipping, forward_only, pre);
  }
  return pass<T>(group, net, prop, neurons, SlotMajor<T>{data, n0, num_rows}, layer_row, hidden_clipping, forward_only, pre);
}

} // namespace symbolic
} // namespace jet

#endif
