// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_SYMBOLIC_HPP
#define TURBO_JET_SYMBOLIC_HPP

#include "battery/utility.hpp"

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
 * One pass (`pass`) does, layer by layer in topological order:
 *   1. compose the forms of the pre-activation from the forms of the source neurons (an input
 *      neuron `k` has the exact form `x_k`, which is not stored);
 *   2. concretize them over the input box to get bounds of the pre-activation;
 *   3. relax the ReLU as DeepPoly: the chord as upper bound (its slope rounded up), and either
 *      `y >= p` or `y >= 0` as lower bound (`y >= p` when `p_ub > -p_lb`);
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
 * Memory: the forms of all non-input neurons, `(num_neurons - n0) * (4 * n0 + 4)` values.
 * A form has `form_size(n0) = 2 * n0 + 2` *slots*: `coef_min_k` at `coef_min_slot(k) = k`,
 * `coef_max_k` at `coef_max_slot(n0, k) = n0 + k`, `const_min` at `const_min_slot(n0) = 2 n0` and
 * `const_max` at `const_max_slot(n0) = 2 n0 + 1`. Only `const_min` of a lower form and `const_max` of
 * an upper form are ever read for a bound; the other end of the constant is kept so that the
 * coefficients and the constant are computed by the same code.
 * Two layouts store the same values, chosen with `-jet_sym_layout` (`layout` of `pass`):
 *   * `NeuronMajor` (`LAYOUT_NEURON`): the slots of a neuron are contiguous, the lower form then
 *     the upper form: slot `s` of the lower form of `i` is at `(i - n0) * (4 n0 + 4) + s`.
 *   * `SlotMajor` (`LAYOUT_SLOT`, default): a slot of all the neurons is contiguous: slot `s` of
 *     the lower form of `i` is at `s * m + (i - n0)`, of the upper form at
 *     `(form_size(n0) + s) * m + (i - n0)`, with `m = num_neurons - n0`.
 *     The threads of a warp handle consecutive neurons of a layer and walk the slots in the same
 *     order, so with this layout they read and write consecutive addresses (coalesced accesses).
 * The composition of the forms does not depend on the layout, so both give bitwise the same forms
 * and, on one thread, the same store. With several threads, the clipping of the inputs reads bounds
 * that other threads may be tightening at the same time; bounds only shrink, so every value read is
 * sound, but how much is clipped in one pass can depend on the timing.
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

/** Values of `-jet_sym_layout`. */
constexpr int LAYOUT_NEURON = 0;
constexpr int LAYOUT_SLOT = 1;

CUDA INLINE constexpr int form_size(int n0) { return 2 * n0 + 2; }
CUDA INLINE constexpr int neuron_stride(int n0) { return 2 * form_size(n0); }
CUDA INLINE constexpr size_t storage_size(int num_neurons, int n0) {
  return static_cast<size_t>(num_neurons - n0) * static_cast<size_t>(neuron_stride(n0));
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

template <class T>
struct NeuronMajor {
  using form = ContiguousForm<T>;
  T* data;
  int n0;
  CUDA INLINE form lower(int i) const {
    return form{data + static_cast<size_t>(i - n0) * static_cast<size_t>(neuron_stride(n0))};
  }
  CUDA INLINE form upper(int i) const {
    return form{data + static_cast<size_t>(i - n0) * static_cast<size_t>(neuron_stride(n0)) + form_size(n0)};
  }
};

template <class T>
struct SlotMajor {
  using form = StridedForm<T>;
  T* data;
  int n0;
  size_t stride;  /**< the number of non-input neurons, which is the distance between two slots of a form. */
  CUDA INLINE form lower(int i) const {
    return form{data + static_cast<size_t>(i - n0), stride};
  }
  CUDA INLINE form upper(int i) const {
    return form{data + static_cast<size_t>(form_size(n0)) * stride + static_cast<size_t>(i - n0), stride};
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

/** One symbolic pass followed by input clipping, executed cooperatively by the threads of `group`.
 * `forms` is a layout (`NeuronMajor` or `SlotMajor`) over `storage_size(...)` values.
 * `hidden_clipping`: also clip the inputs with the hidden neurons (`clip_hidden`).
 * `forward_only` (`-forward_only`): the forms only bound the neurons (steps 1-4); nothing is clipped,
 * neither the inputs nor with the hidden neurons (`hidden_clipping` is ignored), and the comparisons
 * `x - y <= k` of the property are only checked: the store fails if  min_x Low_x(x) - Up_y(x) > k
 * (`difference_lb`). The bounds of the output neurons against the store are checked by their meet.
 * \return `true` (in at least one thread) if the store changed. */
template <class T, class Layout, class Group, class Net, class Prop, class Access>
CUDA bool pass(Group& group, const Net& net, const Prop& prop, Access& neurons, const Layout& forms,
  bool hidden_clipping = false, bool forward_only = false)
{
  const int n0 = net.num_inputs();
  bool changed = false;
  if(forward_only) { hidden_clipping = false; }

  /** 1-4: forms, concretization, ReLU relaxation and meet, layer by layer. */
  for(int l = 1; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    for(int j = group.rank(); j < L.size; j += group.size()) {
      const int target = L.base + j;
      const auto lower = forms.lower(target);
      const auto upper = forms.upper(target);
      if(L.kind == KIND_ADD) {
        set_const(lower, n0, T{0}, T{0});
        set_const(upper, n0, T{0}, T{0});
        const int operands[2] = {net.layers[L.src0].base + j, net.layers[L.src1].base + j};
        for(int src : operands) {
          if(src < n0) {
            add_input(lower, n0, src, T{1});
            add_input(upper, n0, src, T{1});
          }
          else {
            add_scaled(lower, forms.lower(src), n0, T{1});
            add_scaled(upper, forms.upper(src), n0, T{1});
          }
        }
      }
      else {
        const T bias = static_cast<T>(net.biases[L.bias_base + j]);
        set_const(lower, n0, bias, bias);
        set_const(upper, n0, bias, bias);
        const auto& S = net.layers[L.src0];
        for(int c = 0; c < S.size; ++c) {
          /** Weights are column-major: the weight of `(c, j)` is at `weight_base + c * size + j`. */
          const T w = static_cast<T>(net.weights[L.weight_base + c * L.size + j]);
          if(w == T{0}) { continue; }
          const int src = S.base + c;
          if(src < n0) {
            add_input(lower, n0, src, w);
            add_input(upper, n0, src, w);
          }
          else {
            add_scaled(lower, w > T{0} ? forms.lower(src) : forms.upper(src), n0, w);
            add_scaled(upper, w > T{0} ? forms.upper(src) : forms.lower(src), n0, w);
          }
        }
      }
      const T form_lb = concretize_lb<T>(lower, n0, neurons);
      const T form_ub = concretize_ub<T>(upper, n0, neurons);
      if(L.has_relu == 0) {
        if(hidden_clipping && l != net.output_layer) {  // the output neurons are clipped after the loop.
          changed |= clip_hidden<T>(lower, upper, form_lb, form_ub, neurons.lb(target), neurons.ub(target),
            n0, neurons);
        }
        changed |= neurons.meet(target, form_lb, form_ub);
        continue;
      }
      /** Bounds of the pre-activation `p`: `p <= y` always, and `p = y` when `y > 0`. */
      T p_lb = form_lb;
      T p_ub = battery::min(form_ub, neurons.ub(target));
      if(neurons.lb(target) > T{0}) { p_lb = battery::max(p_lb, neurons.lb(target)); }
      if(p_lb > p_ub) {
        changed |= neurons.fail(target);
        set_const(lower, n0, T{0}, T{0});
        set_const(upper, n0, T{0}, T{0});
        continue;
      }
      /** The forms of `p`, before they become those of `y`. */
      if(hidden_clipping) {
        changed |= clip_hidden<T>(lower, upper, form_lb, form_ub, p_lb, p_ub, n0, neurons);
      }
      if(p_ub <= T{0}) {          // inactive: y = 0.
        set_const(lower, n0, T{0}, T{0});
        set_const(upper, n0, T{0}, T{0});
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
        }
        /** `y >= p` holds always; DeepPoly keeps it when `p_ub > -p_lb` (false on an infinite `p_lb`). */
        if(!(p_ub > -p_lb)) { set_const(lower, n0, T{0}, T{0}); }
      }
      changed |= neurons.meet(target, battery::max(p_lb, T{0}), battery::max(p_ub, T{0}));
    }
    group.sync();
  }

  /** `forward_only`: the comparisons of the property are checked, one per thread. */
  if(forward_only) {
    for(int q = group.rank(); q < prop.size(); q += group.size()) {
      const auto& leq = prop.leqs[q];
      if(leq.x < n0 || leq.y < n0) { continue; }  // comparisons on inputs are handled by the interval deduction.
      if(difference_lb<T>(forms.lower(leq.x), forms.upper(leq.y), n0, neurons) > static_cast<T>(leq.k)) {
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
      const auto lower = forms.lower(out);
      changed |= clip_inputs<T>([&](int k, T& coef_min, T& coef_max) {
          coef_min = lower[coef_min_slot(k)];
          coef_max = lower[coef_max_slot(n0, k)];
        },
        lower[const_min_slot(n0)], neurons.ub(out), n0, neurons);
    }
    else if(q < 2 * out_layer.size) {       // -Up_out(x) <= -lb(out)
      const int out = out_layer.base + q - out_layer.size;
      const auto upper = forms.upper(out);
      changed |= clip_inputs<T>([&](int k, T& coef_min, T& coef_max) {
          coef_min = -upper[coef_max_slot(n0, k)];
          coef_max = -upper[coef_min_slot(k)];
        },
        -upper[const_max_slot(n0)], -neurons.lb(out), n0, neurons);
    }
    else {                                  // Low_x(x) - Up_y(x) <= k
      const auto& leq = prop.leqs[q - 2 * out_layer.size];
      if(leq.x < n0 || leq.y < n0) { continue; }  // comparisons on inputs are handled by the interval deduction.
      const auto lower_x = forms.lower(leq.x);
      const auto upper_y = forms.upper(leq.y);
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

/** `pass` over the raw storage `data` (of `storage_size(net.num_neurons, net.num_inputs())`
 * values) with the layout `layout` (`LAYOUT_NEURON` or `LAYOUT_SLOT`). The layout is the same
 * for all the threads, so the branch does not diverge. */
template <class T, class Group, class Net, class Prop, class Access>
CUDA bool pass(Group& group, const Net& net, const Prop& prop, Access& neurons, T* data, int layout,
  bool hidden_clipping = false, bool forward_only = false)
{
  const int n0 = net.num_inputs();
  if(layout == LAYOUT_NEURON) {
    return pass<T>(group, net, prop, neurons, NeuronMajor<T>{data, n0}, hidden_clipping, forward_only);
  }
  return pass<T>(group, net, prop, neurons, SlotMajor<T>{data, n0, static_cast<size_t>(net.num_neurons - n0)},
    hidden_clipping, forward_only);
}

} // namespace symbolic
} // namespace jet

#endif
