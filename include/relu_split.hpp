// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_RELU_SPLIT_HPP
#define TURBO_RELU_SPLIT_HPP

#include <cfloat>
#include <cstdio>
#include <type_traits>
#include "battery/utility.hpp"
#include "lala/split_strategy.hpp"
#include "config.hpp"
#include "uass.hpp"

/** ReLU split of `-arch jet` and `-arch fbarebones`, selected with `-split_on relu`.
 *
 * Instead of the input neurons, the search branches on the phase of a ReLU `y = max(p, 0)`:
 *   * fbarebones: the store has a variable for the pre-activation `p` (the operand of the bytecode
 *     `y = max(p, 0)`), which is split at 0: `p in [lb, 0]` (inactive), then `p in [0, ub]` (active).
 *   * JET: the store only has `y`, which is split at `tau = jet_split_point() = FLT_MIN > 0`:
 *     `y in [lb, tau]` (so `p <= tau`: inactive up to `tau`), then `y in [tau, ub]` (so `lb(y) > 0`, and
 *     the deduction of the ReLU, `tell::fmax`, and the symbolic pass take `p = y`: active).
 *     The split point must be positive, since `y >= 0` holds in both phases: `y <= 0` and `y >= tau`
 *     would leave out every real `p` in `(0, tau)`. The two branches cover every `p` (`p <= tau` or
 *     `p >= tau`) and `tau` is the smallest normal float, so that it is never flushed to zero.
 * In both cases, the variable `v` split at the point `s` is a candidate when `lb(v) < s < ub(v)`; a
 * split leaves it out of the candidates in both branches, so a path of the search tree has at most
 * one decision per ReLU and the search terminates, even if a bound is infinite.
 *
 * The ReLU to split (`-relu_order`):
 *   * `earliest`: the first candidate in the list of the ReLUs, which follows the network (first
 *     layer first; for fbarebones, the order of the bytecodes);
 *   * `widest`: the candidate with the largest `ub(v) - lb(v)`, the earliest one on a tie.
 *
 * The leaves. When no candidate is left and the node did not fail, every ReLU has a phase: the
 * network is affine on the node, but the propagation cannot decide whether a point of the input box
 * satisfies the property. The search then continues as chosen by `-relu_leaf`:
 *   * `input` (the default): the input split of `-split_on input` (`BlockData::split`, called by the
 *     `branch` of the kernels when `split_relu` pushes nothing), with the variable and value orders of
 *     the strategy: the inputs are split until their width is at most `epsilon`, then a solution is
 *     looked for with the underapproximation of the value order (`uass.hpp`; `split`: the middle of
 *     the box). The boxes of the input split cover the leaf, so a failure there is a proof; the ReLUs
 *     keep their phase below the leaf (a split only narrows the domains, so no candidate comes back),
 *     and the whole subtree is an input split.
 *   * `point`: the inputs are fixed to points at once (`fix_inputs`), with the underapproximation
 *     given by the value order of the strategy (`-value_order`):
 *       - `split`: all the inputs at their middle at once, in a decision with a single child, as `SPLIT`
 *         on a box of width at most `epsilon`;
 *       - `lb_split`, `ub_split`, `mid_split`: one input at a time, at its lower bound, its upper bound
 *         or its middle, in a decision with a single child; the propagation runs in between, so that
 *         the next input is fixed in its domain narrowed by the previous ones (an input that the
 *         propagation reduces to a point is not fixed again);
 *       - `mid_lb_split`, `mid_ub_split`: one input at a time, at its middle, then at its lower (upper)
 *         bound: a decision with two children, so up to `2^n` points for `n` inputs (a single child
 *         when the middle is the bound);
 *       - any other value order (it is not an underapproximation): as `split` (the host warns).
 *     With one input at a time, the input is chosen by the variable order of the strategy
 *     (`-var_order`), as `lattice_smallest_split`: `anti_first_fail` the widest, `first_fail` the
 *     narrowest, `largest` the one with the largest upper bound, `smallest` the one with the smallest
 *     lower bound, `input_order` (and `random`) the first; the earliest one on a tie.
 *     Every decision of a leaf underapproximates: the node is a solution when all the inputs are
 *     points and the propagation does not fail (`propagate`), and is counted in `unknowns` when it
 *     fails, so that the answer is `unknown` rather than `unsat`. An input without finite bounds
 *     cannot be fixed: the leaf is then counted in `unknowns` directly (a single child `bot`).
 */
namespace relu_split {

template <class T>
CUDA INLINE constexpr T jet_split_point() { return static_cast<T>(FLT_MIN); }

/** Whether the phase of a variable of domain `[lb, ub]` is unknown with respect to `point`. */
template <class T>
CUDA INLINE bool is_candidate(T lb, T ub, T point) { return lb < point && point < ub; }

/** The score of a candidate for `order` (higher is preferred). */
template <class T>
CUDA INLINE T score(int order, T lb, T ub) { return order == RELU_ORDER_WIDEST ? ub - lb : T{0}; }

/** Whether the candidate (`score_a`, `pos_a`) is preferred to (`score_b`, `pos_b`): a higher score,
 * then a smaller position in the list. "No candidate" is `(-inf, n)`. */
template <class T>
CUDA INLINE bool better(T score_a, int pos_a, T score_b, int pos_b) {
  return score_a > score_b || (score_a == score_b && pos_a < pos_b);
}

/** Whether a leaf fixes one input at a time (`lb_split`, `ub_split`, `mid_split`, `mid_lb_split`,
 * `mid_ub_split`) rather than all of them at their middle at once. */
CUDA INLINE bool fixes_one_input(lala::ValueOrder order) {
  return order == lala::ValueOrder::LB_SPLIT || order == lala::ValueOrder::UB_SPLIT
    || order == lala::ValueOrder::MID_SPLIT || order == lala::ValueOrder::MID_LB_SPLIT
    || order == lala::ValueOrder::MID_UB_SPLIT;
}

/** Whether `order` is one of the underapproximations of the leaves (the others behave as `split`). */
inline bool is_leaf_value_order(lala::ValueOrder order) {
  return order == lala::ValueOrder::SPLIT || fixes_one_input(order);
}

/** Whether an input of domain `[lb, ub]` can be fixed: finite bounds, not a point. */
template <class T>
CUDA INLINE bool is_fixable(T lb, T ub) {
  return battery::limits<T>::neg_inf() < lb && ub < battery::limits<T>::inf() && lb < ub;
}

/** The score of an input of domain `[lb, ub]` for the variable order (higher is preferred, then the
 * earliest input), as the function `f` of `lattice_smallest_split`. */
template <class T>
CUDA INLINE T input_score(lala::VariableOrder order, T lb, T ub) {
  switch(order) {
    case lala::VariableOrder::FIRST_FAIL: return lb - ub;
    case lala::VariableOrder::ANTI_FIRST_FAIL:
    case lala::VariableOrder::GRA_ANTI_FIRST_FAIL: return ub - lb;
    case lala::VariableOrder::LARGEST: return ub;
    case lala::VariableOrder::SMALLEST: return -lb;
    default: return T{0};  // INPUT_ORDER, RANDOM: the first input.
  }
}

/** The points of an input of domain `[lb, ub]` (`is_fixable`) tried by a leaf with one input at a
 * time, in this order: returns their number, 1 (`first`) or 2 (`first`, then `second`). */
template <class T>
CUDA INLINE int leaf_points(lala::ValueOrder order, T lb, T ub, T& first, T& second) {
  const T mid = battery::midpoint(lb, ub);
  switch(order) {
    case lala::ValueOrder::LB_SPLIT: first = lb; return 1;
    case lala::ValueOrder::UB_SPLIT: first = ub; return 1;
    case lala::ValueOrder::MID_LB_SPLIT: first = mid; second = lb; return mid == lb ? 1 : 2;
    case lala::ValueOrder::MID_UB_SPLIT: first = mid; second = ub; return mid == ub ? 1 : 2;
    default: first = mid; return 1;  // MID_SPLIT
  }
}

/** The position of the best candidate among `n`, `n` if there is none: `candidate(i, s)` tells
 * whether `i` is a candidate and gives its score `s`. Sequential version of `block_select` (host). */
template <class T, class F>
int select_best(int n, const F& candidate) {
  T best_score = battery::limits<T>::neg_inf();
  int best_pos = n;
  for(int i = 0; i < n; ++i) {
    T s;
    if(candidate(i, s) && better(s, i, best_score, best_pos)) {
      best_score = s;
      best_pos = i;
    }
  }
  return best_pos;
}

/** The position of the ReLU to split among `n`, `n` if there is none; `dom(i, lb, ub)` gives the
 * domain of the variable `i`. Sequential version of `branch` (host tests). */
template <class T, class Dom>
int select(int n, int order, T point, const Dom& dom) {
  return select_best<T>(n, [&](int i, T& s) {
    T lb, ub;
    dom(i, lb, ub);
    s = score(order, lb, ub);
    return is_candidate(lb, ub, point);
  });
}

/** The position of the input fixed by a leaf with one input at a time among `n`, `n` if none can be
 * fixed; `dom(i, lb, ub)` gives the domain of the input `i`. Sequential version of `branch` (host). */
template <class T, class Dom>
int select_input(int n, lala::VariableOrder order, const Dom& dom) {
  return select_best<T>(n, [&](int i, T& s) {
    T lb, ub;
    dom(i, lb, ub);
    s = input_score(order, lb, ub);
    return is_fixable(lb, ub);
  });
}

/** The decision `var in [lb, point]`, then `var in [point, ub]`. Thread 0 only. */
template <class Block, class T>
CUDA INLINE void push_relu_decision(Block& b, lala::AVar var, T lb, T ub, T point) {
  using Itv = std::remove_reference_t<decltype(b.decisions[0].children[0])>;
  auto& d = b.decisions[b.depth];
  d.var = var;
  d.current_idx = -1;
  d.children[0] = Itv(lb, point);
  d.children[1] = Itv(point, ub);
  uass::push_ropes(b, false);
}

/** The decision of a leaf with one input at a time: the input `var`, of domain `[lb, ub]`
 * (`is_fixable`), at the points of `order` (`leaf_points`). Thread 0 only. */
template <class Block, class T>
CUDA INLINE void push_leaf_decision(Block& b, lala::ValueOrder order, lala::AVar var, T lb, T ub) {
  using Itv = std::remove_reference_t<decltype(b.decisions[0].children[0])>;
  T first, second;
  if(leaf_points(order, lb, ub, first, second) == 1) {
    uass::push_point_decision(b, var, Itv(first, first));
  }
  else {
    uass::push_points_decision(b, var, Itv(first, first), Itv(second, second));
  }
}

#ifdef __CUDACC__

/** `select_best` by all the threads of the block: the best candidate of each thread, of each warp
 * (shuffles), then of the block (thread 0). Every thread gets the result. */
template <class T, class F>
__device__ int block_select(int n, const F& candidate) {
  constexpr int NUM_WARPS = CUDA_THREADS_PER_BLOCK / 32;
  __shared__ T warp_score[NUM_WARPS];
  __shared__ int warp_pos[NUM_WARPS];
  __shared__ int chosen;
  T best_score = battery::limits<T>::neg_inf();
  int best_pos = n;
  for(int i = threadIdx.x; i < n; i += blockDim.x) {
    T s;
    if(candidate(i, s) && better(s, i, best_score, best_pos)) {
      best_score = s;
      best_pos = i;
    }
  }
  for(int offset = 16; offset > 0; offset /= 2) {
    const T other_score = __shfl_down_sync(0xffffffff, best_score, offset);
    const int other_pos = __shfl_down_sync(0xffffffff, best_pos, offset);
    if(better(other_score, other_pos, best_score, best_pos)) {
      best_score = other_score;
      best_pos = other_pos;
    }
  }
  if(threadIdx.x % 32 == 0) {
    warp_score[threadIdx.x / 32] = best_score;
    warp_pos[threadIdx.x / 32] = best_pos;
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    for(int w = 1; w < NUM_WARPS; ++w) {
      if(better(warp_score[w], warp_pos[w], best_score, best_pos)) {
        best_score = warp_score[w];
        best_pos = warp_pos[w];
      }
    }
    chosen = best_pos;
  }
  __syncthreads();
  const int result = chosen;
  __syncthreads();  // `chosen` is read by every thread before a next call can write it.
  return result;
}

/** Push the decision of the ReLU split on the block `b` if a ReLU has an unknown phase (see the top of
 * this file), executed by all the threads of the block: `vars[0..n)` are the variables split (the
 * neurons with a ReLU for JET, the pre-activations for fbarebones), `point` is where they are split,
 * `order` is `-relu_order`. Returns whether a decision was pushed (the same for every thread); when
 * it returns false, the node is a leaf, and the kernel branches with `-relu_leaf`. */
template <class Block, class T>
__device__ bool split_relu(Block& b, const int* vars, int n, T point, int order) {
  auto& store = *b.store;
  const int chosen = block_select<T>(n, [&](int i, T& s) {
    const T lb = store[vars[i]].lb().value();
    const T ub = store[vars[i]].ub().value();
    s = score(order, lb, ub);
    return is_candidate(lb, ub, point);
  });
  if(chosen < n && threadIdx.x == 0) {
    const int v = vars[chosen];
    push_relu_decision(b, lala::AVar{store.aty(), v}, T(store[v].lb().value()), T(store[v].ub().value()), point);
  }
  return chosen < n;
}

/** `-relu_leaf point`: fix the inputs of a leaf to points (see the top of this file), executed by all
 * the threads of the block. `strategy`: the current search strategy, whose variables (the input
 * neurons) are fixed, with its variable and value orders.
 * \param has_changed: A Boolean in shared memory. */
template <class Block, class T, class Strategy>
__device__ void fix_inputs(Block& b, bool& has_changed, const Strategy& strategy) {
  using Itv = std::remove_reference_t<decltype(b.decisions[0].children[0])>;
  auto& store = *b.store;
  const auto& inputs = strategy.vars;
  const int num_inputs = static_cast<int>(inputs.size());

  /** I. One input at a time. */
  if(fixes_one_input(strategy.val_order)) {
    const int pos = block_select<T>(num_inputs, [&](int i, T& s) {
      const T lb = store[inputs[i].vid()].lb().value();
      const T ub = store[inputs[i].vid()].ub().value();
      s = input_score(strategy.var_order, lb, ub);
      return is_fixable(lb, ub);
    });
    if(threadIdx.x == 0) {
      if(pos < num_inputs) {
        const auto& dom = store[inputs[pos].vid()];
        push_leaf_decision(b, strategy.val_order, inputs[pos], T(dom.lb().value()), T(dom.ub().value()));
      }
      else {  // The node is not a solution, so an input is not a point: it has no finite bounds.
        uass::push_point_decision(b, lala::AVar{store.aty(), 0}, Itv::bot());
      }
    }
    return;
  }

  /** II. All the inputs at their middle at once. */
  if(threadIdx.x == 0) {
    has_changed = false;
  }
  __syncthreads();
  for(int i = threadIdx.x; i < num_inputs; i += blockDim.x) {
    const auto& dom = store[inputs[i].vid()];
    if(dom.lb().is_top() || dom.ub().is_top()) {
      has_changed = true;  // no middle point.
    }
    else if(dom.lb().value() != dom.ub().value()) {
      const T mid = battery::midpoint(dom.lb().value(), dom.ub().value());
      store.embed(inputs[i], Itv(mid, mid));
    }
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    if(num_inputs == 0 || has_changed) {
      uass::push_point_decision(b, lala::AVar{store.aty(), 0}, Itv::bot());
    }
    else {
      const auto& dom = store[inputs[0].vid()];
      uass::push_point_decision(b, inputs[0], Itv(dom.lb().value(), dom.ub().value()));
    }
  }
}

#endif // __CUDACC__

} // namespace relu_split

#endif
