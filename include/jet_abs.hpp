// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_ABS_HPP
#define TURBO_JET_ABS_HPP

#include <cmath>
#include "battery/utility.hpp"
#include "config.hpp"

/** The choice of the input to bisect of `-jet_var_select` (`-arch jet`, input split).
 *
 * The candidates are the inputs that the input split may bisect: finite, wider than `epsilon`, and whose
 * middle lies strictly inside the interval (`uass::bisectable`; "the variables whose domains cannot be
 * bisected must also be excluded", Granvilliers, CP 2012, Sect. 3.2). Among them:
 *   * `JET_VAR_MAXDOM`: the widest one;
 *   * `JET_VAR_RR`: round-robin, the one bisected the fewest times on the current branch;
 *   * `JET_VAR_SMEAR`: the maximum smear `s(x_k)` (Kearfott and Novoa, INTBIS, ACM TOMS 1990), see
 *     `jet_jacobian.hpp`;
 *   * `JET_VAR_ABS`: the adaptive bisection strategy of Granvilliers (Adaptive Bisection of Numerical CSPs,
 *     CP 2012, Sect. 3.2): with `s_l` and `s_u` the lowest and greatest smear of the candidates, the
 *     restricted candidate list (RCL) is the candidates of smear at least `S = s_l + alpha (s_u - s_l)`,
 *     with `alpha = phi(psi(s))`, `phi(y) = 1 / (1 + beta y)` and `psi` the standard deviation of the smear
 *     values (the choice of the paper, Sect. 4.2), or here optionally their coefficient of variation; the
 *     input is the one of the RCL bisected the fewest times on the current branch. `alpha = 1` is the
 *     maximum smear, `alpha = 0` makes the RCL all the candidates.
 * Ties, which the paper leaves open: the highest smear (ABS), then the lowest index (all), so that the
 * choice is deterministic.
 *
 * The same `select` chooses the ReLU to split (`-relu_order smear|abs`) and, with `-split_on mixed`, the
 * ReLU or the input (`relu_select_split` of `jet_dive_and_solve.hpp`): the candidates are then the ReLUs of
 * unknown phase (and the inputs), `smear(k)` their score, and `count(k)` the layer of a ReLU (`-relu_order
 * abs`: a ReLU is split at most once on a branch, so the earliest layer replaces the fewest bisections) or,
 * with `mixed`, the bisections of an input and 0 for a ReLU.
 *
 * `select` runs on one thread (the thread 0 of a block on the GPU): the inputs of an input split are few,
 * and the cost of a branching is the smear (`jet_jacobian.hpp`), computed by the whole block. */
namespace jet {
namespace abs {

/** What `select` computed, for the statistics and the tests. */
struct Info {
  int candidates = 0;  /**< number of candidates. */
  int rcl = 0;         /**< size of the restricted candidate list (ABS). */
  double s_l = 0, s_u = 0, mean = 0, stdev = 0, psi = 0, alpha = 1, threshold = 0;
};

/** The input to bisect among `0..n0-1`, `n0` if no input is a candidate. `candidate(k)`: whether the input
 * `k` is a candidate; `width(k)`, `smear(k)`, `count(k)`: its width, smear and number of bisections on
 * the current branch (`smear` is not read by `maxdom` and `rr`, `count` not by `maxdom` and `smear`).
 * The inputs before `start` are not candidates (`next_unassigned_var`). */
template <class Candidate, class Width, class Smear, class Count>
CUDA int select(int mode, double beta, int psi_mode, int n0, int start,
  const Candidate& candidate, const Width& width, const Smear& smear, const Count& count, Info* info = nullptr)
{
  Info local;
  Info& I = info ? *info : local;
  I = Info{};
  int best = n0;
  if(mode == JET_VAR_MAXDOM || mode == JET_VAR_RR || mode == JET_VAR_SMEAR) {
    double best_score = 0;
    for(int k = start; k < n0; ++k) {
      if(!candidate(k)) { continue; }
      ++I.candidates;
      const double s = mode == JET_VAR_MAXDOM ? static_cast<double>(width(k))
        : (mode == JET_VAR_RR ? -static_cast<double>(count(k)) : static_cast<double>(smear(k)));
      if(best == n0 || s > best_score) { best = k; best_score = s; }
    }
    return best;
  }
  /** ABS. 1. The candidates and the statistics of their smear values. */
  double sum = 0, sum2 = 0;
  bool first = true;
  for(int k = start; k < n0; ++k) {
    if(!candidate(k)) { continue; }
    const double s = static_cast<double>(smear(k));
    ++I.candidates;
    sum += s; sum2 += s * s;
    if(first) { I.s_l = s; I.s_u = s; first = false; }
    else { I.s_l = s < I.s_l ? s : I.s_l; I.s_u = s > I.s_u ? s : I.s_u; }
  }
  if(I.candidates == 0) { return n0; }
  I.mean = sum / I.candidates;
  const double var = sum2 / I.candidates - I.mean * I.mean;
  I.stdev = var > 0 ? sqrt(var) : 0.0;
  I.psi = psi_mode == ABS_PSI_CV ? (I.mean > 0 ? I.stdev / I.mean : 0.0) : I.stdev;
  /** 2. The restricted candidate list. */
  I.alpha = 1.0 / (1.0 + beta * I.psi);
  I.threshold = I.s_l + I.alpha * (I.s_u - I.s_l);
  /** `alpha = 1` must give the maximum smear: `s_l + (s_u - s_l)` may round above `s_u`. */
  if(I.threshold > I.s_u) { I.threshold = I.s_u; }
  /** 3. The member of the RCL bisected the fewest times, then the highest smear, then the lowest index. */
  int best_count = 0;
  double best_smear = 0;
  for(int k = start; k < n0; ++k) {
    if(!candidate(k)) { continue; }
    const double s = static_cast<double>(smear(k));
    if(s < I.threshold) { continue; }
    ++I.rcl;
    const int c = count(k);
    if(best == n0 || c < best_count || (c == best_count && s > best_smear)) {
      best = k; best_count = c; best_smear = s;
    }
  }
  return best;
}

/** The number of bisections of the dive of an input, estimated as the number of halvings from its width
 * `w_root` at the root of the disjunct to its width `w_sub` at the root of the subproblem (`-abs_dive_count`):
 * `round(log2(w_root / w_sub))`, 0 if it is not defined. */
CUDA INLINE int dive_bisections(double w_root, double w_sub) {
  if(!(w_sub > 0) || !(w_root > w_sub) || !(w_root < 1e300)) { return 0; }
  const double h = log2(w_root / w_sub);
  return h > 0 ? static_cast<int>(h + 0.5) : 0;
}

} // namespace abs
} // namespace jet

#endif
