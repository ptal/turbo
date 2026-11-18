// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_SIGMOID_HPP
#define TURBO_JET_SIGMOID_HPP

#include <cmath>
#include "battery/utility.hpp"
#include "lala/finterval.hpp"

/** The sigmoid activation of `-arch jet`: \f$ \sigma(z) = 1 / (1 + e^{-z}) \f$, sound enclosures of
 * its values, of its inverse \f$ \mathrm{logit}(y) = \ln(y / (1 - y)) \f$ and of its derivative
 * \f$ \sigma'(z) = \sigma(z)(1 - \sigma(z)) \f$, the interval propagator of a sigmoid neuron
 * (`tell`), and the linear relaxation used by the symbolic pass (`relax`).
 *
 * Soundness of the scalar functions. `exp`, `log` and `log1p` are not correctly rounded (neither on
 * the host nor in CUDA, which documents their largest error in ulps), and no directed rounding mode
 * exists for them. The values are computed in `double` with the default functions, then widened by
 * a relative margin `REL` (and an absolute margin `ABS` where the value can be 0): the computed value
 * of each function has an error of a few ulps of `double` (about 1e-16 relative each), far below
 * `REL = 1e-12`, so the widened interval encloses the real value. The `float` versions round the
 * widened `double` outward (`rd_cast`, `ru_cast`), whose ulp (about 6e-8 relative) makes the margin
 * invisible in the store. The `double` versions are used by the strict test of the candidate points
 * (`jet_candidates.hpp`), where the margin costs about 1e-12 relative per sigmoid.
 *
 * Properties used below: \f$ \sigma \f$ is increasing with \f$ 0 < \sigma < 1 \f$; \f$ \sigma' > 0 \f$
 * reaches its maximum 1/4 at 0 and decreases with \f$ |z| \f$; \f$ \sigma \f$ is convex on
 * \f$ z \le 0 \f$ and concave on \f$ z \ge 0 \f$.
 *
 * A NaN bound gives no information (the widest enclosure), and an infinite bound the limit. */
namespace jet {

/** The activation of a layer (`LayerInfo::has_relu`, `LayerInfo::sigmoid`), as the deductions of
 * `affine_v2::deduce` take it. */
constexpr int ACT_NONE = 0;
constexpr int ACT_RELU = 1;
constexpr int ACT_SIGMOID = 2;
template <class Layer>
CUDA INLINE int activation_of(const Layer& L) {
  return L.sigmoid != 0 ? ACT_SIGMOID : (L.has_relu != 0 ? ACT_RELU : ACT_NONE);
}

namespace sigmoid {

constexpr double REL = 1e-12;
constexpr double ABS = 1e-12;
/** An upper bound of \f$ \sigma(z) \f$ when its computed value underflows: \f$ \sigma(z) < e^{z} \f$,
 * and `exp` underflows only below -700, where \f$ e^{z} < 1e-300 \f$. */
constexpr double TINY = 1e-300;
/** Below `SMALL`, a computed value may be a subnormal number, whose relative error is not bounded:
 * a lower bound is then 0. */
constexpr double SMALL = 1e-290;

/** \f$ \sigma(z) \f$, computed without overflow (not an enclosure). */
CUDA INLINE double approx(double z) {
  if(z >= 0.0) {
    const double e = ::exp(-z);
    return 1.0 / (1.0 + e);
  }
  const double e = ::exp(z);
  return e / (1.0 + e);
}

/** A lower bound of \f$ \sigma(z) \f$ in `double`. */
CUDA INLINE double lb_d(double z) {
  if(!(z == z) || z == battery::limits<double>::neg_inf()) { return 0.0; }
  if(z == battery::limits<double>::inf()) { return 1.0 - REL; }
  const double s = approx(z);
  if(!(s >= SMALL)) { return 0.0; }
  const double r = s - s * REL;
  return r > 0.0 ? r : 0.0;
}

/** An upper bound of \f$ \sigma(z) \f$ in `double`. */
CUDA INLINE double ub_d(double z) {
  if(!(z == z) || z == battery::limits<double>::inf()) { return 1.0; }
  if(z == battery::limits<double>::neg_inf()) { return TINY; }
  const double s = approx(z);
  double r = s + s * REL;
  if(r < TINY) { r = TINY; }
  return r < 1.0 ? r : 1.0;
}

/** A lower bound of \f$ \sigma'(z) = e^{-|z|} / (1 + e^{-|z|})^2 \f$ in `double` (0 on NaN). */
CUDA INLINE double deriv_lb_d(double z) {
  if(!(z == z)) { return 0.0; }
  const double a = z < 0.0 ? -z : z;
  const double e = ::exp(-a);
  const double d = e / ((1.0 + e) * (1.0 + e));
  if(!(d >= SMALL)) { return 0.0; }
  const double r = d - d * REL;
  return r > 0.0 ? r : 0.0;
}

/** \f$ \mathrm{logit}(y) = \ln y - \ln(1 - y) \f$ for \f$ 0 < y < 1 \f$ (not an enclosure). */
CUDA INLINE double logit_approx(double y) {
  return ::log(y) - ::log1p(-y);
}

/** A lower bound of \f$ \{ z : \sigma(z) \ge y \} \f$: \f$ -\infty \f$ when \f$ y \le 0 \f$ (or NaN),
 * \f$ +\infty \f$ when \f$ y \ge 1 \f$ (no real \f$ z \f$ gives \f$ \sigma(z) \ge 1 \f$). */
CUDA INLINE double logit_lb_d(double y) {
  if(!(y == y) || y <= 0.0) { return battery::limits<double>::neg_inf(); }
  if(y >= 1.0) { return battery::limits<double>::inf(); }
  const double L = logit_approx(y);
  const double m = (L < 0.0 ? -L : L) * REL + ABS;
  return L - m;
}

/** An upper bound of \f$ \{ z : \sigma(z) \le y \} \f$: \f$ +\infty \f$ when \f$ y \ge 1 \f$ (or NaN),
 * \f$ -\infty \f$ when \f$ y \le 0 \f$. */
CUDA INLINE double logit_ub_d(double y) {
  if(!(y == y) || y >= 1.0) { return battery::limits<double>::inf(); }
  if(y <= 0.0) { return battery::limits<double>::neg_inf(); }
  const double L = logit_approx(y);
  const double m = (L < 0.0 ? -L : L) * REL + ABS;
  return L + m;
}

/** The enclosures in the type `T` of the store (`float` or `double`), rounded outward. */
template <class T> CUDA INLINE T lb(T z) { return battery::rd_cast<T>(lb_d(static_cast<double>(z))); }
template <class T> CUDA INLINE T ub(T z) { return battery::ru_cast<T>(ub_d(static_cast<double>(z))); }
template <class T> CUDA INLINE T deriv_lb(T z) { return battery::rd_cast<T>(deriv_lb_d(static_cast<double>(z))); }
template <class T> CUDA INLINE T logit_lb(T y) { return battery::rd_cast<T>(logit_lb_d(static_cast<double>(y))); }
template <class T> CUDA INLINE T logit_ub(T y) { return battery::ru_cast<T>(logit_ub_d(static_cast<double>(y))); }

/** The propagator of \f$ y = \sigma(p) \f$ on two intervals: forward, \f$ y \gets y \cap
 * [\sigma(p_{lb}), \sigma(p_{ub})] \f$; backward, \f$ p \gets p \cap [\mathrm{logit}(y_{lb}),
 * \mathrm{logit}(y_{ub})] \f$, both enclosures rounded outward. \f$ \sigma \f$ is increasing and
 * invertible, so both projections are the exact hulls up to the rounding. */
template <class T>
CUDA INLINE void tell(lala::FInterval<T>& y, lala::FInterval<T>& p) {
  using RItv = lala::FInterval<T>;
  if(p.is_bot() || y.is_bot()) {
    y.meet_bot();
    p.meet_bot();
    return;
  }
  y.meet(RItv(lb<T>(p.lb().load()), ub<T>(p.ub().load())));
  if(y.is_bot()) {
    p.meet_bot();
    return;
  }
  p.meet(RItv(logit_lb<T>(y.lb().load()), logit_ub<T>(y.ub().load())));
}

/** The lower bound of the slope of the chord of \f$ \sigma \f$ between `l < u` (finite), at least 0. */
template <class T>
CUDA INLINE T chord_lb(T l, T u) {
  const T num = battery::sub_down<T>(lb<T>(u), ub<T>(l));
  if(!(num > T{0})) { return T{0}; }
  const T r = battery::div_down<T>(num, battery::sub_up<T>(u, l));
  return r > T{0} ? r : T{0};
}

/** Two lines enclosing \f$ \sigma \f$ on \f$ [l, u] \f$:
 *   \f$ a_L z + b_L \le \sigma(z) \le a_U z + b_U \f$ for every real \f$ z \in [l, u] \f$, with
 *   \f$ a_L, a_U \ge 0 \f$ (so that a lower form of \f$ z \f$ gives a lower form of \f$ \sigma(z) \f$).
 * Each line goes through an end of the interval, with a slope rounded down (`deriv_lb`, `chord_lb`):
 *   * lower line \f$ \underline{\sigma}(l) + s_L (z - l) \f$: on \f$ z \ge l \f$, a smaller slope
 *     only lowers it;
 *   * upper line \f$ \overline{\sigma}(u) + s_U (z - u) \f$: on \f$ z \le u \f$, a smaller slope
 *     only raises it.
 * The slopes, by the shape of \f$ \sigma \f$:
 *   * \f$ u \le 0 \f$ (convex): \f$ s_L = \sigma'(l) \f$ (the tangent at \f$ l \f$ is below), \f$ s_U \f$
 *     the chord (above);
 *   * \f$ l \ge 0 \f$ (concave): \f$ s_L \f$ the chord (below), \f$ s_U = \sigma'(u) \f$ (the tangent
 *     at \f$ u \f$ is above);
 *   * \f$ l < 0 < u \f$: \f$ s_L = s_U = \min(\sigma'(l), \sigma'(u)) \f$, a lower bound of
 *     \f$ \sigma' \f$ on \f$ [l, u] \f$ (\f$ \sigma' \f$ decreases with \f$ |z| \f$), so that
 *     \f$ \sigma(z) - \sigma(l) = \int_l^z \sigma' \ge s (z - l) \f$ and likewise from \f$ u \f$ (the
 *     relaxation of DeepPoly for the S-shaped activations).
 * The lines are written \f$ a z + b \f$ with \f$ b_L \f$ rounded down and \f$ b_U \f$ rounded up.
 * With an infinite or NaN end, or \f$ l \ge u \f$, the lines are the constants
 * \f$ \underline{\sigma}(l) \f$ and \f$ \overline{\sigma}(u) \f$ (\f$ \sigma \f$ is increasing). */
template <class T>
CUDA void relax(T l, T u, T& aL, T& bL, T& aU, T& bU) {
  const T s_l = lb<T>(l);
  const T s_u = ub<T>(u);
  aL = T{0}; bL = s_l;
  aU = T{0}; bU = s_u;
  if(!(l < u) || !(l > battery::limits<T>::neg_inf()) || !(u < battery::limits<T>::inf())) { return; }
  T sL, sU;
  if(u <= T{0}) {
    sL = deriv_lb<T>(l);
    sU = chord_lb<T>(l, u);
  }
  else if(l >= T{0}) {
    sL = chord_lb<T>(l, u);
    sU = deriv_lb<T>(u);
  }
  else {
    const T dl = deriv_lb<T>(l), du = deriv_lb<T>(u);
    sL = sU = dl < du ? dl : du;
  }
  if(sL > T{0}) {
    const T b = battery::sub_down<T>(s_l, battery::mul_up<T>(sL, l));
    if(b == b && b > battery::limits<T>::neg_inf()) { aL = sL; bL = b; }
  }
  if(sU > T{0}) {
    const T b = battery::sub_up<T>(s_u, battery::mul_down<T>(sU, u));
    if(b == b && b < battery::limits<T>::inf()) { aU = sU; bU = b; }
  }
}

} // namespace sigmoid
} // namespace jet

#endif
