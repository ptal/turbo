// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_SCIP_HPP
#define TURBO_JET_SCIP_HPP

/** `-under scip` (`-arch jet`): the counterexamples of the nodes are looked for by SCIP in exact mode,
 * on the CPU, while the GPU goes on with its search.
 *
 * The flow. In the search of a block (`jet_dive_and_solve.hpp`, `scip_offer`), the first node of a path
 * with at most `-scip_max_unstable` unstable ReLUs (`relu_split::is_candidate` at `jet_split_point`) is
 * copied to a queue in pinned memory: the bounds of all its neurons and its disjunct. The block does not
 * wait: it searches the subtree of the node as usual, and its descendants are not sent (the node covers
 * them). A dispatcher thread of the host moves the nodes of the queue to a `Manager`, whose
 * `-scip_threads` workers solve them in parallel, one MIP per node (`solve_node`). A worker is a
 * process (the executable started again with `worker_flag()`, `worker_main`), not a thread: SCIP in
 * exact mode is not thread-safe (the LU factorization of its project-and-shift method has global
 * state), and a process can be killed as soon as the answer is known. The answers of SCIP prune
 * nothing: a node without counterexample, or out of time, is only counted.
 *
 * The MIP of a node (`solve_node`). Exact rational arithmetic: every coefficient is a `float` of the
 * network or a `double` bound, so it is a rational without rounding, and SCIP solves the MIP exactly.
 *   * One variable `y_i` per neuron, with the bounds of the node (the inputs also meet the region of the
 *     disjunct, see below).
 *   * A neuron without activation: `y_i = e_i`, where `e_i` is the affine expression of its layer
 *     (`W y_src + b`, dense or with the stored weights of a sparse layer, `y_a + y_b`, `s y_src + t`).
 *   * A ReLU neuron, with `[l, u]` the bounds of its pre-activation (an interval evaluation of `e_i` on
 *     the bounds of the node, met with `e_i <= ub(y_i)` and, when `lb(y_i) > 0`, `e_i >= lb(y_i)`):
 *     inactive (`u <= 0`): `y_i = 0` and `e_i <= 0`; active (`l >= 0`): `y_i = e_i` and `y_i >= 0`;
 *     unstable: a binary `z_i` and `y_i >= e_i`, `y_i >= 0`, `y_i <= e_i - l (1 - z_i)`, `y_i <= u z_i`.
 *     Whatever `l` and `u` are, a solution has `y_i = max(e_i, 0)` exactly: the bounds only decide which
 *     points of the node are kept, never the value of a neuron. So a solution of the MIP is an exact
 *     evaluation of the real-valued network.
 *   * Numerics. The bounds of the node are first met with an interval evaluation from the inputs, so that
 *     they are finite and of the size of the values of the network (a node with a bound above `kHuge` is
 *     `UNSUPPORTED`); a pre-activation within `kTiny` of 0 on one side only takes the phase of the other
 *     side (the branch `y <= jet_split_point()` of a ReLU split gives `u = FLT_MIN`, a big-M coefficient
 *     that SCIP turns into an infinite one); and the MILP presolver (PaPILO) is disabled.
 *   * The property, by a margin `s` in `[0, 1]`, maximized: `y_o - s >= c` and `y_o + s <= c` for the
 *     bounds of the outputs, `y_x - y_y + s <= k` for the comparisons. The bounds of the region are those
 *     of the strict test (`apply_disjunct_inner`: the vnnlib bounds rounded inward). SCIP stops as soon
 *     as a solution reaches the margin `-scip_margin` (`limits/primal`): a point on the boundary of the
 *     region (`s = 0`) is a counterexample of the real-valued network, but it fails the strict test.
 *   * The sigmoid has no exact linear encoding: a network with a sigmoid is not supported (`supported`).
 *
 * The point. The inputs of the best solution are rounded to the nearest `float` (clamped to their range,
 * whose ends are floats), and the point is a counterexample only if it passes the strict test of JET
 * (`strict_check`: `candidates::evaluate` in `double` rounded outward, `neuron_ok`, `leq_ok`), as the
 * candidates and the leaves. Exact mode is not needed for this soundness, the strict test decides; it
 * makes the answers of SCIP (no counterexample in a node) exact, for the statistics and later uses.
 *
 * Everything here runs on the host: `HostNet` is a copy of the network (the workers never read the
 * managed memory of the kernel), and `Region` a copy of the inner bounds of a disjunct. The SCIP part
 * (`solve_node`, `worker_main`, `Manager`) needs `JET_WITH_SCIP` (`cmake -DWITH_SCIP=ON`) and Linux
 * (`posix_spawn`, `/proc/self/exe`); `supported`, `count_unstable`, `pre_activation` and `strict_check`
 * do not.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "jet_candidates.hpp"
#include "relu_split.hpp"

#ifdef JET_WITH_SCIP
#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <scip/cons_exactlinear.h>
#include <scip/rational.h>
#include <scip/scip_exact.h>
#include <cerrno>
#include <csignal>
#include <memory>
#include <set>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace jet {
namespace scip_under {

/** A copy of a layer of the network (the fields of `jet::LayerInfo`). */
struct HostLayer { int kind, base, size, src0, src1, weight_base, bias_base, has_relu, sigmoid, csc_base; };

/** A copy of the network on the host (the fields read by `candidates::evaluate`). */
struct HostNet {
  int num_neurons = 0;
  int output_layer = 0;
  std::vector<HostLayer> layers;
  std::vector<float> weights;
  std::vector<float> biases;
  /** The rows of the sparse layers (`symbolic::KIND_SPARSE`); the columns are not used here. */
  std::vector<int> sp_row, sp_col;
  std::vector<float> sp_val;
  HostNet() = default;
  template <class Net>
  explicit HostNet(const Net& net) : num_neurons(net.num_neurons), output_layer(net.output_layer) {
    for(int l = 0; l < static_cast<int>(net.layers.size()); ++l) {
      const auto& L = net.layers[l];
      layers.push_back(HostLayer{L.kind, L.base, L.size, L.src0, L.src1, L.weight_base, L.bias_base, L.has_relu, L.sigmoid, L.csc_base});
    }
    weights.assign(net.weights.data(), net.weights.data() + net.weights.size());
    biases.assign(net.biases.data(), net.biases.data() + net.biases.size());
    sp_row.assign(net.sp_row.data(), net.sp_row.data() + net.sp_row.size());
    sp_col.assign(net.sp_col.data(), net.sp_col.data() + net.sp_col.size());
    sp_val.assign(net.sp_val.data(), net.sp_val.data() + net.sp_val.size());
  }
  int num_inputs() const { return layers.empty() ? 0 : layers[0].size; }
  int num_layers() const { return static_cast<int>(layers.size()); }
};

/** A comparison `y[x] - y[y] <= k` of the region. */
struct Leq { int x, y; double k; };

/** The region of a disjunct for the strict test: bounds of every neuron (infinite when unconstrained)
 * and the comparisons (`apply_disjunct_inner`). */
struct Region {
  std::vector<double> lb, ub;
  std::vector<Leq> leqs;
  /** The region rounded outward (`apply_disjunct`, the region of the propagation), for the MIP of the
   * proof mode (`Options::prove`); empty: the inner one. */
  std::vector<double> olb, oub;
  std::vector<Leq> oleqs;
  int size() const { return static_cast<int>(leqs.size()); }
};

/** A node sent by the GPU: its disjunct (an index of the DNF), its sequence number (the order of the
 * sends), and the bounds of its neurons. */
struct Job {
  int dnf = 0;
  unsigned long long seq = 0;
  std::vector<float> lb, ub;
};

enum Status {
  FOUND = 0,        /**< a point that passes the strict test. */
  NONE = 1,         /**< no counterexample of margin `-scip_margin` in the node (exact). */
  LIMIT = 2,        /**< the time limit of the node. */
  UNCERTIFIED = 3,  /**< a solution of SCIP whose point fails the strict test. */
  UNSUPPORTED = 4,  /**< an infinite bound where a finite one is needed. */
  FAILED = 5,       /**< an error of SCIP. */
  CANCELLED = 6,    /**< cancelled before it was solved. */
  CLOSED = 7        /**< proof mode: no point of margin >= 0 in the node and the outer region (exact). */
};
constexpr int kNumStatus = 8;

/** How a solution of SCIP was taken (`Result::cex_path`), for the statistics (`scip_cex_*`):
 *   * `CEX_EXACT`: the exact rational point (`-scip_cex exact`);
 *   * `CEX_FLOAT`: rounded to floats and checked by the strict test (`-scip_cex float`, or an inexact
 *     solution);
 *   * `CEX_RELAXED`: `-scip_cex exact`, but the proof mode relaxed a ReLU, so the point was rounded to floats
 *     and checked by the strict test (exception 1);
 *   * `CEX_OUTSIDE`: `-scip_cex exact`, but the exact point lies outside the inner region (the MIP of the
 *     proof mode is over the outer one), so the point was rounded to floats and checked by the strict test
 *     (exception 2). */
constexpr int CEX_NONE = 0;
constexpr int CEX_EXACT = 1;
constexpr int CEX_FLOAT = 2;
constexpr int CEX_RELAXED = 3;
constexpr int CEX_OUTSIDE = 4;
constexpr int kNumCexPaths = 5;
inline const char* name_of_cex_path(int p) {
  return p == CEX_EXACT ? "exact" : (p == CEX_FLOAT ? "float" : (p == CEX_RELAXED ? "relaxed" : (p == CEX_OUTSIDE ? "outside" : "none")));
}

inline const char* name_of_status(Status s) {
  switch(s) {
    case FOUND: return "found";
    case NONE: return "none";
    case LIMIT: return "limit";
    case UNCERTIFIED: return "uncertified";
    case UNSUPPORTED: return "unsupported";
    case FAILED: return "failed";
    case CLOSED: return "closed";
    default: return "cancelled";
  }
}

struct Options {
  double time_limit = 10.0;  /**< seconds per node (`-scip_time`). */
  double margin = 1e-6;      /**< `-scip_margin`. */
  /** The proof mode (`-scip_mode race`): the MIP is a relaxation of the node over the region rounded
   * outward (`Region::olb`, `oub`, `oleqs`), with the margin `s >= 0`, so that an infeasible MIP proves
   * that the node has no counterexample (`CLOSED`): a pre-activation within `kTiny` of 0 is relaxed
   * (`0 <= y <= kTiny`, `y >= p` for `p <= kTiny`; `p <= y <= p + kTiny`, `y >= 0` for `p >= -kTiny`)
   * instead of being given a phase, and the end of the solving without solution is `CLOSED` only when
   * SCIP proved the infeasibility. A solution is still checked by the strict test over the inner region. */
  bool prove = false;
  /** `-scip_cex exact` (the default): a solution of SCIP in exact mode is a counterexample of the
   * real-valued network as it is, without rounding nor strict test: its exact values satisfy the network
   * (`y = max(e, 0)` exactly, see the top of the file), and it is checked, in rational arithmetic, to lie in
   * the inner region (the vnnlib bounds rounded inward, inside the real region: the MIP of the proof mode
   * is over the outer one). `-scip_cex float`: the inputs are rounded to the nearest floats and the point
   * must pass the strict test. */
  bool exact_cex = true;
};

struct Result {
  Status status = FAILED;
  /** With `FOUND`: the evaluation of the point, `[lo[i], hi[i]]` for every neuron (`candidates::evaluate`). */
  std::vector<double> lo, hi;
  /** The margin `s` of the best solution of SCIP, -1 without solution. */
  double margin = -1.0;
  int binaries = 0;
  double seconds = 0.0;
  /** With `FOUND` and `Options::exact_cex`: the exact point of SCIP, the rational value of every input as
   * `p/q`, one per line (`lo`, `hi` are then the values of the neurons rounded outward to doubles); empty
   * when the point was rounded to floats and checked by the strict test. */
  std::string exact;
  /** How a solution of SCIP was taken (`CexPath`), `CEX_NONE` without solution. */
  int cex_path = 0;
};

/** Whether `-under scip` supports the network: no sigmoid. */
template <class Net>
bool supported(const Net& net, std::string& why) {
  for(int l = 1; l < net.num_layers(); ++l) {
    if(net.layers[l].sigmoid != 0) { why = "the network has a sigmoid layer"; return false; }
  }
  return true;
}

/** The number of ReLU neurons of unknown phase in bounds `lb`, `ub` (sequential version of the count of
 * `scip_offer`). */
template <class Net>
int count_unstable(const Net& net, const float* lb, const float* ub) {
  const float tau = relu_split::jet_split_point<float>();
  int n = 0;
  for(int l = 1; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    if(L.has_relu == 0) { continue; }
    for(int j = 0; j < L.size; ++j) {
      n += relu_split::is_candidate(lb[L.base + j], ub[L.base + j], tau) ? 1 : 0;
    }
  }
  return n;
}

/** The interval of the affine expression of the neuron `j` of the layer `l` (before its activation) on
 * the bounds `ylo`, `yhi` of its sources, in `double` rounded outward. */
template <class Net>
void pre_activation(const Net& net, int l, int j, const std::vector<double>& ylo, const std::vector<double>& yhi,
  double& a, double& b)
{
  const auto& L = net.layers[l];
  if(L.kind == symbolic::KIND_ADD) {
    const int s0 = net.layers[L.src0].base + j, s1 = net.layers[L.src1].base + j;
    a = battery::add_down<double>(ylo[s0], ylo[s1]);
    b = battery::add_up<double>(yhi[s0], yhi[s1]);
  }
  else if(L.kind == symbolic::KIND_SCALE) {
    const int src = net.layers[L.src0].base + j;
    double pa, pb;
    candidates::product(static_cast<double>(net.weights[L.weight_base + j]), ylo[src], yhi[src], pa, pb);
    const double t = static_cast<double>(net.biases[L.bias_base + j]);
    a = battery::add_down<double>(pa, t);
    b = battery::add_up<double>(pb, t);
  }
  else if(L.kind == symbolic::KIND_SPARSE) {
    a = b = static_cast<double>(net.biases[L.bias_base + j]);
    const auto& S = net.layers[L.src0];
    int begin, end;
    symbolic::sparse_row(net, L, j, begin, end);
    for(int e = begin; e < end; ++e) {
      const double w = static_cast<double>(net.sp_val[e]);
      if(w == 0.0) { continue; }
      double pa, pb;
      candidates::product(w, ylo[S.base + net.sp_col[e]], yhi[S.base + net.sp_col[e]], pa, pb);
      a = battery::add_down<double>(a, pa);
      b = battery::add_up<double>(b, pb);
    }
  }
  else {
    a = b = static_cast<double>(net.biases[L.bias_base + j]);
    const auto& S = net.layers[L.src0];
    for(int c = 0; c < S.size; ++c) {
      const double w = static_cast<double>(net.weights[L.weight_base + c * L.size + j]);
      if(w == 0.0) { continue; }
      double pa, pb;
      candidates::product(w, ylo[S.base + c], yhi[S.base + c], pa, pb);
      a = battery::add_down<double>(a, pa);
      b = battery::add_up<double>(b, pb);
    }
  }
}

struct HostGroup {
  int rank() const { return 0; }
  int size() const { return 1; }
  void sync() const {}
};

/** The strict test of JET on the point `x` (the inputs): evaluates the network in `lo`, `hi`, and
 * returns whether every bound and comparison of the region holds on the whole intervals. */
template <class Net>
bool strict_check(const Net& net, const Region& region, const std::vector<double>& x,
  std::vector<double>& lo, std::vector<double>& hi)
{
  lo.assign(net.num_neurons, 0.0);
  hi.assign(net.num_neurons, 0.0);
  for(int k = 0; k < net.num_inputs(); ++k) { lo[k] = hi[k] = x[k]; }
  HostGroup group;
  candidates::evaluate(group, net, lo.data(), hi.data());
  struct Inner {
    const Region& r;
    double lb(int i) const { return r.lb[i]; }
    double ub(int i) const { return r.ub[i]; }
  } inner{region};
  for(int i = 0; i < net.num_neurons; ++i) {
    if(!candidates::neuron_ok(i, lo.data(), hi.data(), inner)) { return false; }
  }
  for(const Leq& q : region.leqs) {
    if(!candidates::leq_ok(q, lo.data(), hi.data())) { return false; }
  }
  return true;
}

/** The nearest `float` to `v`, clamped to `[lo, hi]` (floats). */
inline double nearest_float_in(double v, double lo, double hi) {
  double f = static_cast<double>(static_cast<float>(v));
  if(f < lo) { f = lo; }
  if(f > hi) { f = hi; }
  return f;
}

#ifdef JET_WITH_SCIP

namespace detail {

/** The rationals of one constraint, freed by the destructor. */
struct Row {
  SCIP* scip;
  std::vector<SCIP_VAR*> vars;
  std::vector<double> vals;
  explicit Row(SCIP* s) : scip(s) {}
  void add(SCIP_VAR* v, double c) { if(c != 0.0) { vars.push_back(v); vals.push_back(c); } }
  /** Adds `lhs <= sum vals * vars <= rhs` (infinite sides allowed) as an exact linear constraint. */
  SCIP_RETCODE post(const char* name, double lhs, double rhs) {
    const int n = static_cast<int>(vars.size());
    SCIP_RATIONAL** coefs = nullptr;
    SCIP_RATIONAL* l = nullptr;
    SCIP_RATIONAL* r = nullptr;
    SCIP_CALL( SCIPrationalCreateBufferArray(SCIPbuffer(scip), &coefs, std::max(n, 1)) );
    SCIP_CALL( SCIPrationalCreateBuffer(SCIPbuffer(scip), &l) );
    SCIP_CALL( SCIPrationalCreateBuffer(SCIPbuffer(scip), &r) );
    for(int k = 0; k < n; ++k) { SCIPrationalSetReal(coefs[k], vals[k]); }
    if(std::isinf(lhs)) { SCIPrationalSetNegInfinity(l); } else { SCIPrationalSetReal(l, lhs); }
    if(std::isinf(rhs)) { SCIPrationalSetInfinity(r); } else { SCIPrationalSetReal(r, rhs); }
    SCIP_CONS* cons = nullptr;
    SCIP_CALL( SCIPcreateConsBasicExactLinear(scip, &cons, name, n, vars.data(), coefs, l, r) );
    SCIP_CALL( SCIPaddCons(scip, cons) );
    SCIP_CALL( SCIPreleaseCons(scip, &cons) );
    SCIPrationalFreeBuffer(SCIPbuffer(scip), &r);
    SCIPrationalFreeBuffer(SCIPbuffer(scip), &l);
    SCIPrationalFreeBufferArray(SCIPbuffer(scip), &coefs, std::max(n, 1));
    return SCIP_OKAY;
  }
};

inline SCIP_RETCODE make_var(SCIP* scip, SCIP_VAR** var, const char* name, double lb, double ub, double obj, SCIP_VARTYPE type) {
  const double inf = SCIPinfinity(scip);
  const double l = std::isinf(lb) ? -inf : lb;
  const double u = std::isinf(ub) ? inf : ub;
  SCIP_CALL( SCIPcreateVarBasic(scip, var, name, l, u, obj, type) );
  /** The exact data of the variable from its floating-point data, which are exact (`SCIPvarAddExactData`). */
  SCIP_CALL( SCIPaddVarExactData(scip, *var, nullptr, nullptr, nullptr) );
  SCIP_CALL( SCIPaddVar(scip, *var) );
  return SCIP_OKAY;
}

/** Bounds of a magnitude above `kHuge` are not sent to SCIP (`UNSUPPORTED`): its infinity is 1e20, and
 * its scaling of the big-M constraints multiplies them. A pre-activation bound of magnitude below `kTiny`
 * is taken as 0 (see `build_and_solve`). */
constexpr double kHuge = 1e12;
constexpr double kTiny = 1e-9;

/** Builds and solves the MIP of a node, see the top of the file. `vars` and `result` are filled. */
template <class Net>
SCIP_RETCODE build_and_solve(SCIP* scip, const Net& net, const Region& region, const float* nlb, const float* nub,
  const Options& opt, const std::atomic<bool>& cancel, Result& result)
{
  const int N = net.num_neurons;
  const int n0 = net.num_inputs();
  /** The region of the MIP: rounded outward in the proof mode, so that its infeasibility is a proof; a
   * node shown empty is then `CLOSED`. */
  const bool prove = opt.prove;
  const bool outer = prove && !region.olb.empty();
  const std::vector<double>& RLB = outer ? region.olb : region.lb;
  const std::vector<double>& RUB = outer ? region.oub : region.ub;
  const std::vector<Leq>& RLEQ = outer ? region.oleqs : region.leqs;
  const Status EMPTY = prove ? CLOSED : NONE;
  /** The ReLUs relaxed by the proof mode: a solution is then not an exact evaluation of the network. */
  int relaxed = 0;
  std::vector<double> ylo(N), yhi(N);
  for(int i = 0; i < N; ++i) { ylo[i] = static_cast<double>(nlb[i]); yhi[i] = static_cast<double>(nub[i]); }
  for(int k = 0; k < n0; ++k) {
    ylo[k] = std::max(ylo[k], RLB[k]);
    yhi[k] = std::min(yhi[k], RUB[k]);
    if(!(ylo[k] <= yhi[k])) { result.status = EMPTY; return SCIP_OKAY; }
    if(!std::isfinite(ylo[k]) || !std::isfinite(yhi[k])) { result.status = UNSUPPORTED; return SCIP_OKAY; }
  }
  /** The bounds of the node, met with an interval evaluation from the inputs, layer by layer: every bound
   * is then finite and of the size of the values of the network, whatever the store of the node holds
   * (huge or infinite bounds of the float store would become infinite coefficients for SCIP). `plo`, `phi`:
   * the bounds of the pre-activations of the ReLU neurons (`pre_activation`, met with `p <= ub(y)` and,
   * when `lb(y) > 0`, `p >= lb(y)`). */
  std::vector<double> plo(N, 0.0), phi(N, 0.0);
  for(int l = 1; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    for(int j = 0; j < L.size; ++j) {
      const int i = L.base + j;
      double a, b;
      pre_activation(net, l, j, ylo, yhi, a, b);
      if(L.has_relu != 0) {
        b = std::min(b, yhi[i]);
        if(ylo[i] > 0.0) { a = std::max(a, ylo[i]); }
        plo[i] = a;
        phi[i] = b;
        a = std::max(a, 0.0);
        b = std::max(b, 0.0);
      }
      ylo[i] = std::max(ylo[i], a);
      yhi[i] = std::min(yhi[i], b);
      if(!(ylo[i] <= yhi[i])) { result.status = EMPTY; return SCIP_OKAY; }
      if(!std::isfinite(ylo[i]) || !std::isfinite(yhi[i]) || std::fabs(ylo[i]) > kHuge || std::fabs(yhi[i]) > kHuge) {
        result.status = UNSUPPORTED;
        return SCIP_OKAY;
      }
    }
  }
  /** A bound of a hidden neuron within `kTiny` of 0 on the wrong side is moved to 0 (outward: the node is
   * only enlarged, e.g. `y >= jet_split_point()` of the active branch of a ReLU split becomes `y >= 0`, the
   * phase being kept by `p >= 0`). */
  for(int i = n0; i < N; ++i) {
    if(ylo[i] > 0.0 && ylo[i] < kTiny) { ylo[i] = 0.0; }
    if(yhi[i] < 0.0 && yhi[i] > -kTiny) { yhi[i] = 0.0; }
  }

  SCIPsetMessagehdlrQuiet(scip, TRUE);
  SCIP_CALL( SCIPincludeDefaultPlugins(scip) );
  SCIP_CALL( SCIPenableExactSolving(scip, TRUE) );
  SCIP_CALL( SCIPcreateProbBasic(scip, "jet_node") );
  SCIP_CALL( SCIPsetObjsense(scip, SCIP_OBJSENSE_MAXIMIZE) );
  SCIP_CALL( SCIPsetRealParam(scip, "limits/primal", opt.margin) );
  /** No MILP presolver (PaPILO), when SCIP is built with it: in exact mode it scales the big-M constraints
   * into infinite coefficients when a bound is close to 0 (`[cons_exactlinear.c] coefficient ... is
   * infinite`), and on the small MIPs of the nodes it made SCIP hit its time limit where it otherwise
   * answers in milliseconds (`tests/scip_under_test.cpp` with SCIP 10.1 and PaPILO 3.0.2). */
  if(SCIPgetParam(scip, "presolving/milp/maxrounds") != nullptr) {
    SCIP_CALL( SCIPsetIntParam(scip, "presolving/milp/maxrounds", 0) );
  }

  std::vector<SCIP_VAR*> y(N, nullptr);
  std::vector<SCIP_VAR*> extra;
  char name[64];
  /** The pre-activation bounds and the phase of the ReLU neurons, decided before their variable. */
  for(int k = 0; k < n0; ++k) {
    std::snprintf(name, sizeof(name), "x%d", k);
    SCIP_CALL( make_var(scip, &y[k], name, ylo[k], yhi[k], 0.0, SCIP_VARTYPE_CONTINUOUS) );
  }
  int binaries = 0;
  for(int l = 1; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    for(int j = 0; j < L.size; ++j) {
      const int i = L.base + j;
      /** The affine expression `e_i = sum terms + bias`. */
      Row row(scip);
      double bias = 0.0;
      if(L.kind == symbolic::KIND_ADD) {
        row.add(y[net.layers[L.src0].base + j], 1.0);
        row.add(y[net.layers[L.src1].base + j], 1.0);
      }
      else if(L.kind == symbolic::KIND_SCALE) {
        row.add(y[net.layers[L.src0].base + j], static_cast<double>(net.weights[L.weight_base + j]));
        bias = static_cast<double>(net.biases[L.bias_base + j]);
      }
      else if(L.kind == symbolic::KIND_SPARSE) {
        const auto& S = net.layers[L.src0];
        int begin, end;
        symbolic::sparse_row(net, L, j, begin, end);
        for(int e = begin; e < end; ++e) {
          row.add(y[S.base + net.sp_col[e]], static_cast<double>(net.sp_val[e]));
        }
        bias = static_cast<double>(net.biases[L.bias_base + j]);
      }
      else {
        const auto& S = net.layers[L.src0];
        for(int c = 0; c < S.size; ++c) {
          row.add(y[S.base + c], static_cast<double>(net.weights[L.weight_base + c * L.size + j]));
        }
        bias = static_cast<double>(net.biases[L.bias_base + j]);
      }
      std::snprintf(name, sizeof(name), "y%d", i);
      if(L.has_relu == 0) {
        /** y_i - sum = bias. */
        SCIP_CALL( make_var(scip, &y[i], name, ylo[i], yhi[i], 0.0, SCIP_VARTYPE_CONTINUOUS) );
        row.add(y[i], -1.0);
        for(double& v : row.vals) { v = -v; }  // y_i - sum
        std::snprintf(name, sizeof(name), "def%d", i);
        SCIP_CALL( row.post(name, bias, bias) );
        continue;
      }
      const double pl = plo[i], pu = phi[i];
      /** A pre-activation within `kTiny` of 0 on one side only is given the phase of the other side:
       * `p <= 0` when `pu < kTiny` (e.g. the inactive branch `y <= jet_split_point()` of a ReLU split, where
       * `pu = FLT_MIN`), `p >= 0` when `pl > -kTiny`. This keeps a part of the node only, so a solution is
       * still an exact evaluation of the network, and it avoids the coefficients `u` or `l` of the big-M
       * constraints close to 0, which the exact presolving of SCIP turns into infinite ones when it scales
       * the constraint (`cap: y - u z <= 0` with `u = 1.2e-38`). */
      if(prove && pl < 0.0 && pu > 0.0 && pu < kTiny) {
        /** Proof mode, almost inactive: 0 <= y <= kTiny and y >= p (a relaxation, no phase). */
        ++relaxed;
        SCIP_CALL( make_var(scip, &y[i], name, std::max(ylo[i], 0.0), kTiny, 0.0, SCIP_VARTYPE_CONTINUOUS) );
        Row ge(scip);
        ge.add(y[i], 1.0);
        for(size_t t = 0; t < row.vars.size(); ++t) { ge.add(row.vars[t], -row.vals[t]); }
        std::snprintf(name, sizeof(name), "rge%d", i);
        SCIP_CALL( ge.post(name, bias, std::numeric_limits<double>::infinity()) );
      }
      else if(prove && pl < 0.0 && pl > -kTiny && pu > 0.0) {
        /** Proof mode, almost active: p <= y <= p + kTiny and y >= 0 (`y - p = max(-p, 0) <= -l`). */
        ++relaxed;
        SCIP_CALL( make_var(scip, &y[i], name, std::max(ylo[i], 0.0), yhi[i], 0.0, SCIP_VARTYPE_CONTINUOUS) );
        Row band(scip);
        band.add(y[i], 1.0);
        for(size_t t = 0; t < row.vars.size(); ++t) { band.add(row.vars[t], -row.vals[t]); }
        std::snprintf(name, sizeof(name), "rband%d", i);
        SCIP_CALL( band.post(name, bias, bias + kTiny) );
      }
      else if(pu <= 0.0 || (pl < 0.0 && pu < kTiny)) {
        /** Inactive: y_i = 0 and sum + bias <= 0. */
        if(ylo[i] > 0.0) { result.status = EMPTY; return SCIP_OKAY; }
        SCIP_CALL( make_var(scip, &y[i], name, 0.0, 0.0, 0.0, SCIP_VARTYPE_CONTINUOUS) );
        std::snprintf(name, sizeof(name), "off%d", i);
        SCIP_CALL( row.post(name, -std::numeric_limits<double>::infinity(), -bias) );
      }
      else if(pl >= 0.0 || pl > -kTiny) {
        /** Active: y_i = sum + bias, y_i >= 0. */
        SCIP_CALL( make_var(scip, &y[i], name, std::max(ylo[i], 0.0), yhi[i], 0.0, SCIP_VARTYPE_CONTINUOUS) );
        row.add(y[i], -1.0);
        for(double& v : row.vals) { v = -v; }
        std::snprintf(name, sizeof(name), "on%d", i);
        SCIP_CALL( row.post(name, bias, bias) );
      }
      else {
        if(!std::isfinite(pl) || !std::isfinite(pu)) { result.status = UNSUPPORTED; return SCIP_OKAY; }
        ++binaries;
        SCIP_CALL( make_var(scip, &y[i], name, std::max(ylo[i], 0.0), yhi[i], 0.0, SCIP_VARTYPE_CONTINUOUS) );
        SCIP_VAR* z = nullptr;
        std::snprintf(name, sizeof(name), "z%d", i);
        SCIP_CALL( make_var(scip, &z, name, 0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY) );
        extra.push_back(z);
        /** y_i - sum >= bias. */
        Row ge(scip);
        ge.add(y[i], 1.0);
        for(size_t t = 0; t < row.vars.size(); ++t) { ge.add(row.vars[t], -row.vals[t]); }
        std::snprintf(name, sizeof(name), "ge%d", i);
        SCIP_CALL( ge.post(name, bias, std::numeric_limits<double>::infinity()) );
        /** y_i - sum - l z_i <= bias - l. */
        Row up(scip);
        up.add(y[i], 1.0);
        for(size_t t = 0; t < row.vars.size(); ++t) { up.add(row.vars[t], -row.vals[t]); }
        up.add(z, -pl);
        std::snprintf(name, sizeof(name), "up%d", i);
        SCIP_CALL( up.post(name, -std::numeric_limits<double>::infinity(), bias - pl) );
        /** y_i - u z_i <= 0. */
        Row cap(scip);
        cap.add(y[i], 1.0);
        cap.add(z, -pu);
        std::snprintf(name, sizeof(name), "cap%d", i);
        SCIP_CALL( cap.post(name, -std::numeric_limits<double>::infinity(), 0.0) );
      }
    }
  }
  result.binaries = binaries;

  /** The property with the margin `s`, maximized. */
  SCIP_VAR* s = nullptr;
  SCIP_CALL( make_var(scip, &s, "s", 0.0, 1.0, 1.0, SCIP_VARTYPE_CONTINUOUS) );
  int props = 0;
  for(int i = n0; i < N; ++i) {
    if(std::isfinite(RLB[i])) {
      Row r(scip); r.add(y[i], 1.0); r.add(s, -1.0);
      std::snprintf(name, sizeof(name), "plb%d", i);
      SCIP_CALL( r.post(name, RLB[i], std::numeric_limits<double>::infinity()) );
      ++props;
    }
    if(std::isfinite(RUB[i])) {
      Row r(scip); r.add(y[i], 1.0); r.add(s, 1.0);
      std::snprintf(name, sizeof(name), "pub%d", i);
      SCIP_CALL( r.post(name, -std::numeric_limits<double>::infinity(), RUB[i]) );
      ++props;
    }
  }
  for(size_t q = 0; q < RLEQ.size(); ++q) {
    const Leq& c = RLEQ[q];
    Row r(scip); r.add(y[c.x], 1.0); r.add(y[c.y], -1.0); r.add(s, 1.0);
    std::snprintf(name, sizeof(name), "leq%zu", q);
    SCIP_CALL( r.post(name, -std::numeric_limits<double>::infinity(), c.k) );
    ++props;
  }
  (void)props;

  if(cancel.load()) { result.status = CANCELLED; return SCIP_OKAY; }
  SCIP_CALL( SCIPsetRealParam(scip, "limits/time", opt.time_limit) );
  SCIP_CALL( SCIPsolve(scip) );
  const SCIP_STATUS status = SCIPgetStatus(scip);

  SCIP_SOL* sol = SCIPgetNSols(scip) > 0 ? SCIPgetBestSol(scip) : nullptr;
  bool exact_found = false;
  if(sol != nullptr && opt.exact_cex && SCIPsolIsExact(sol) && relaxed == 0) {
    /** `-scip_cex exact`: the exact solution is the counterexample, if it lies in the inner region (and the
     * MIP relaxed no ReLU, so that its values are those of the network; otherwise, the strict test). */
    result.margin = SCIPgetSolVal(scip, sol, s);
    SCIP_RATIONAL* v = nullptr;
    SCIP_RATIONAL* w = nullptr;
    SCIP_CALL( SCIPrationalCreateBuffer(SCIPbuffer(scip), &v) );
    SCIP_CALL( SCIPrationalCreateBuffer(SCIPbuffer(scip), &w) );
    bool ok = true;
    std::vector<double> lo(N), hi(N);
    std::string ex;
    for(int i = 0; i < N && ok; ++i) {
      if(y[i] == nullptr) { ok = false; break; }
      SCIPgetSolValExact(scip, sol, y[i], v);
      lo[i] = SCIPrationalRoundReal(v, SCIP_R_ROUND_DOWNWARDS);
      hi[i] = SCIPrationalRoundReal(v, SCIP_R_ROUND_UPWARDS);
      if(std::isfinite(region.lb[i]) && !SCIPrationalIsGEReal(v, region.lb[i])) { ok = false; }
      if(std::isfinite(region.ub[i]) && !SCIPrationalIsLEReal(v, region.ub[i])) { ok = false; }
      if(i < n0) {
        std::vector<char> buf(static_cast<size_t>(SCIPrationalStrLen(v)) + 2);
        SCIPrationalToString(v, buf.data(), static_cast<int>(buf.size()));
        ex += buf.data();
        ex += '\n';
      }
    }
    for(size_t q = 0; q < region.leqs.size() && ok; ++q) {
      const Leq& c = region.leqs[q];
      SCIPgetSolValExact(scip, sol, y[c.x], v);
      SCIPgetSolValExact(scip, sol, y[c.y], w);
      SCIPrationalDiff(v, v, w);
      if(!SCIPrationalIsLEReal(v, c.k)) { ok = false; }
    }
    SCIPrationalFreeBuffer(SCIPbuffer(scip), &w);
    SCIPrationalFreeBuffer(SCIPbuffer(scip), &v);
    if(ok) {
      result.status = FOUND;
      result.lo = std::move(lo);
      result.hi = std::move(hi);
      result.exact = std::move(ex);
      result.cex_path = CEX_EXACT;
      exact_found = true;
    }
    else { result.cex_path = CEX_OUTSIDE; }
  }
  if(exact_found) {}
  else if(sol != nullptr) {
    if(result.cex_path == CEX_NONE) {
      result.cex_path = (opt.exact_cex && SCIPsolIsExact(sol) && relaxed > 0) ? CEX_RELAXED : CEX_FLOAT;
    }
    result.margin = SCIPgetSolVal(scip, sol, s);
    SCIP_RATIONAL* v = nullptr;
    SCIP_CALL( SCIPrationalCreateBuffer(SCIPbuffer(scip), &v) );
    std::vector<double> x(n0);
    for(int k = 0; k < n0; ++k) {
      double r;
      if(SCIPsolIsExact(sol)) {
        SCIPgetSolValExact(scip, sol, y[k], v);
        r = SCIPrationalRoundReal(v, SCIP_R_ROUND_NEAREST);
      }
      else {
        r = SCIPgetSolVal(scip, sol, y[k]);
      }
      x[k] = nearest_float_in(r, ylo[k], yhi[k]);
    }
    SCIPrationalFreeBuffer(SCIPbuffer(scip), &v);
    if(strict_check(net, region, x, result.lo, result.hi)) {
      result.status = FOUND;
    }
    else {
      result.status = UNCERTIFIED;
      result.lo.clear();
      result.hi.clear();
    }
  }
  else if(status == SCIP_STATUS_TIMELIMIT) {
    result.status = LIMIT;
  }
  else if(status == SCIP_STATUS_USERINTERRUPT || status == SCIP_STATUS_TERMINATE) {
    result.status = CANCELLED;
  }
  else if(prove) {
    /** Proof mode: only a proved infeasibility closes the node (any other end, e.g. a memory limit, decides
     * nothing). */
    result.status = status == SCIP_STATUS_INFEASIBLE ? CLOSED : LIMIT;
  }
  else {
    /** Optimal below the margin, infeasible: no counterexample of the margin in the node. */
    result.status = NONE;
  }
  for(SCIP_VAR*& v : y) { if(v != nullptr) { SCIP_CALL( SCIPreleaseVar(scip, &v) ); } }
  for(SCIP_VAR*& v : extra) { SCIP_CALL( SCIPreleaseVar(scip, &v) ); }
  SCIP_CALL( SCIPreleaseVar(scip, &s) );
  return SCIP_OKAY;
}

} // namespace detail

/** Searches the node of bounds `nlb`, `nub` (all the neurons) of the disjunct of region `region` for a
 * counterexample with SCIP in exact mode, see the top of the file. Thread-safe: one SCIP per call. */
template <class Net>
Result solve_node(const Net& net, const Region& region, const float* nlb, const float* nub,
  const Options& opt, const std::atomic<bool>& cancel)
{
  Result result;
  const auto t0 = std::chrono::steady_clock::now();
  std::string why;
  if(!supported(net, why)) { result.status = UNSUPPORTED; return result; }
  SCIP* scip = nullptr;
  if(SCIPcreate(&scip) != SCIP_OKAY) { result.status = FAILED; return result; }
  const SCIP_RETCODE rc = detail::build_and_solve(scip, net, region, nlb, nub, opt, cancel, result);
  if(rc != SCIP_OKAY) { result.status = FAILED; }
  SCIPfree(&scip);
  result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return result;
}

/** The messages between the `Manager` and its worker processes (`worker_main`), on a socket. All the
 * processes run on the same machine, so the values are sent in their memory representation. */
namespace wire {

inline bool write_all(int fd, const void* data, size_t n) {
  const char* p = static_cast<const char*>(data);
  while(n > 0) {
    const ssize_t k = ::send(fd, p, n, MSG_NOSIGNAL);
    if(k < 0 && errno == EINTR) { continue; }
    if(k <= 0) { return false; }
    p += k; n -= static_cast<size_t>(k);
  }
  return true;
}

inline bool read_all(int fd, void* data, size_t n) {
  char* p = static_cast<char*>(data);
  while(n > 0) {
    const ssize_t k = ::recv(fd, p, n, 0);
    if(k < 0 && errno == EINTR) { continue; }
    if(k <= 0) { return false; }
    p += k; n -= static_cast<size_t>(k);
  }
  return true;
}

template <class T> bool put(int fd, const T& v) { return write_all(fd, &v, sizeof(T)); }
template <class T> bool get(int fd, T& v) { return read_all(fd, &v, sizeof(T)); }
template <class T> bool put_vec(int fd, const std::vector<T>& v) {
  const long long n = static_cast<long long>(v.size());
  return put(fd, n) && (n == 0 || write_all(fd, v.data(), sizeof(T) * v.size()));
}
template <class T> bool get_vec(int fd, std::vector<T>& v) {
  long long n = 0;
  if(!get(fd, n) || n < 0) { return false; }
  v.resize(static_cast<size_t>(n));
  return n == 0 || read_all(fd, v.data(), sizeof(T) * v.size());
}

inline bool put_net(int fd, const HostNet& net, const Options& opt) {
  return put(fd, 'N') && put(fd, net.num_neurons) && put(fd, net.output_layer) && put_vec(fd, net.layers)
    && put_vec(fd, net.weights) && put_vec(fd, net.biases) && put_vec(fd, net.sp_row) && put_vec(fd, net.sp_col)
    && put_vec(fd, net.sp_val) && put(fd, opt);
}
inline bool put_region(int fd, int dnf, const Region& r) {
  return put(fd, 'R') && put(fd, dnf) && put_vec(fd, r.lb) && put_vec(fd, r.ub) && put_vec(fd, r.leqs)
    && put_vec(fd, r.olb) && put_vec(fd, r.oub) && put_vec(fd, r.oleqs);
}
inline bool put_job(int fd, const Job& j) {
  return put(fd, 'J') && put(fd, j.dnf) && put(fd, j.seq) && put_vec(fd, j.lb) && put_vec(fd, j.ub);
}
inline bool put_result(int fd, const Result& r) {
  const int status = static_cast<int>(r.status);
  return put(fd, status) && put(fd, r.cex_path) && put(fd, r.margin) && put(fd, r.binaries) && put(fd, r.seconds)
    && put_vec(fd, r.lo) && put_vec(fd, r.hi) && put_vec(fd, std::vector<char>(r.exact.begin(), r.exact.end()));
}
inline bool get_result(int fd, Result& r) {
  int status = 0;
  if(!(get(fd, status) && get(fd, r.cex_path) && get(fd, r.margin) && get(fd, r.binaries) && get(fd, r.seconds)
    && get_vec(fd, r.lo) && get_vec(fd, r.hi))) { return false; }
  std::vector<char> ex;
  if(!get_vec(fd, ex)) { return false; }
  r.exact.assign(ex.begin(), ex.end());
  r.status = static_cast<Status>(status);
  return true;
}

} // namespace wire

/** The argument that turns the executable into a worker of `-under scip` (see `worker_main`). */
inline const char* worker_flag() { return "--jet-scip-worker"; }

/** The loop of a worker process, on the socket `fd`: it receives the network, the regions and the
 * nodes, solves each node with `solve_node`, and answers with its result. It ends when the socket is
 * closed (the manager ended, or killed the worker). The executables that run a `Manager` (`turbo`, the
 * tests) call it first thing in `main` when their first argument is `worker_flag()`. */
inline int worker_main(int fd = 3) {
  HostNet net;
  Options opt;
  std::map<int, Region> regions;
  const std::atomic<bool> no_cancel{false};
  while(true) {
    char kind = 0;
    if(!wire::get(fd, kind)) { return 0; }
    if(kind == 'N') {
      if(!(wire::get(fd, net.num_neurons) && wire::get(fd, net.output_layer) && wire::get_vec(fd, net.layers)
        && wire::get_vec(fd, net.weights) && wire::get_vec(fd, net.biases) && wire::get_vec(fd, net.sp_row)
        && wire::get_vec(fd, net.sp_col) && wire::get_vec(fd, net.sp_val) && wire::get(fd, opt))) { return 1; }
    }
    else if(kind == 'R') {
      int dnf = 0;
      Region r;
      if(!(wire::get(fd, dnf) && wire::get_vec(fd, r.lb) && wire::get_vec(fd, r.ub) && wire::get_vec(fd, r.leqs)
        && wire::get_vec(fd, r.olb) && wire::get_vec(fd, r.oub) && wire::get_vec(fd, r.oleqs))) { return 1; }
      regions[dnf] = std::move(r);
    }
    else if(kind == 'J') {
      Job j;
      if(!(wire::get(fd, j.dnf) && wire::get(fd, j.seq) && wire::get_vec(fd, j.lb) && wire::get_vec(fd, j.ub))) { return 1; }
      Result r;
      auto it = regions.find(j.dnf);
      if(it == regions.end() || static_cast<int>(j.lb.size()) != net.num_neurons) { r.status = FAILED; }
      else { r = solve_node(net, it->second, j.lb.data(), j.ub.data(), opt, no_cancel); }
      if(!wire::put_result(fd, r)) { return 1; }
    }
    else { return 1; }
  }
}

/** The path of the running executable (Linux). */
inline std::string self_executable() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if(n <= 0) { return std::string(); }
  buf[n] = '\0';
  return std::string(buf);
}

/** The workers of `-under scip`. Each worker is a process (the executable itself, started with
 * `worker_flag()`, running `worker_main`), driven by a thread of the manager: SCIP in exact mode is not
 * thread-safe (the LU factorization of its project-and-shift method has global state), and a process
 * can be stopped at any time by killing it. `submit` queues a node; the workers take the nodes in the
 * order of their sequence number; the first counterexample found is kept (`found`), calls `on_found`
 * (which stops the GPU) and cancels the other nodes. `cancel` drops the pending nodes and kills the
 * workers: a manager is used for one solving. */
class Manager {
public:
  struct Stats {
    std::atomic<unsigned long long> submitted{0}, solved{0};
    std::atomic<unsigned long long> by_status[kNumStatus] = {};
    std::atomic<unsigned long long> binaries{0};
    /** The solutions of SCIP by `CexPath`, and those of them that were counterexamples (`FOUND`). */
    std::atomic<unsigned long long> cex_paths[kNumCexPaths] = {};
    std::atomic<unsigned long long> cex_found[kNumCexPaths] = {};
    std::atomic<unsigned long long> max_pending{0};
    std::atomic<long long> micros{0};
  };

  /** Starts `workers` worker processes of the executable `exe` (default: this one). */
  Manager(const HostNet& net, int workers, Options opt, std::function<void()> on_found = {},
    std::string exe = self_executable())
   : net_(net), opt_(opt), on_found_(std::move(on_found))
  {
    for(int t = 0; t < std::max(1, workers); ++t) {
      auto w = std::make_unique<Worker>();
      if(!spawn(*w, exe)) { break; }
      workers_.push_back(std::move(w));
    }
    for(auto& w : workers_) {
      Worker* p = w.get();
      p->thread = std::thread([this, p]() { work(*p); });
    }
  }

  ~Manager() { shutdown(); }

  /** The number of worker processes started. */
  int num_workers() const { return static_cast<int>(workers_.size()); }

  /** The region of the disjunct `dnf` of the DNF (once; later calls are ignored). */
  void add_region(int dnf, const Region& r) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(regions_.find(dnf) == regions_.end()) { regions_.emplace(dnf, r); }
  }

  /** Queues a node. Ignored once a counterexample is found or after `cancel`. */
  void submit(Job&& job) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if(cancel_) { return; }
      pending_.push(std::move(job));
      stats_.submitted++;
      const unsigned long long p = pending_.size();
      if(p > stats_.max_pending.load()) { stats_.max_pending = p; }
    }
    cv_.notify_one();
  }

  /** Drops the pending nodes and kills the workers (their running nodes are lost). */
  /** `-scip_mode race`: `f(seq, status)` is called with the answer of every node solved (by the thread of
   * its worker), and a node is skipped (`CANCELLED`) when `skip(seq)` holds when a worker takes it. To be
   * set before the first `submit`. */
  void set_on_result(std::function<void(unsigned long long, Status)> f) { on_result_ = std::move(f); }
  void set_skip(std::function<bool(unsigned long long)> f) { skip_ = std::move(f); }

  void cancel() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if(cancel_) { return; }
      cancel_ = true;
      while(!pending_.empty()) { pending_.pop(); }
    }
    for(auto& w : workers_) {
      if(w->pid > 0) { ::kill(w->pid, SIGKILL); }
    }
    cv_.notify_all();
    idle_cv_.notify_all();
  }

  /** The number of nodes waiting for a worker. */
  size_t pending() {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.size();
  }

  /** Whether no node is pending or running. */
  bool idle() {
    std::lock_guard<std::mutex> lock(mutex_);
    return (pending_.empty() || alive_ == 0) && running_ == 0;
  }

  /** Waits until the workers are idle, a counterexample is found, or `keep_waiting` returns false
   * (checked every 50 ms). */
  void wait_idle(const std::function<bool()>& keep_waiting) {
    std::unique_lock<std::mutex> lock(mutex_);
    while(!((pending_.empty() || alive_ == 0) && running_ == 0) && !found_ && keep_waiting()) {
      idle_cv_.wait_for(lock, std::chrono::milliseconds(50));
    }
  }

  bool found() {
    std::lock_guard<std::mutex> lock(mutex_);
    return found_;
  }

  /** The counterexample: its disjunct and the evaluation of its point (valid when `found()`). */
  int found_dnf() { std::lock_guard<std::mutex> lock(mutex_); return found_dnf_; }
  std::vector<double> found_lo() { std::lock_guard<std::mutex> lock(mutex_); return found_lo_; }
  /** The exact point of the counterexample (`Result::exact`), empty if it was checked in floats. */
  std::string found_exact() { std::lock_guard<std::mutex> lock(mutex_); return found_exact_; }
  /** How the counterexample was taken (`CexPath`). */
  int found_path() { std::lock_guard<std::mutex> lock(mutex_); return found_path_; }
  std::vector<double> found_hi() { std::lock_guard<std::mutex> lock(mutex_); return found_hi_; }

  const Stats& stats() const { return stats_; }

  /** Cancels, then waits for the threads and the processes. */
  void shutdown() {
    cancel();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for(auto& w : workers_) {
      if(w->thread.joinable()) { w->thread.join(); }
      if(w->fd >= 0) { ::close(w->fd); w->fd = -1; }
      if(w->pid > 0) { int st; ::waitpid(w->pid, &st, 0); w->pid = -1; }
    }
  }

  void print_statistics() const {
    printf("%%%%%%nnv-stat: scip_workers=%d\n", num_workers());
    printf("%%%%%%nnv-stat: scip_nodes_received=%llu\n", stats_.submitted.load());
    printf("%%%%%%nnv-stat: scip_nodes_solved=%llu\n", stats_.solved.load());
    for(int s = 0; s < kNumStatus; ++s) {
      printf("%%%%%%nnv-stat: scip_nodes_%s=%llu\n", name_of_status(static_cast<Status>(s)), stats_.by_status[s].load());
    }
    printf("%%%%%%nnv-stat: scip_binaries=%llu\n", stats_.binaries.load());
    for(int p = CEX_EXACT; p < kNumCexPaths; ++p) {
      printf("%%%%%%nnv-stat: scip_solutions_%s=%llu\n", name_of_cex_path(p), stats_.cex_paths[p].load());
      printf("%%%%%%nnv-stat: scip_counterexamples_%s=%llu\n", name_of_cex_path(p), stats_.cex_found[p].load());
    }
    printf("%%%%%%nnv-stat: scip_found_by=\"%s\"\n", found_ ? name_of_cex_path(found_path_) : "none");
    printf("%%%%%%nnv-stat: scip_max_pending=%llu\n", stats_.max_pending.load());
    printf("%%%%%%nnv-stat: scip_time_ms=%lld\n", stats_.micros.load() / 1000);
  }

private:
  struct Worker {
    pid_t pid = -1;
    int fd = -1;
    std::set<int> regions;
    std::thread thread;
  };

  struct Later {
    bool operator()(const Job& a, const Job& b) const { return a.seq > b.seq; }
  };

  /** Starts a worker process with the socket on its descriptor 3, and sends it the network. */
  bool spawn(Worker& w, const std::string& exe) {
    if(exe.empty()) { return false; }
    int sv[2];
    if(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) != 0) { return false; }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, sv[1], 3);
    std::string flag = worker_flag();
    char* argv[] = {const_cast<char*>(exe.c_str()), const_cast<char*>(flag.c_str()), nullptr};
    pid_t pid = -1;
    const int rc = ::posix_spawn(&pid, exe.c_str(), &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    ::close(sv[1]);
    if(rc != 0) { ::close(sv[0]); return false; }
    w.pid = pid;
    w.fd = sv[0];
    if(!wire::put_net(w.fd, net_, opt_)) {
      ::kill(pid, SIGKILL);
      int st; ::waitpid(pid, &st, 0);
      ::close(w.fd);
      w.pid = -1; w.fd = -1;
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++alive_;
    }
    return true;
  }

  void work(Worker& w) {
    while(true) {
      Job job;
      Region region;
      bool send_region = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return stop_ || cancel_ || !pending_.empty(); });
        if(stop_ || cancel_) { break; }
        job = std::move(const_cast<Job&>(pending_.top()));
        pending_.pop();
        if(skip_ && skip_(job.seq)) {
          stats_.by_status[static_cast<int>(CANCELLED)]++;
          continue;
        }
        auto it = regions_.find(job.dnf);
        if(it == regions_.end()) { continue; }
        if(w.regions.count(job.dnf) == 0) { region = it->second; send_region = true; }
        ++running_;
      }
      Result r;
      bool ok = (!send_region || wire::put_region(w.fd, job.dnf, region)) && wire::put_job(w.fd, job);
      if(ok && send_region) { w.regions.insert(job.dnf); }
      ok = ok && wire::get_result(w.fd, r);
      bool first = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        --running_;
        if(ok) {
          stats_.solved++;
          stats_.by_status[static_cast<int>(r.status)]++;
          stats_.binaries += static_cast<unsigned long long>(r.binaries);
          if(r.cex_path > CEX_NONE && r.cex_path < kNumCexPaths) {
            stats_.cex_paths[r.cex_path]++;
            if(r.status == FOUND) { stats_.cex_found[r.cex_path]++; }
          }
          stats_.micros += static_cast<long long>(r.seconds * 1e6);
          if(r.status == FOUND && !found_ && !r.lo.empty()) {
            found_ = true;
            first = true;
            found_dnf_ = job.dnf;
            found_lo_ = r.lo;
            found_hi_ = r.hi;
            found_exact_ = r.exact;
            found_path_ = r.cex_path;
          }
        }
        else if(!cancel_) {
          /** The worker died (or its socket failed): its node is lost, counted as failed. */
          stats_.by_status[static_cast<int>(FAILED)]++;
        }
      }
      if(ok && on_result_) { on_result_(job.seq, r.status); }
      if(first) {
        if(on_found_) { on_found_(); }
        cancel();
      }
      idle_cv_.notify_all();
      if(!ok) { break; }
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      --alive_;
    }
    idle_cv_.notify_all();
  }

  HostNet net_;
  Options opt_;
  std::function<void()> on_found_;
  std::function<void(unsigned long long, Status)> on_result_;
  std::function<bool(unsigned long long)> skip_;
  std::mutex mutex_;
  std::condition_variable cv_, idle_cv_;
  std::priority_queue<Job, std::vector<Job>, Later> pending_;
  std::map<int, Region> regions_;
  bool cancel_ = false;
  bool stop_ = false;
  int running_ = 0;
  int alive_ = 0;
  bool found_ = false;
  int found_dnf_ = -1;
  std::vector<double> found_lo_, found_hi_;
  std::string found_exact_;
  int found_path_ = CEX_NONE;
  Stats stats_;
  std::vector<std::unique_ptr<Worker>> workers_;
};

#endif // JET_WITH_SCIP

} // namespace scip_under
} // namespace jet

#endif
