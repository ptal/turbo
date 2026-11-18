// Copyright 2022 Pierre Talbot

#ifndef TURBO_CONFIG_HPP
#define TURBO_CONFIG_HPP

#include "battery/allocator.hpp"
#include "battery/string.hpp"
#include <cinttypes>
#include <optional>

#ifdef __CUDACC__
  #include <cuda.h>
#endif

enum class Arch {
  CPU,
  GPU,
  BAREBONES,
  FBAREBONES,
  HYBRID,
  JET
};

enum class FixpointKind {
  AC1,
  WAC1
};

/** Values of `Configuration::split_on` (`-split_on`), `Configuration::relu_order` (`-relu_order`) and
 * `Configuration::relu_leaf` (`-relu_leaf`). */
constexpr int SPLIT_ON_INPUT = 0;
constexpr int SPLIT_ON_RELU = 1;
/** JET only: the inputs and the ReLUs together, chosen by `-jet_var_select smear|abs` (`jet_abs.hpp`). */
constexpr int SPLIT_ON_MIXED = 2;
constexpr int RELU_ORDER_EARLIEST = 0;
constexpr int RELU_ORDER_WIDEST = 1;
/** JET only: the ReLU of maximum score (`-relu_score`), and the adaptive bisection of Granvilliers (CP 2012)
 * over the ReLUs (`jet_abs.hpp`). */
constexpr int RELU_ORDER_SMEAR = 2;
constexpr int RELU_ORDER_ABS = 3;
/** Values of `Configuration::relu_score` (`-relu_score`): the score of a ReLU of unknown phase for
 * `-relu_order smear|abs` and `-split_on mixed`, `g * m` with `g` the largest derivative of a row of the
 * property with respect to its output `y`: `RELU_SCORE_SMEAR`: `m` is the width of `y` (the smear of the
 * variable `y`); `RELU_SCORE_GAP`: `m` is the largest gap of the relaxation of the ReLU at the node, the
 * width of `y` when the node ran no linear relaxation, `-l u / (u - l)` (the chord of DeepPoly and CROWN,
 * `l < 0 < u` the bounds of the pre-activation) when it ran the symbolic pass or the backward contractor. */
constexpr int RELU_SCORE_SMEAR = 0;
constexpr int RELU_SCORE_GAP = 1;
CUDA inline const char* name_of_split_on(int v) { return v == SPLIT_ON_RELU ? "relu" : (v == SPLIT_ON_MIXED ? "mixed" : "input"); }
CUDA inline const char* name_of_relu_order(int v) {
  return v == RELU_ORDER_WIDEST ? "widest" : (v == RELU_ORDER_SMEAR ? "smear" : (v == RELU_ORDER_ABS ? "abs" : "earliest"));
}
constexpr int RELU_LEAF_INPUT = 0;
constexpr int RELU_LEAF_POINT = 1;
/** Values of `Configuration::jet_interval_prop` (`-jet_interval_prop`). */
constexpr int JET_INTERVAL_FIXPOINT = 0;
constexpr int JET_INTERVAL_FORWARD = 1;
constexpr int JET_INTERVAL_FORWARD_BACKWARD = 2;
/** Values of `Configuration::jet_leaf_check` (`-jet_leaf_check`). */
constexpr int JET_LEAF_STRICT = 0;
constexpr int JET_LEAF_LENIENT = 1;
/** Values of `Configuration::jet_disjuncts` (`-jet_disjuncts`). */
constexpr int JET_DISJUNCTS_INTERLEAVED = 0;
constexpr int JET_DISJUNCTS_SEQUENTIAL = 1;
/** Values of `Configuration::jet_backward` (`-jet_backward`, `jet_backward.hpp`). */
constexpr int JET_BACKWARD_OFF = 0;
constexpr int JET_BACKWARD_ROOT = 1;
constexpr int JET_BACKWARD_SPEC = 2;
constexpr int JET_BACKWARD_ALL = 3;
/** Values of `Configuration::jet_linear` (`-jet_linear`). */
constexpr int JET_LINEAR_BOTH = 0;
constexpr int JET_LINEAR_FORWARD = 1;
constexpr int JET_LINEAR_BACKWARD = 2;
constexpr int JET_LINEAR_AUTO = 3;
CUDA inline const char* name_of_jet_backward(int b) {
  return b == JET_BACKWARD_ROOT ? "root" : (b == JET_BACKWARD_SPEC ? "spec" : (b == JET_BACKWARD_ALL ? "all" : "off"));
}
CUDA inline const char* name_of_jet_linear(int v) {
  return v == JET_LINEAR_FORWARD ? "forward" : (v == JET_LINEAR_BACKWARD ? "backward" : (v == JET_LINEAR_AUTO ? "auto" : "both"));
}
/** Values of `Configuration::under` (`-under`, `jet_scip.hpp`): how the counterexamples of the nodes are
 * looked for. `UNDER_JET`: the underapproximation of JET (`-jet_candidates`, the strict test of the
 * leaves). `UNDER_SCIP`: the nodes with few unstable ReLUs are sent to SCIP in exact mode. */
constexpr int UNDER_JET = 0;
constexpr int UNDER_SCIP = 1;
CUDA inline const char* name_of_under(int u) { return u == UNDER_SCIP ? "scip" : "jet"; }
/** Values of `Configuration::scip_mode` (`-scip_mode`, with `-under scip`): `SCIP_MODE_OBSERVE`, SCIP only
 * looks for counterexamples and its other answers prune nothing; `SCIP_MODE_RACE`, SCIP also proves nodes
 * (the MIP of the proof mode of `jet_scip.hpp`): a node proved without counterexample ends the search of
 * its subtree by the GPU, and the unknown leaves under it are not counted. */
constexpr int SCIP_MODE_OBSERVE = 0;
constexpr int SCIP_MODE_RACE = 1;
CUDA inline const char* name_of_scip_mode(int m) { return m == SCIP_MODE_RACE ? "race" : "observe"; }
/** Values of `Configuration::jet_var_select` (`-jet_var_select`, `jet_abs.hpp`): how JET chooses the input
 * to bisect. `JET_VAR_DEFAULT`: the variable order of the strategy (`-var_order`, as before);
 * `JET_VAR_MAXDOM`: the widest input; `JET_VAR_RR`: round-robin, the input bisected the fewest times on
 * the current branch; `JET_VAR_SMEAR`: the maximum smear (Kearfott and Novoa, Algorithm 681 INTBIS);
 * `JET_VAR_ABS`: the adaptive bisection of Granvilliers (CP 2012), a restricted candidate list of the
 * inputs of high smear, and round-robin in it. */
constexpr int JET_VAR_DEFAULT = 0;
constexpr int JET_VAR_MAXDOM = 1;
constexpr int JET_VAR_RR = 2;
constexpr int JET_VAR_SMEAR = 3;
constexpr int JET_VAR_ABS = 4;
CUDA inline const char* name_of_jet_var_select(int v) {
  return v == JET_VAR_MAXDOM ? "maxdom" : (v == JET_VAR_RR ? "rr" : (v == JET_VAR_SMEAR ? "smear" : (v == JET_VAR_ABS ? "abs" : "default")));
}
/** Values of `Configuration::abs_psi` (`-abs_psi`): the scalar of the smear values that sets the size of
 * the candidate list of ABS. `ABS_PSI_STD`: their standard deviation (the paper); `ABS_PSI_CV`: their
 * coefficient of variation (standard deviation over mean), which does not depend on the scale of the
 * outputs of the network. */
constexpr int ABS_PSI_STD = 0;
constexpr int ABS_PSI_CV = 1;
/** Values of `Configuration::abs_grad` (`-abs_grad`): the derivatives of the smear. `ABS_GRAD_INTERVAL`:
 * the interval Jacobian of the comparisons of the property with respect to the inputs, on the box of the
 * node (`jet_jacobian.hpp`), as in the paper; `ABS_GRAD_RELAX`: the coefficients of the symbolic forms of
 * the node (`-jet_symbolic`), the interval Jacobian when the node has no forms. */
constexpr int ABS_GRAD_INTERVAL = 0;
constexpr int ABS_GRAD_RELAX = 1;
/** The name of a value of `Configuration::jet_interval_prop`, as `-jet_interval_prop` reads it. */
CUDA inline const char* name_of_jet_interval_prop(int p) {
  return p == JET_INTERVAL_FORWARD ? "forward" : (p == JET_INTERVAL_FORWARD_BACKWARD ? "forward_backward" : "fixpoint");
}

enum class InputFormat {
  XCSP3,
  FLATZINC,
  VNNLIB,
  ONNX,
  SMT2
};

template<class Allocator>
struct Configuration {
  using allocator_type = Allocator;
  bool print_intermediate_solutions; // (only optimization problems).
  size_t stop_after_n_solutions; // 0 for all solutions (satisfaction problems only).
  size_t stop_after_n_nodes; // size_t MAX values for all nodes.
  bool free_search;
  bool print_statistics;
  int verbose_solving;
  bool print_ast;
  bool only_global_memory;
  bool force_ternarize;
  bool disable_simplify;
  bool disable_network_analysis;
  size_t timeout_ms;
  size_t or_nodes;
  int subproblems_power;
  size_t subproblems_factor;
  size_t stack_kb;
  Arch arch;
  FixpointKind fixpoint;
  size_t wac1_threshold;
  size_t seed;
  double epsilon;
  battery::string<allocator_type> var_order;
  battery::string<allocator_type> value_order;
  battery::string<allocator_type> eps_var_order;
  battery::string<allocator_type> eps_value_order;
  battery::string<allocator_type> problem_path;
  battery::string<allocator_type> vnnlib_path;
  battery::string<allocator_type> onnx_path;
  battery::string<allocator_type> version;
  battery::string<allocator_type> hardware;
  /** JET only: version of the deduction of the affine neurons (1: original, 2: `jet_affine_v2.hpp`). */
  int jet_affine;
  /** JET only: alternate the interval fixpoint with symbolic rounds (`jet_symbolic.hpp`); fbarebones
   * ignores it with a warning. */
  bool jet_symbolic;
  /** JET only: maximum number of symbolic rounds per node, 0 for "until nothing changes". */
  int jet_sym_rounds;
  /** JET only: memory layout of the symbolic forms (0: neuron-major, 1: slot-major, coalesced). */
  int jet_sym_layout;
  /** fbarebones only: deduction of the bytecodes (1: `PIR::fdeduce`, 2: `PIR::fdeduce_v2`, bitwise the same results). */
  int pir_fdeduce;
  /** JET and fbarebones: the variables the search branches on (`SPLIT_ON_INPUT`: the input neurons,
   * `SPLIT_ON_RELU`: the phase of the ReLU neurons, see `relu_split.hpp`; JET only: `SPLIT_ON_MIXED`, both,
   * see `jet_abs.hpp`). */
  int split_on;
  /** JET and fbarebones, with `split_on == SPLIT_ON_RELU`: which ReLU is split first
   * (`RELU_ORDER_EARLIEST`: the first one in the order of the network, `RELU_ORDER_WIDEST`: the one
   * with the widest pre-activation; JET only: `RELU_ORDER_SMEAR` the one of maximum score
   * (`relu_score`), `RELU_ORDER_ABS` the adaptive bisection of Granvilliers over the ReLUs, with `abs_beta`,
   * `abs_psi` and `abs_grad`, see `jet_abs.hpp`). */
  int relu_order;
  /** JET and fbarebones, with `split_on == SPLIT_ON_RELU`: the search once every ReLU has a phase
   * (`RELU_LEAF_INPUT`: the input split, down to boxes of width at most `epsilon` and their
   * underapproximation, `RELU_LEAF_POINT`: the inputs fixed to points at once), see `relu_split.hpp`. */
  int relu_leaf;
  /** JET only, with `jet_symbolic`: also clip the inputs with the forms of the hidden neurons
   * (`jet_symbolic.hpp`, `clip_hidden`). */
  bool jet_sym_clip_hidden;
  /** JET only, with `jet_symbolic`: the symbolic rounds are done on the root node only, once before the
   * search (`gpu_root_symbolic`); the nodes of the search run the interval fixpoint only. */
  bool jet_sym_root_only;
  /** JET only, with `jet_symbolic` and `jet_sym_rounds > 0`: no interval propagation after the last
   * symbolic round (`node_fixpoint`), e.g. with `jet_sym_rounds = 1`, a node is the interval
   * propagation, then one symbolic round (forms and clipping), and nothing more. */
  bool jet_sym_skip_final_interval;
  /** JET only (`-jet_candidates`): test candidate points of the box of a node as counterexamples, with a
   * strict test, before the search goes on (`jet_candidates.hpp`): the middle of the box, and with
   * `jet_symbolic` the corners chosen by the forms of the node. */
  bool jet_candidates;
  /** JET only (`-jet_cand_every k`): the candidates are tested at the nodes of depth multiple of `k`
   * (`k = 0`: at depth 0 only, that is the nodes of the dive to each subproblem). */
  int jet_cand_every;
  /** JET only (`-jet_disjuncts`): how the disjuncts of a disjunctive property are searched.
   * `JET_DISJUNCTS_INTERLEAVED` (default): one search, the subproblems of all the disjuncts taken in
   * turn by the blocks, so that every disjunct progresses from the start. `JET_DISJUNCTS_SEQUENTIAL`:
   * one search per disjunct, each one finished before the next one starts. */
  int jet_disjuncts;
  /** JET only (`-jet_leaf_check`): how a leaf of the search (the variables of the strategy are points
   * and the propagation is not empty) is decided. `JET_LEAF_STRICT` (default): its point (the middle
   * of the node within the vnnlib bounds rounded inward) is a solution only if it passes the strict
   * test of `jet_candidates.hpp` (the network evaluated in `double` rounded outward satisfies every
   * constraint of the region); otherwise the leaf is counted unknown. `JET_LEAF_LENIENT`: the leaf is
   * a solution as soon as its propagation is not empty, which only says that the outward-rounded
   * intervals meet the region (the behaviour before `-jet_leaf_check`). */
  int jet_leaf_check;
  /** JET only (`-jet_backward`): the backward contractor (`jet_backward.hpp`), back-substitution of
   * linear rows through the ReLU and sigmoid relaxations (CROWN). `JET_BACKWARD_OFF` (default): not
   * used. `JET_BACKWARD_ROOT`: on the root of every disjunct, once, before the search (the rows of the
   * unstable neurons, of the outputs and of the comparisons), alternated with the interval fixpoint.
   * `JET_BACKWARD_SPEC`: as `ROOT`, then in every node the rows of the outputs and of the comparisons
   * only (the bounds of the hidden neurons come from the root and the interval propagation).
   * `JET_BACKWARD_ALL`: as `SPEC`, and every row in the nodes of depth 0 or multiple of
   * `jet_bwd_every`. */
  int jet_backward;
  /** JET only (`-jet_linear`): which linear-relaxation contractors may run: `BOTH` (default; the forward
   * symbolic pass with `jet_symbolic`, the backward contractor with `jet_backward`), `FORWARD` (the
   * backward contractor is disabled), `BACKWARD` (the symbolic pass is disabled), `AUTO` (the symbolic
   * pass when the network has at most `jet_linear_n0` inputs, the backward contractor otherwise). */
  int jet_linear;
  int jet_linear_n0;
  /** JET only, with `jet_backward`: the number of rows back-substituted together (`-jet_bwd_rows`,
   * even, default 32), the depth period of the full passes of `ALL` (`-jet_bwd_every`, 0: depth 0
   * only), the maximal number of backward passes in the fixpoint of a node (`-jet_bwd_rounds`, default
   * 4), the clipping of the inputs with the rows (`-jet_bwd_clip`), and the early termination of the
   * rows of the neurons once their phase is decided (`-jet_bwd_et`). */
  int jet_bwd_rows;
  int jet_bwd_every;
  int jet_bwd_rounds;
  bool jet_bwd_clip;
  bool jet_bwd_et;
  /** JET and fbarebones: forward propagation only. The deductions of the neurons project from a layer
   * onto the next ones and never tighten the neurons they read (`affine_v2::deduce`, `deduce_neuron`,
   * `deduce_add`; the directed bytecodes of `fbarebones_forward.hpp`), and the symbolic rounds clip
   * nothing (`symbolic::pass`). With JET, it implies `jet_interval_prop` forward. */
  bool forward_only;
  /** JET only: the interval propagation (`-jet_interval_prop`). `JET_INTERVAL_FIXPOINT`: the deductions
   * of the neurons project forward and backward. `JET_INTERVAL_FORWARD`: forward only, as with
   * `forward_only`, but the symbolic rounds of `jet_symbolic` still clip. `JET_INTERVAL_FORWARD_BACKWARD`:
   * no fixpoint, one forward sweep (layers in order, forward projections only), the comparisons of the
   * property once, then one backward sweep (layers in reverse order, forward and backward projections). */
  int jet_interval_prop;
  /** JET only: the underapproximation of the nodes (`-under`, `UNDER_JET` or `UNDER_SCIP`). With
   * `UNDER_SCIP` (`jet_scip.hpp`), the first node of a path with at most `scip_max_unstable` unstable
   * ReLUs (`-scip_max_unstable`, default 12) is copied to a queue, and `scip_threads` workers on the CPU
   * (`-scip_threads`, default 4) search it with SCIP in exact mode, at most `scip_time` seconds per node
   * (`-scip_time`, default 10), for a counterexample of margin at least `scip_margin` (`-scip_margin`,
   * default 1e-6) that passes the strict test. The search of the GPU does not wait for them and prunes
   * nothing with their answers. `scip_queue`: the number of slots of the queue (`-scip_queue`, default
   * 1024; a node is dropped when its slot is busy). */
  int under;
  int scip_threads;
  int scip_max_unstable;
  double scip_time;
  double scip_margin;
  int scip_queue;
  /** JET only, with `-under scip`: `scip_mode` (`-scip_mode`, `SCIP_MODE_*`, default race); the time limit
   * of a node relative to the timeout of the solving `T` (`-t`): `clamp(scip_time_ratio * T, scip_time_min,
   * scip_time_max)` seconds (`-scip_time_ratio`, default 0.02, `-scip_time_min` 0.5, `-scip_time_max` 30),
   * `scip_time` when there is no timeout or the ratio is 0. */
  int scip_mode;
  /** JET only, with `-under scip` (`-scip_cex`): the point of SCIP is a counterexample as it is, exact
   * rational values (`true`, `exact`, the default), or after rounding to floats and the strict test
   * (`false`, `float`). */
  bool scip_cex_exact;
  double scip_time_ratio;
  double scip_time_min;
  double scip_time_max;
  /** JET only, with `-jet_candidates` (`-jet_cand_grad`, default on): in a node without the symbolic forms
   * of the node, the corners are chosen by the signs of the interval Jacobian of their rows
   * (`jet_jacobian.hpp`, the middle of the interval derivative) instead of not being tested. */
  bool jet_cand_grad;
  /** JET only: the choice of the input to bisect (`-jet_var_select`, `JET_VAR_*`, `jet_abs.hpp`), with the
   * input split (`-split_on input`, and below the leaves of `-split_on relu -relu_leaf input`). With
   * `JET_VAR_ABS`: `abs_beta` (`-abs_beta`, default 0.5) is the curvature of `phi(y) = 1 / (1 + beta y)`,
   * `abs_psi` (`-abs_psi`, `ABS_PSI_*`) the scalar of the smear values, and `abs_dive_count`
   * (`-abs_dive_count`) adds to the bisections of the current branch an estimate of those of the dive
   * (`log2` of the width at the root of the disjunct over the current width). `abs_grad` (`-abs_grad`,
   * `ABS_GRAD_*`): the derivatives of the smear (`smear` and `abs`). */
  int jet_var_select;
  double abs_beta;
  int abs_psi;
  int abs_grad;
  bool abs_dive_count;
  /** JET only: the score of a ReLU (`-relu_score`, `RELU_SCORE_*`) for `-relu_order smear|abs` and
   * `-split_on mixed`. */
  int relu_score;

  /** JET: whether the ReLU to split is chosen by its score (`-relu_order smear|abs`, `-split_on mixed`). */
  CUDA bool jet_relu_select() const {
    return (split_on == SPLIT_ON_RELU && (relu_order == RELU_ORDER_SMEAR || relu_order == RELU_ORDER_ABS))
      || split_on == SPLIT_ON_MIXED;
  }
  /** JET: whether the derivatives of the rows of the property are needed (`jet_jacobian.hpp`). */
  CUDA bool jet_uses_jacobian() const {
    return jet_var_select == JET_VAR_SMEAR || jet_var_select == JET_VAR_ABS || jet_relu_select();
  }

  CUDA Configuration(const allocator_type& alloc = allocator_type{}):
    print_intermediate_solutions(false),
    stop_after_n_solutions(1),
    stop_after_n_nodes(std::numeric_limits<size_t>::max()),
    free_search(false),
    verbose_solving(0),
    print_ast(false),
    print_statistics(false),
    only_global_memory(false),
    force_ternarize(false),
    disable_simplify(false),
    disable_network_analysis(false),
    timeout_ms(0),
    or_nodes(0),
    subproblems_power(-1),
    subproblems_factor(300),
    stack_kb(
      #ifdef TURBO_IPC_ABSTRACT_DOMAIN
        32
      #else
        0
      #endif
    ),
    arch(
      #ifdef __CUDACC__
        Arch::BAREBONES
      #else
        Arch::CPU
      #endif
    ),
    fixpoint(
      #ifdef __CUDACC__
        FixpointKind::WAC1
      #else
        FixpointKind::AC1
      #endif
    ),
    wac1_threshold(0),
    seed(0),
    epsilon(1e-6),
    value_order("default", alloc),
    var_order("default", alloc),
    eps_value_order("default", alloc),
    eps_var_order("default", alloc),
    problem_path(alloc),
    vnnlib_path(alloc),
    onnx_path(alloc),
    version(alloc),
    hardware(alloc),
    jet_affine(1),
    jet_symbolic(false),
    jet_sym_rounds(0),
    jet_sym_layout(1),
    pir_fdeduce(1),
    split_on(SPLIT_ON_INPUT),
    relu_order(RELU_ORDER_EARLIEST),
    relu_leaf(RELU_LEAF_INPUT),
    jet_sym_clip_hidden(false),
    jet_sym_root_only(false),
    jet_sym_skip_final_interval(false),
    jet_candidates(false),
    jet_cand_every(0),
    jet_disjuncts(JET_DISJUNCTS_INTERLEAVED),
    jet_leaf_check(JET_LEAF_STRICT),
    jet_backward(JET_BACKWARD_OFF),
    jet_linear(JET_LINEAR_BOTH),
    jet_linear_n0(256),
    jet_bwd_rows(32),
    jet_bwd_every(0),
    jet_bwd_rounds(4),
    jet_bwd_clip(false),
    jet_bwd_et(false),
    forward_only(false),
    jet_interval_prop(JET_INTERVAL_FIXPOINT),
    under(UNDER_JET),
    scip_threads(4),
    scip_max_unstable(20),
    scip_time(10.0),
    scip_margin(1e-6),
    scip_queue(1024),
    scip_mode(SCIP_MODE_RACE),
    scip_cex_exact(true),
    scip_time_ratio(0.02),
    scip_time_min(0.5),
    scip_time_max(30.0),
    jet_cand_grad(true),
    jet_var_select(JET_VAR_DEFAULT),
    abs_beta(0.5),
    abs_psi(ABS_PSI_STD),
    abs_grad(ABS_GRAD_INTERVAL),
    relu_score(RELU_SCORE_SMEAR),
    abs_dive_count(true)
  {}

  Configuration(Configuration<allocator_type>&&) = default;
  Configuration(const Configuration<allocator_type>&) = default;

  template<class Alloc>
  CUDA Configuration(const Configuration<Alloc>& other, const allocator_type& alloc = allocator_type{}) :
    print_intermediate_solutions(other.print_intermediate_solutions),
    stop_after_n_solutions(other.stop_after_n_solutions),
    stop_after_n_nodes(other.stop_after_n_nodes),
    free_search(other.free_search),
    print_statistics(other.print_statistics),
    verbose_solving(other.verbose_solving),
    print_ast(other.print_ast),
    only_global_memory(other.only_global_memory),
    force_ternarize(other.force_ternarize),
    disable_simplify(other.disable_simplify),
    disable_network_analysis(other.disable_network_analysis),
    timeout_ms(other.timeout_ms),
    or_nodes(other.or_nodes),
    subproblems_power(other.subproblems_power),
    subproblems_factor(other.subproblems_factor),
    stack_kb(other.stack_kb),
    arch(other.arch),
    fixpoint(other.fixpoint),
    wac1_threshold(other.wac1_threshold),
    seed(other.seed),
    epsilon(other.epsilon),
    var_order(other.var_order, alloc),
    value_order(other.value_order, alloc),
    eps_var_order(other.eps_var_order, alloc),
    eps_value_order(other.eps_value_order, alloc),
    problem_path(other.problem_path, alloc),
    vnnlib_path(other.vnnlib_path, alloc),
    onnx_path(other.onnx_path, alloc),
    version(other.version, alloc),
    hardware(other.hardware, alloc),
    jet_affine(other.jet_affine),
    jet_symbolic(other.jet_symbolic),
    jet_sym_rounds(other.jet_sym_rounds),
    jet_sym_layout(other.jet_sym_layout),
    pir_fdeduce(other.pir_fdeduce),
    split_on(other.split_on),
    relu_order(other.relu_order),
    relu_leaf(other.relu_leaf),
    jet_sym_clip_hidden(other.jet_sym_clip_hidden),
    jet_sym_root_only(other.jet_sym_root_only),
    jet_sym_skip_final_interval(other.jet_sym_skip_final_interval),
    jet_candidates(other.jet_candidates),
    jet_cand_every(other.jet_cand_every),
    jet_disjuncts(other.jet_disjuncts),
    jet_leaf_check(other.jet_leaf_check),
    jet_backward(other.jet_backward),
    jet_linear(other.jet_linear),
    jet_linear_n0(other.jet_linear_n0),
    jet_bwd_rows(other.jet_bwd_rows),
    jet_bwd_every(other.jet_bwd_every),
    jet_bwd_rounds(other.jet_bwd_rounds),
    jet_bwd_clip(other.jet_bwd_clip),
    jet_bwd_et(other.jet_bwd_et),
    forward_only(other.forward_only),
    jet_interval_prop(other.jet_interval_prop),
    under(other.under),
    scip_threads(other.scip_threads),
    scip_max_unstable(other.scip_max_unstable),
    scip_time(other.scip_time),
    scip_margin(other.scip_margin),
    scip_queue(other.scip_queue),
    scip_mode(other.scip_mode),
    scip_cex_exact(other.scip_cex_exact),
    scip_time_ratio(other.scip_time_ratio),
    scip_time_min(other.scip_time_min),
    scip_time_max(other.scip_time_max),
    jet_cand_grad(other.jet_cand_grad),
    jet_var_select(other.jet_var_select),
    abs_beta(other.abs_beta),
    abs_psi(other.abs_psi),
    abs_grad(other.abs_grad),
    relu_score(other.relu_score),
    abs_dive_count(other.abs_dive_count)
  {}

  template <class Alloc2>
  CUDA Configuration<allocator_type>& operator=(const Configuration<Alloc2>& other) {
    print_intermediate_solutions = other.print_intermediate_solutions;
    stop_after_n_solutions = other.stop_after_n_solutions;
    stop_after_n_nodes = other.stop_after_n_nodes;
    free_search = other.free_search;
    verbose_solving = other.verbose_solving;
    print_ast = other.print_ast;
    print_statistics = other.print_statistics;
    only_global_memory = other.only_global_memory;
    force_ternarize = other.force_ternarize;
    disable_simplify = other.disable_simplify;
    disable_network_analysis = other.disable_network_analysis;
    timeout_ms = other.timeout_ms;
    or_nodes = other.or_nodes;
    subproblems_power = other.subproblems_power;
    subproblems_factor = other.subproblems_factor;
    stack_kb = other.stack_kb;
    arch = other.arch;
    fixpoint = other.fixpoint;
    wac1_threshold = other.wac1_threshold;
    seed = other.seed;
    epsilon = other.epsilon;
    var_order = other.var_order;
    value_order = other.value_order;
    eps_var_order = other.eps_var_order;
    eps_value_order = other.eps_value_order;
    problem_path = other.problem_path;
    vnnlib_path = other.vnnlib_path;
    onnx_path = other.onnx_path;
    version = other.version;
    hardware = other.hardware;
    jet_affine = other.jet_affine;
    jet_symbolic = other.jet_symbolic;
    jet_sym_rounds = other.jet_sym_rounds;
    jet_sym_layout = other.jet_sym_layout;
    pir_fdeduce = other.pir_fdeduce;
    split_on = other.split_on;
    relu_order = other.relu_order;
    relu_leaf = other.relu_leaf;
    jet_sym_clip_hidden = other.jet_sym_clip_hidden;
    jet_sym_root_only = other.jet_sym_root_only;
    jet_sym_skip_final_interval = other.jet_sym_skip_final_interval;
    jet_candidates = other.jet_candidates;
    jet_cand_every = other.jet_cand_every;
    jet_disjuncts = other.jet_disjuncts;
    jet_leaf_check = other.jet_leaf_check;
    jet_backward = other.jet_backward;
    jet_linear = other.jet_linear;
    jet_linear_n0 = other.jet_linear_n0;
    jet_bwd_rows = other.jet_bwd_rows;
    jet_bwd_every = other.jet_bwd_every;
    jet_bwd_rounds = other.jet_bwd_rounds;
    jet_bwd_clip = other.jet_bwd_clip;
    jet_bwd_et = other.jet_bwd_et;
    forward_only = other.forward_only;
    jet_interval_prop = other.jet_interval_prop;
    under = other.under;
    scip_threads = other.scip_threads;
    scip_max_unstable = other.scip_max_unstable;
    scip_time = other.scip_time;
    scip_margin = other.scip_margin;
    scip_queue = other.scip_queue;
    scip_mode = other.scip_mode;
    scip_cex_exact = other.scip_cex_exact;
    scip_time_ratio = other.scip_time_ratio;
    scip_time_min = other.scip_time_min;
    scip_time_max = other.scip_time_max;
    jet_cand_grad = other.jet_cand_grad;
    jet_var_select = other.jet_var_select;
    abs_beta = other.abs_beta;
    abs_psi = other.abs_psi;
    abs_grad = other.abs_grad;
    relu_score = other.relu_score;
    abs_dive_count = other.abs_dive_count;
  }

  CUDA void print_commandline(const char* program_name) {
    printf("%s -t %" PRIu64 " %s-n %" PRIu64 " %s%s%s%s",
      program_name,
      timeout_ms,
      (print_intermediate_solutions ? "-a ": ""),
      stop_after_n_solutions,
      (print_intermediate_solutions ? "-i ": ""),
      (free_search ? "-f " : ""),
      (print_statistics ? "-s " : ""),
      (print_ast ? "-ast " : "")
    );
    for(int i = 0; i < verbose_solving; ++i) {
      printf("-v ");
    }
    if(arch != Arch::CPU) {
      printf("-arch %s -or %" PRIu64 " -sub %d -subfactor %" PRIu64 " -stack %" PRIu64 " ", name_of_arch(arch), or_nodes, subproblems_power, subproblems_factor, stack_kb);
      if(only_global_memory) { printf("-globalmem "); }
    }
    else {
      printf("-arch cpu -p %" PRIu64 " ", or_nodes);
    }
    if(disable_simplify) { printf("-disable_simplify "); }
    if(force_ternarize) { printf("-force_ternarize "); }
    if(disable_network_analysis) { printf("-disable_network_analysis "); }
    printf("-fp %s ", name_of_fixpoint(fixpoint));
    if(fixpoint == FixpointKind::WAC1) {
      printf("-wac1_threshold %" PRIu64 " ", wac1_threshold);
    }
    printf("-seed %" PRIu64 " ", seed);
    printf("-epsilon %.10f", epsilon);
    printf("-var_order %s ", var_order.data());
    printf("-value_order %s ", value_order.data());
    printf("-eps_var_order %s ", eps_var_order.data());
    printf("-eps_value_order %s ", eps_value_order.data());
    if(version.size() != 0) {
      printf("-version %s ", version.data());
    }
    if(hardware.size() != 0) {
      printf("-hardware \'%s\' ", hardware.data());
    }
    printf("-cutnodes %" PRIu64 " ", stop_after_n_nodes == std::numeric_limits<size_t>::max() ? 0 : stop_after_n_nodes);
    if(arch == Arch::JET) {
      printf("-jet_affine v%d ", jet_affine);
      printf("-jet_interval_prop %s ", name_of_jet_interval_prop(jet_interval_prop));
      printf("-jet_disjuncts %s ", jet_disjuncts == JET_DISJUNCTS_SEQUENTIAL ? "sequential" : "interleaved");
      printf("-jet_leaf_check %s ", jet_leaf_check == JET_LEAF_LENIENT ? "lenient" : "strict");
    }
    if(arch == Arch::JET && jet_symbolic) {
      printf("-jet_symbolic -jet_sym_rounds %d -jet_sym_layout %s ", jet_sym_rounds, jet_sym_layout == 0 ? "neuron" : "slot");
      if(jet_sym_clip_hidden) { printf("-jet_sym_clip_hidden "); }
      if(jet_sym_root_only) { printf("-jet_sym_root_only "); }
      if(jet_sym_skip_final_interval) { printf("-jet_sym_skip_final_interval "); }
    }
    if((arch == Arch::JET || arch == Arch::FBAREBONES) && forward_only) {
      printf("-forward_only ");
    }
    if(arch == Arch::JET && jet_candidates) {
      printf("-jet_candidates -jet_cand_every %d -jet_cand_grad %s ", jet_cand_every, jet_cand_grad ? "on" : "off");
    }
    if(arch == Arch::JET) {
      printf("-under %s ", name_of_under(under));
      if(under == UNDER_SCIP) {
        printf("-scip_cex %s ", scip_cex_exact ? "exact" : "float");
        printf("-scip_mode %s -scip_threads %d -scip_max_unstable %d -scip_time %g -scip_time_ratio %g -scip_time_min %g -scip_time_max %g -scip_margin %g -scip_queue %d ",
          name_of_scip_mode(scip_mode), scip_threads, scip_max_unstable, scip_time, scip_time_ratio, scip_time_min, scip_time_max, scip_margin, scip_queue);
      }
      if(jet_var_select != JET_VAR_DEFAULT) {
        printf("-jet_var_select %s ", name_of_jet_var_select(jet_var_select));
        if(jet_var_select == JET_VAR_ABS) {
          printf("-abs_beta %g -abs_psi %s ", abs_beta, abs_psi == ABS_PSI_CV ? "cv" : "std");
        }
        if(jet_var_select == JET_VAR_ABS || jet_var_select == JET_VAR_RR) {
          printf("-abs_dive_count %s ", abs_dive_count ? "on" : "off");
        }
        if(jet_var_select == JET_VAR_ABS || jet_var_select == JET_VAR_SMEAR) {
          printf("-abs_grad %s ", abs_grad == ABS_GRAD_RELAX ? "relax" : "interval");
        }
      }
    }
    if(arch == Arch::JET && jet_backward != JET_BACKWARD_OFF) {
      printf("-jet_backward %s -jet_bwd_rows %d -jet_bwd_every %d -jet_bwd_rounds %d ", name_of_jet_backward(jet_backward), jet_bwd_rows, jet_bwd_every, jet_bwd_rounds);
      if(jet_bwd_clip) { printf("-jet_bwd_clip "); }
      if(jet_bwd_et) { printf("-jet_bwd_et "); }
    }
    if(arch == Arch::JET && jet_linear != JET_LINEAR_BOTH) {
      printf("-jet_linear %s -jet_linear_n0 %d ", name_of_jet_linear(jet_linear), jet_linear_n0);
    }
    if(arch == Arch::JET || arch == Arch::FBAREBONES) {
      printf("-split_on %s ", name_of_split_on(split_on));
      if(split_on == SPLIT_ON_RELU) {
        printf("-relu_order %s ", name_of_relu_order(relu_order));
        printf("-relu_leaf %s ", relu_leaf == RELU_LEAF_POINT ? "point" : "input");
      }
      if(jet_relu_select()) {
        printf("-relu_score %s ", relu_score == RELU_SCORE_GAP ? "gap" : "smear");
        if(split_on == SPLIT_ON_RELU && relu_order == RELU_ORDER_ABS && jet_var_select != JET_VAR_ABS) {
          printf("-abs_beta %g -abs_psi %s ", abs_beta, abs_psi == ABS_PSI_CV ? "cv" : "std");
        }
        if(jet_var_select != JET_VAR_SMEAR && jet_var_select != JET_VAR_ABS) {
          printf("-abs_grad %s ", abs_grad == ABS_GRAD_RELAX ? "relax" : "interval");
        }
      }
    }
    if(arch == Arch::FBAREBONES) {
      printf("-pir_fdeduce v%d ", pir_fdeduce);
    }
    printf("%s", problem_path.data());
  }

  CUDA const char* name_of_fixpoint(FixpointKind fixpoint) const {
    switch(fixpoint) {
      case FixpointKind::AC1:
        return "ac1";
      case FixpointKind::WAC1:
        return "wac1";
      default:
        assert(0);
        return "Unknown";
    }
  }

  CUDA const char* name_of_arch(Arch arch) const {
    switch(arch) {
      case Arch::CPU:
        return "cpu";
      case Arch::GPU:
        return "gpu";
      case Arch::BAREBONES:
        return "barebones";
      case Arch::FBAREBONES:
        return "fbarebones";
      case Arch::HYBRID:
        return "hybrid";
      case Arch::JET:
        return "jet";
      default:
        assert(0);
        return "Unknown";
    }
  }

  CUDA void print_mzn_statistics() const {
    printf("%%%%%%mzn-stat: problem_path=\"%s\"\n", problem_path.data());
    printf("%%%%%%mzn-stat: solver=\"Turbo\"\n");
    printf("%%%%%%mzn-stat: version=\"%s\"\n", (version.size() == 0) ? "1.2.9" : version.data());
    printf("%%%%%%mzn-stat: hardware=\"%s\"\n", (hardware.size() == 0) ? "unspecified" : hardware.data());
    printf("%%%%%%mzn-stat: arch=\"%s\"\n", name_of_arch(arch));
    printf("%%%%%%mzn-stat: fixpoint=\"%s\"\n", name_of_fixpoint(fixpoint));
    // printf("%%%%%%mzn-stat: subproblems_power=\"%d\"\n", subproblems_power); // do not print because it must be printed before it is modified in barebones.
    printf("%%%%%%mzn-stat: subproblems_factor=%" PRIu64 "\n", subproblems_factor);
    if(fixpoint == FixpointKind::WAC1) {
      printf("%%%%%%mzn-stat: wac1_threshold=%" PRIu64 "\n", wac1_threshold);
    }
    printf("%%%%%%mzn-stat: seed=%" PRIu64 "\n", seed);
    printf("%%%%%%nnv-stat: epsilon=\"%.10f\"\n", epsilon);
    if(arch == Arch::JET) {
      printf("%%%%%%nnv-stat: jet_affine=\"v%d\"\n", jet_affine);
      printf("%%%%%%nnv-stat: jet_interval_prop=\"%s\"\n", name_of_jet_interval_prop(jet_interval_prop));
      printf("%%%%%%nnv-stat: jet_disjuncts=\"%s\"\n", jet_disjuncts == JET_DISJUNCTS_SEQUENTIAL ? "sequential" : "interleaved");
      printf("%%%%%%nnv-stat: jet_leaf_check=\"%s\"\n", jet_leaf_check == JET_LEAF_LENIENT ? "lenient" : "strict");
      printf("%%%%%%nnv-stat: jet_symbolic=\"%s\"\n", jet_symbolic ? "yes" : "no");
      printf("%%%%%%nnv-stat: jet_sym_rounds=%d\n", jet_sym_rounds);
      printf("%%%%%%nnv-stat: jet_sym_layout=\"%s\"\n", jet_sym_layout == 0 ? "neuron" : "slot");
      printf("%%%%%%nnv-stat: jet_sym_clip_hidden=\"%s\"\n", jet_sym_clip_hidden ? "yes" : "no");
      printf("%%%%%%nnv-stat: jet_sym_root_only=\"%s\"\n", jet_sym_root_only ? "yes" : "no");
      printf("%%%%%%nnv-stat: jet_sym_skip_final_interval=\"%s\"\n", jet_sym_skip_final_interval ? "yes" : "no");
      printf("%%%%%%nnv-stat: jet_candidates=\"%s\"\n", jet_candidates ? "yes" : "no");
      printf("%%%%%%nnv-stat: jet_cand_every=%d\n", jet_cand_every);
      printf("%%%%%%nnv-stat: jet_cand_grad=\"%s\"\n", jet_cand_grad ? "on" : "off");
      printf("%%%%%%nnv-stat: jet_backward=\"%s\"\n", name_of_jet_backward(jet_backward));
      printf("%%%%%%nnv-stat: jet_linear=\"%s\"\n", name_of_jet_linear(jet_linear));
      printf("%%%%%%nnv-stat: jet_linear_n0=%d\n", jet_linear_n0);
      printf("%%%%%%nnv-stat: jet_bwd_rows=%d\n", jet_bwd_rows);
      printf("%%%%%%nnv-stat: jet_bwd_every=%d\n", jet_bwd_every);
      printf("%%%%%%nnv-stat: jet_bwd_rounds=%d\n", jet_bwd_rounds);
      printf("%%%%%%nnv-stat: jet_bwd_clip=\"%s\"\n", jet_bwd_clip ? "yes" : "no");
      printf("%%%%%%nnv-stat: jet_bwd_et=\"%s\"\n", jet_bwd_et ? "yes" : "no");
      printf("%%%%%%nnv-stat: under=\"%s\"\n", name_of_under(under));
      if(under == UNDER_SCIP) {
        printf("%%%%%%nnv-stat: scip_threads=%d\n", scip_threads);
        printf("%%%%%%nnv-stat: scip_max_unstable=%d\n", scip_max_unstable);
        printf("%%%%%%nnv-stat: scip_time=%g\n", scip_time);
        printf("%%%%%%nnv-stat: scip_margin=%g\n", scip_margin);
        printf("%%%%%%nnv-stat: scip_queue=%d\n", scip_queue);
        printf("%%%%%%nnv-stat: scip_mode=\"%s\"\n", name_of_scip_mode(scip_mode));
        printf("%%%%%%nnv-stat: scip_cex=\"%s\"\n", scip_cex_exact ? "exact" : "float");
        printf("%%%%%%nnv-stat: scip_time_ratio=%g\n", scip_time_ratio);
        printf("%%%%%%nnv-stat: scip_time_min=%g\n", scip_time_min);
        printf("%%%%%%nnv-stat: scip_time_max=%g\n", scip_time_max);
      }
      printf("%%%%%%nnv-stat: jet_var_select=\"%s\"\n", name_of_jet_var_select(jet_var_select));
      if(jet_var_select == JET_VAR_ABS) {
        printf("%%%%%%nnv-stat: abs_beta=%g\n", abs_beta);
        printf("%%%%%%nnv-stat: abs_psi=\"%s\"\n", abs_psi == ABS_PSI_CV ? "cv" : "std");
      }
      if(jet_var_select == JET_VAR_ABS || jet_var_select == JET_VAR_RR) {
        printf("%%%%%%nnv-stat: abs_dive_count=\"%s\"\n", abs_dive_count ? "on" : "off");
      }
      if(jet_var_select == JET_VAR_ABS || jet_var_select == JET_VAR_SMEAR) {
        printf("%%%%%%nnv-stat: abs_grad=\"%s\"\n", abs_grad == ABS_GRAD_RELAX ? "relax" : "interval");
      }
    }
    if(arch == Arch::JET || arch == Arch::FBAREBONES) {
      printf("%%%%%%nnv-stat: forward_only=\"%s\"\n", forward_only ? "yes" : "no");
      printf("%%%%%%nnv-stat: split_on=\"%s\"\n", name_of_split_on(split_on));
      printf("%%%%%%nnv-stat: relu_order=\"%s\"\n", name_of_relu_order(relu_order));
      if(jet_relu_select()) {
        printf("%%%%%%nnv-stat: relu_score=\"%s\"\n", relu_score == RELU_SCORE_GAP ? "gap" : "smear");
        printf("%%%%%%nnv-stat: relu_abs_beta=%g\n", abs_beta);
        printf("%%%%%%nnv-stat: relu_abs_psi=\"%s\"\n", abs_psi == ABS_PSI_CV ? "cv" : "std");
        printf("%%%%%%nnv-stat: relu_abs_grad=\"%s\"\n", abs_grad == ABS_GRAD_RELAX ? "relax" : "interval");
      }
      printf("%%%%%%nnv-stat: relu_leaf=\"%s\"\n", relu_leaf == RELU_LEAF_POINT ? "point" : "input");
    }
    if(arch == Arch::FBAREBONES) {
      printf("%%%%%%nnv-stat: pir_fdeduce=\"v%d\"\n", pir_fdeduce);
    }
    printf("%%%%%%mzn-stat: var_order=\"%s\"\n", var_order.data());
    printf("%%%%%%mzn-stat: value_order=\"%s\"\n", value_order.data());
    printf("%%%%%%mzn-stat: eps_var_order=\"%s\"\n", eps_var_order.data());
    printf("%%%%%%mzn-stat: eps_value_order=\"%s\"\n", eps_value_order.data());
    printf("%%%%%%mzn-stat: free_search=\"%s\"\n", free_search ? "yes" : "no");
    printf("%%%%%%mzn-stat: or_nodes=%" PRIu64 "\n", or_nodes);
    printf("%%%%%%mzn-stat: timeout_ms=%" PRIu64 "\n", timeout_ms);
    if(arch != Arch::CPU) {
      printf("%%%%%%mzn-stat: threads_per_block=%d\n", CUDA_THREADS_PER_BLOCK);
      printf("%%%%%%mzn-stat: stack_size=%" PRIu64 "\n", stack_kb * 1000);
      #ifdef CUDA_VERSION
        printf("%%%%%%mzn-stat: cuda_version=%d\n", CUDA_VERSION);
      #endif
      #ifdef __CUDA_ARCH__
        printf("%%%%%%mzn-stat: cuda_architecture=%d\n", __CUDA_ARCH__);
      #endif
    }
    printf("%%%%%%mzn-stat: cutnodes=%" PRIu64 "\n", stop_after_n_nodes == std::numeric_limits<size_t>::max() ? 0 : stop_after_n_nodes);
  }

  CUDA InputFormat input_format() const {
    if(problem_path.ends_with(".fzn")) {
      return InputFormat::FLATZINC;
    }
    else if(problem_path.ends_with(".xml")) {
      return InputFormat::XCSP3;
    }
    else if (problem_path.ends_with(".vnnlib")){
      return InputFormat::VNNLIB;
    }
    else if (problem_path.ends_with(".onnx")){
      return InputFormat::ONNX;
    }
    else if (problem_path.ends_with(".smt2")){
      return InputFormat::SMT2;
    }
    else {
      printf("ERROR: Unknown input format for the file %s [supported extension: .xml and .fzn].\n", problem_path.data());
      exit(EXIT_FAILURE);
    }
  }
};

void usage_and_exit(const std::string& program_name);
Configuration<battery::standard_allocator> parse_args(int argc, char** argv);

#endif
