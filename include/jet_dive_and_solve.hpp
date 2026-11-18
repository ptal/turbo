// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_FASTFBAREBONES_DIVE_AND_SOLVE_HPP
#define TURBO_FASTFBAREBONES_DIVE_AND_SOLVE_HPP

#include "battery/allocator.hpp"
#include "common_solving.hpp"
#include "memory_gpu.hpp"
#include "lala/light_branch.hpp"
#include <mutex>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <string>
#include <vector>
#include <algorithm>
#include <cfenv>

#include "lala/finterval.hpp"
#include "lala/vstore.hpp"
#include "jet_affine_v2.hpp"
#include "jet_symbolic.hpp"
#include "uass.hpp"
#include "relu_split.hpp"
#include "jet_network.hpp"
#include "jet_parsing.hpp"
#include "jet_candidates.hpp"
#include "jet_subproblems.hpp"
#include "jet_backward.hpp"
#include "jet_scip.hpp"
#include "jet_jacobian.hpp"
#include "jet_abs.hpp"

/** This is required in order to guess the usage of global memory, and increase the CUDA default limit. */
#define MAX_SEARCH_DEPTH 10000

namespace bt = ::battery;

/**
 * The full GPU version (`gpu_dive_and_solve`) is not compiling on modern GPU hardware (SM >= 9) due to the kernel being too large.
 * We circuvanted this issue by creating an hybrid version where only propagation is executed on the GPU (`hybrid_dive_and_solve`).
 * This has the disadvantage of memory transfers between CPU and GPU and synchronization overheads.
 * We propose a new "fbarebones" version which contains less abstractions than the GPU and hybrid versions, but have the same functionalities.
 * In particular, we directly implement the branch-and-bound algorithm here and avoid using `lala::BAB` and `lala::SearchTree` which are nice from a software engineering perspective but bring significant overhead.
 * This version is intended to reach the best possible performance.
 *
 * Terminology:
 *  * unified data: data available to both the CPU and GPU.
 *  * block data: data used within a single block.
 *  * grid data: data shared among all blocks in the grid.
 */

#ifdef __CUDACC__

#include <cuda/std/chrono>
#include <cuda/semaphore>
// #include "lala/onnxruntime-linux-x64-gpu-1.19.2/include/onnxruntime_cxx_api.h"
// #include <cuda_runtime.h>

#endif

namespace jet {

#ifdef __CUDACC__
#ifndef TURBO_IPC_ABSTRACT_DOMAIN


/** Fast neural network verification design on GPU.
 * This is the abstract domain of this architecture, it replaces `CP<FItv>` (a store of variables
 * plus the propagators of `PIR`): the constraints are not ternarized and not represented as
 * bytecodes, they are the network itself, and `deduce` implements the forward and backward
 * propagation of one neuron directly.
 * The store is held by pointer so that each block can own its own copy (possibly in shared memory)
 * while sharing the network `net`.
 */
template <class StoreAlloc, class NetAlloc>
struct FastNNRelu {
  using allocator_type = StoreAlloc;
  using NStore = VStore<FItv, StoreAlloc>;
  using network_type = FastNNReluNetwork<NetAlloc>;
  using property_type = FastProperty<NetAlloc>;
  using this_type = FastNNRelu<StoreAlloc, NetAlloc>;

  /** The network, shared and read-only. */
  const network_type* net;
  /** The non-unary constraints of the property, shared and read-only. */
  const property_type* prop;
  /** The bounds of the neurons, private to the block. */
  abstract_ptr<NStore> store;
  /** Which deduction of the affine neurons is used: `1` is `deduce_neuron` below (default), `2`
   * is `affine_v2::deduce` (see `jet_affine_v2.hpp`). Selected with `-jet_affine <v1|v2>`. */
  int affine_version;
  /** `-forward_only` or `-jet_interval_prop forward`: the deductions of the neurons only project
   * forward, from a layer onto the next ones (the sum, the bias and the ReLU), and never tighten the
   * neurons they read. */
  bool forward_only;

  CUDA FastNNRelu(const network_type* net, const property_type* prop, abstract_ptr<NStore> store,
    int affine_version = 1, bool forward_only = false)
   : net(net), prop(prop), store(store), affine_version(affine_version), forward_only(forward_only)
  {}

public:
  // `i` designates the target neuron among those that have a deduction, that is, every neuron but
  // those of the input layer: `i` ranges over `[0, num_deductions())` and updates the neuron
  // `neurons[num_inputs() + i]`. Consecutive `i` are therefore consecutive neurons of the same
  // layer, except across a layer boundary.
  // One thread handles one neuron: it reads the intervals of the neurons of the previous layer,
  // multiplies them by the weights of the connections into the target, adds its bias, applies the
  // ReLU (except on the output layer, which is affine only), and merges the result into the store
  // with a meet. The affine part and the ReLU are fused,
  // so the pre-activation never leaves the registers and a layer is updated in one deduction per
  // neuron, without any intra-warp reduction.
  // The shape of a layer and the layer it reads from are given by its `LayerInfo`, which also says
  // whether a ReLU applies; a residual layer is handled by `deduce_add` instead.
  /** Deduction of one neuron `j` of a residual layer \f$ target = src0_j + src1_j \f$.
   * `tell::fadd` does the forward projection and both backward ones in a single call, so the sum
   * narrows the target and the target narrows the two branches. With `fwd_only` (`forward_only`, or
   * the forward sweep of `-jet_interval_prop forward_backward`), the two branches are left as they are. */
  CUDA bool deduce_add(const LayerInfo& L, int target, int j, bool fwd_only) {
    using bound_type = typename FItv::LB::value_type;
    using RItv = FInterval<bound_type>;
    using local_itv = typename FItv::local_type;
    auto& neurons = *store;
    const int a_idx = net->layers[L.src0].base + j;
    const int b_idx = net->layers[L.src1].base + j;

    RItv a(neurons[a_idx].lb().value(), neurons[a_idx].ub().value());
    RItv b(neurons[b_idx].lb().value(), neurons[b_idx].ub().value());
    RItv t(neurons[target].lb().value(), neurons[target].ub().value());

    bool has_changed = false;
    if(L.has_relu == 0 && L.sigmoid == 0) {
      /** The neuron holds the sum itself, so one `fadd` propagates in the three directions. */
      tell::fadd(t, a, b);
      has_changed |= neurons.embed(target,
        local_itv(typename local_itv::LB(t.lb().load()),
                  typename local_itv::UB(t.ub().load())));
    }
    else {
      /** With a ReLU the neuron holds `max(sum, 0)`, so the sum is an intermediate value: we
       * compute it forward, let the ReLU narrow it from the domain of the neuron, then push that
       * narrowing back to the two branches with a second `fadd`. */
      RItv sum;
      sum.join_top();
      tell::fadd(sum, a, b);
      if(L.sigmoid != 0) {
        sigmoid::tell<bound_type>(t, sum);
      }
      else {
        RItv zero(bound_type{0});
        tell::fmax(t, sum, zero);
      }
      tell::fadd(sum, a, b);
      has_changed |= neurons.embed(target,
        local_itv(typename local_itv::LB(t.lb().load()),
                  typename local_itv::UB(t.ub().load())));
    }
    if(fwd_only) {
      return has_changed;
    }
    has_changed |= neurons.embed(a_idx,
      local_itv(typename local_itv::LB(a.lb().load()),
                typename local_itv::UB(a.ub().load())));
    has_changed |= neurons.embed(b_idx,
      local_itv(typename local_itv::LB(b.lb().load()),
                typename local_itv::UB(b.ub().load())));
    return has_changed;
  }

  /** Deduction of one neuron `j` of a scale layer \f$ target = s_j \cdot src0_j + t_j \f$
   * (`LAYER_SCALE`, e.g. the normalization of the inputs of `cora_2024`), optionally followed by a
   * ReLU. The product and the sum are the ternary propagators `tell::fmul` and `tell::fadd`, which
   * round outward: forward, the target is narrowed from the operand; backward, the operand is
   * narrowed from the target (nothing when `s_j = 0`, `fmul` then leaves it at top). With
   * `fwd_only`, the operand is left as it is. */
  CUDA bool deduce_scale(const LayerInfo& L, int target, int j, bool fwd_only) {
    using bound_type = typename FItv::LB::value_type;
    using RItv = FInterval<bound_type>;
    using local_itv = typename FItv::local_type;
    auto& neurons = *store;
    const int a_idx = net->layers[L.src0].base + j;
    RItv a(neurons[a_idx].lb().value(), neurons[a_idx].ub().value());
    RItv s(static_cast<bound_type>(net->weights[L.weight_base + j]));
    RItv t(static_cast<bound_type>(net->biases[L.bias_base + j]));
    RItv y(neurons[target].lb().value(), neurons[target].ub().value());
    /** `p = s * a`, from top: only its forward projection does work here. */
    RItv p;
    p.join_top();
    tell::fmul(p, a, s);
    if(L.has_relu == 0 && L.sigmoid == 0) {
      /** `y = p + t`: narrows `y` from `p`, and `p` from `y`. */
      tell::fadd(y, p, t);
    }
    else {
      /** With an activation, the pre-activation `v = p + t` is an intermediate value, as in
       * `deduce_add`. */
      RItv v;
      v.join_top();
      tell::fadd(v, p, t);
      if(L.sigmoid != 0) {
        sigmoid::tell<bound_type>(y, v);
      }
      else {
        RItv zero(bound_type{0});
        tell::fmax(y, v, zero);
      }
      tell::fadd(v, p, t);
    }
    bool has_changed = neurons.embed(target,
      local_itv(typename local_itv::LB(y.lb().load()), typename local_itv::UB(y.ub().load())));
    if(fwd_only) {
      return has_changed;
    }
    /** And back through the product: narrows `a` from `p`. */
    tell::fmul(p, a, s);
    has_changed |= neurons.embed(a_idx,
      local_itv(typename local_itv::LB(a.lb().load()), typename local_itv::UB(a.ub().load())));
    return has_changed;
  }

  /** Access to the store of neurons in the form expected by `affine_v2::deduce`. */
  struct NeuronsAccess {
    using bound_type = typename FItv::LB::value_type;
    using local_itv = typename FItv::local_type;
    NStore& s;
    CUDA INLINE bound_type lb(int i) const { return s[i].lb().value(); }
    CUDA INLINE bound_type ub(int i) const { return s[i].ub().value(); }
    /** `embed` meets each bound only if it is strictly better, so a NaN bound is ignored. */
    CUDA INLINE bool meet(int i, bound_type l, bound_type u) {
      return s.embed(i, local_itv(typename local_itv::LB(l), typename local_itv::UB(u)));
    }
    CUDA INLINE bool fail(int i) { return s.embed(i, local_itv::bot()); }
  };

  /** Version 2 of the deduction of an affine neuron (`-jet_affine v2`), see `jet_affine_v2.hpp`.
   * Same projection as `deduce_neuron`, with less arithmetic and a skipped backward pass when no
   * term can be tightened. */
  CUDA bool deduce_neuron_v2(const LayerInfo& L, int target, int j, bool fwd_only) {
    using bound_type = typename FItv::LB::value_type;
    const LayerInfo& S = net->layers[L.src0];
    if(L.kind == LAYER_SPARSE) {
      /** The stored weights of the row `j`, in increasing order of their sources. */
      int begin, end;
      symbolic::sparse_row(*net, L, j, begin, end);
      NeuronsAccess access{*store};
      const auto& col = net->sp_col;
      const auto& val = net->sp_val;
      return affine_v2::deduce_indexed<bound_type>(access,
        [&](int e) { return static_cast<bound_type>(val[begin + e]); },
        [&](int e) { return S.base + col[begin + e]; },
        end - begin, target, static_cast<bound_type>(net->biases[L.bias_base + j]),
        activation_of(L), !fwd_only);
    }
    const int layer_size = L.size;
    /** Weights are column-major: the weight of `(c, j)` is at `weight_base + c * size + j`. */
    const auto& weights = net->weights;
    const int wbase = L.weight_base + j;
    NeuronsAccess access{*store};
    return affine_v2::deduce<bound_type>(access,
      [&](int c) { return static_cast<bound_type>(weights[wbase + c * layer_size]); },
      S.base, S.size, target, static_cast<bound_type>(net->biases[L.bias_base + j]),
      activation_of(L), !fwd_only);
  }

  CUDA bool deduce_neuron(int i) {
    return deduce_neuron_dir(i, forward_only);
  }

  /** The deduction `i` of the neuron of the layer `l` at index `j` within the layer, as
   * `deduce_neuron_dir` (used by the sweeps of `-jet_interval_prop forward_backward`). */
  CUDA INLINE bool deduce_layer_neuron(int l, int j, bool fwd_only) {
    return deduce_neuron_dir(net->layers[l].base + j - net->num_inputs(), fwd_only);
  }

  /** The deduction `i` of a neuron. With `fwd_only`, the forward projection only: the neuron is
   * computed from the neurons it reads, which are left as they are. Otherwise, the forward and the
   * backward projections. */
  CUDA bool deduce_neuron_dir(int i, bool fwd_only) {
    /** Local aliases so the deduction reads exactly as if the store and the network parameters were
     * members of this class. */
    auto& neurons = *store;
    const auto& weights = net->weights;
    const auto& biases = net->biases;

    assert(i >= 0 && i < net->num_deductions());
    /** The input layer owns the first neurons and has no deduction, so the deductions map onto the
     * remaining neurons in order. The owning layer is read off `neuron_layer` in O(1), instead of
     * walking the layers as a chain would allow. */
    const int target = net->num_inputs() + i;
    const LayerInfo& L = net->layers[net->neuron_layer[target]];
    const int j = target - L.base;  /**< index of the target within its own layer. */
    assert(j >= 0 && j < L.size);

    if(L.kind == LAYER_ADD) {
      return deduce_add(L, target, j, fwd_only);
    }
    if(L.kind == LAYER_SCALE) {
      return deduce_scale(L, target, j, fwd_only);
    }

    if(affine_version == 2) {
      return deduce_neuron_v2(L, target, j, fwd_only);
    }

    using bound_type = typename FItv::LB::value_type;
    /** The interval of lala-interval, held in registers. Only the final result of the neuron is
     * merged back into `neurons`, which stores the shared `FItv` of the solver. */
    using RItv = FInterval<bound_type>;
    using local_itv = typename FItv::local_type;

    const LayerInfo& S = net->layers[L.src0];
    const int layer_size = L.size;
    const int prev_base = S.base;
    /** The terms of the neuron: for a dense layer, the `S.size` neurons of the source layer, the
     * weight of the connection `(c, j)` being at `c * layer_size + j` within the block of the layer
     * (column-major, see the layout conventions above); for a sparse layer, the stored weights of the
     * row `j`, in increasing order of their sources. The zero weights of a dense layer give zero terms,
     * so both add the same nonzero terms in the same order. */
    const int wbase = L.weight_base + j;
    const bool sparse = L.kind == LAYER_SPARSE;
    int row_begin = 0, row_end = 0;
    if(sparse) { symbolic::sparse_row(*net, L, j, row_begin, row_end); }
    const int fan_in = sparse ? row_end - row_begin : S.size;
    const auto& sp_col = net->sp_col;
    const auto& sp_val = net->sp_val;
    auto source = [&](int c) { return sparse ? prev_base + sp_col[row_begin + c] : prev_base + c; };
    auto weight = [&](int c) {
      return sparse ? static_cast<bound_type>(sp_val[row_begin + c]) : static_cast<bound_type>(weights[wbase + c * layer_size]);
    };

    RItv sum(bound_type{0});  /**< running pre-activation without the bias. */
    RItv r1,r2,r3;

    // STEP 1: compute the pre-activation \f$ s = \sum_c w_{jc} * x_c + b_j \f$ and store it in `sum`.

    /** Pre-activation \f$ s = \sum_c w_{jc} * x_c + b_j \f$. Each result is a variable that we just
     * reset to top, so we use the forward projections instead of the relational propagators: the
     * backward passes would only narrow operands that are already exact. */
    for(int c = 0; c < fan_in; ++c) {
      // Temporarily necessary to convert between the two kinds of interval (in lala-core and lala-interval).
      r1 = RItv(neurons[source(c)].lb().value(), neurons[source(c)].ub().value());
      r2 = RItv(weight(c));
      r3.join_top();
      r3.mul(r1, r2);   // neuron X weight.
      /** Running sum, accumulated bound by bound to avoid the copy that `add` would need (it meets
       * its result instead of assigning it). The rounding must go outward, `+` would round to
       * nearest and could cut off solutions on either side. */
      if(r3.is_bot()) { sum.meet_bot(); break; }
      sum.lb() = battery::add_down<bound_type>(sum.lb().load(), r3.lb().load());
      sum.ub() = battery::add_up<bound_type>(sum.ub().load(), r3.ub().load());
    }

    /** A bot term makes the whole neuron bot. We write it to the store immediately so that the
     * solver observes the failure and can stop, and we skip the rest: the backward propagation has
     * nothing sound to say about a contradiction. */
    if(sum.is_bot()) {
      return neurons.embed(target, local_itv::bot());
    }

    // STEP 2: add the bias and apply the ReLU, then merge the result into the neuron `j` of layer `l`.

    /** Each affine layer owns a block of `size` biases at `bias_base`, so the bias of the neuron
     * `j` of this layer is `biases[L.bias_base + j]`. */
    r1 = RItv(static_cast<bound_type>(biases[L.bias_base + j]));
    r2.join_top();

    /** Forward projection rather than `tell::fadd`: `add` takes its operands by value, so `sum`
     * provably keeps the plain forward accumulation that STEP 3 needs to undo. */
    r2.add(sum, r1);  // Pre-activation + bias.

    /** Whether the layer is followed by a ReLU is read off the graph rather than assumed: the
     * output layer is usually affine only, but in `tllverifybench` every other hidden layer is
     * affine too. Applying a ReLU where the graph has none would cut off the negative values of the
     * layer, hence unsound conclusions on the property. */
    const bool apply_relu = (L.has_relu != 0);

    RItv zero(bound_type{0});
    r3 = RItv(neurons[target].lb().value(), neurons[target].ub().value());
    if(apply_relu) {
      tell::fmax(r3, r2, zero);  // ReLU. Its backward pass narrows `r2` from the domain of neuron `j`.
    }
    else if(L.sigmoid != 0) {
      sigmoid::tell<bound_type>(r3, r2);  // sigmoid, forward on `r3` and backward on `r2` (`jet_sigmoid.hpp`).
    }
    else {
      /** Without the ReLU, the neuron holds the pre-activation itself: the forward pass and the
       * backward pass of STEP 3 are then the same intersection of `r2` with the domain of the
       * neuron. */
      r3.meet(r2);
      r2 = r3;
    }

    /** `embed` meets the result into the neuron and returns `true` if its domain got smaller. */
    bool has_changed = neurons.embed(target,
      local_itv(typename local_itv::LB(r3.lb().load()),
                typename local_itv::UB(r3.ub().load())));

    // STEP 3: Perform backward propagation to update the neurons of the previous layer.
    if(fwd_only) {
      return has_changed;
    }

    /** `r2` is the pre-activation narrowed by the ReLU above, and `r1` still holds the bias. The
     * bias is a singleton, so pairing opposite bounds costs nothing and `sub` is exact here: `nsum`
     * is the narrowed value of \f$ \sum_c w_{jc} * x_c \f$, to be confronted with the forward
     * `sum`, which is left untouched by STEP 2 and still holds the plain forward accumulation. */
    RItv nsum;
    nsum.sub(r2, r1);
    if(!nsum.is_bot()) {
      RItv partial;  /**< the sum of all the *other* terms. */
      for(int c = 0; c < fan_in; ++c) {
        r1 = RItv(neurons[source(c)].lb().value(), neurons[source(c)].ub().value());
        r2 = RItv(weight(c));
        /**< r3 is the term \f$ w_{jc} * x_c \f$ that STEP 1 accumulated. */
        r3.join_top();
        r3.mul(r1, r2);

        /** `partial = sum (-) term`, where `(-)` undoes the addition of STEP 1 bound by bound.
         * Interval addition is separable — the lower bound of a sum is the sum of the lower bounds
         * — so subtracting the *same* side recovers \f$ \sum_{c' \neq c} w_{jc'} * x_{c'} \f$
         * exactly (up to one rounding). The interval subtraction `sum - term` would instead pair
         * opposite bounds and widen the result by the width of `term` on each side, which would
         * leave almost nothing to propagate. This is only a valid undo because `sum` is the
         * untouched forward value; reversing a narrowed sum would cut off solutions. */
        partial.lb() = battery::sub_down<bound_type>(sum.lb().load(), r3.lb().load());
        partial.ub() = battery::sub_up<bound_type>(sum.ub().load(), r3.ub().load());

        /** The narrowing comes from `nsum`: only `r3.sub(nsum, partial)` does work here. */
        tell::fadd(nsum, partial, r3);
        /** And back through the product: `r1.mul_back(r3, r2)` narrows the neuron `c`. */
        tell::fmul(r3, r1, r2);
        has_changed |= neurons.embed(source(c),
          local_itv(typename local_itv::LB(r1.lb().load()),
                    typename local_itv::UB(r1.ub().load())));
      }
    }
    return has_changed;
  }

  /** Deduction of the comparison `neurons[x] - neurons[y] <= k` of the property.
   * The narrowing is the standard one for an inequality: the upper bound of `x` cannot exceed that
   * of `y` (shifted by `k`), and the lower bound of `y` cannot go below that of `x`. Both are
   * rounded outward so that no solution is cut off. */
  CUDA bool deduce_property(int i) {
    using bound_type = typename FItv::LB::value_type;
    using local_itv = typename FItv::local_type;
    using LB2 = typename local_itv::LB;
    using UB2 = typename local_itv::UB;
    auto& neurons = *store;
    const NeuronLeq& c = prop->leqs[i];
    const bound_type k = static_cast<bound_type>(c.k);
    const bound_type ub_y = neurons[c.y].ub().value();
    const bound_type lb_x = neurons[c.x].lb().value();
    bool has_changed = neurons.embed(c.x,
      local_itv(LB2::top(), UB2(battery::add_up<bound_type>(ub_y, k))));
    has_changed |= neurons.embed(c.y,
      local_itv(LB2(battery::sub_down<bound_type>(lb_x, k)), UB2::top()));
    return has_changed;
  }

  /** The deductions of the abstract domain are the neurons first, then the comparisons of the
   * property, so that `i` indexes them contiguously for the fixpoint engine. */
  CUDA INLINE bool deduce(int i) {
    const int n = net->num_deductions();
    return i < n ? deduce_neuron(i) : deduce_property(i - n);
  }

  /** The fixpoint engines of `lala-core` expect a deduction to be split into a `load_deduce` and a
   * `deduce` (see `fwarp_fixpoint`). We have nothing to load: a deduction is entirely described by
   * its index. */
  CUDA INLINE int load_deduce(int i) const {
    return i;
  }

  /** `epsilon` is unused: unlike the propagators of `PIR`, the deduction of a neuron is always
   * performed, its precision being that of the interval arithmetic. */
  CUDA INLINE local::B fdeduce(int i, const double epsilon) {
    return deduce(i);
  }

  CUDA INLINE local::B is_bot() const {
    return store->is_bot();
  }

  CUDA INLINE int num_deductions() const {
    return net->num_deductions() + prop->size();
  }

  CUDA INLINE AType aty() const {
    return store->aty();
  }

  CUDA void print() const {
    printf("%% In total, we have %d neurons in the network\n", (int)store->vars());
  }
};


/** The root of the problem, in managed memory, hence shared between the CPU and the GPU.
 * It replaces `GridCP` (an `AbstractDomains` over `FItv`): the abstract domain is now the store of
 * neurons `store` together with the propagator `FastNNRelu`, so neither `PIR` nor the ternarization
 * of the constraints is needed here.
 * It exposes `config`, `stats` and `prune()` so that the host helpers `must_quit`, `check_timeout`
 * and `wait_solving_ends` keep working unchanged.
 */
struct NNRoot {
  Configuration<ConcurrentAllocator> config;
  Statistics<ConcurrentAllocator> stats;

  /** The network, shared by all the blocks. */
  Network net;

  /** The disjuncts searched by a kernel launch (`fbarebones_dive_and_solve`): all of them with
   * `-jet_disjuncts interleaved`, one with `sequential`. A disjunctive property is expanded into DNF on
   * the host; the disjunct `k` of a launch (`0 <= k < num_disjuncts()`) is the disjunct
   * `disjunct_ids[k]` of the DNF. For each one:
   *   * `stores[k]`: the bounds of the neurons at the root of its search, the input box and, the
   *     property being unary in this application, the bounds of the output neurons expressing it;
   *   * `props[k]`: its non-unary constraints (comparisons of two neurons);
   *   * `cand_stores[k]` and `cand_props[k]`: the disjunct rounded inward (`apply_disjunct_inner`),
   *     for the strict test of the candidate points and of the leaves (`-jet_leaf_check strict`).
   * A block switches to the disjunct of each subproblem it takes (`jet_subproblems.hpp`). The vectors
   * are rewritten between two launches. */
  bt::vector<abstract_ptr<NNStore>, ConcurrentAllocator> stores;
  bt::vector<Property, ConcurrentAllocator> props;
  bt::vector<abstract_ptr<NNStore>, ConcurrentAllocator> cand_stores;
  bt::vector<Property, ConcurrentAllocator> cand_props;
  bt::vector<int, ConcurrentAllocator> disjunct_ids;
  /** `-jet_backward`: the bounds of the pre-activations at the root of each disjunct of the launch, computed
   * by the backward contractor on the root (`gpu_root_symbolic`) and valid in every node of its tree
   * (`backward::pass`, `root_p`): `2 * net.num_neurons` values per disjunct, the lower bounds then the
   * upper ones (meaningful for the neurons with an activation only). Empty without `-jet_backward`. */
  bt::vector<typename FItv::LB::value_type, ConcurrentAllocator> bwd_root_p;
  /** `-jet_var_select smear|abs` (`jet_jacobian.hpp`): the rows of the Jacobian of each disjunct of the launch,
   * those of the disjunct `k` being `jac_rows[jac_row_start[k] .. jac_row_start[k+1])`: its comparisons
   * `x - y <= c`, and its output neurons with a finite bound in the region. */
  bt::vector<jacobian::Row, ConcurrentAllocator> jac_rows;
  bt::vector<int, ConcurrentAllocator> jac_row_start;

  /** `stores[0]`: the shape of the stores of the blocks (all the disjuncts have the same neurons). */
  abstract_ptr<NNStore> store;

  /** The disjunct of the DNF of the counterexample `best`, -1 if none. */
  int found_disjunct;

  /** The bounds of the neurons of the counterexample, when one is found. */
  abstract_ptr<NNStore> best;

  /** The branching strategy. The orders are resolved on the host because `config` holds them as
   * strings and the parsing functions are host-only. */
  VariableOrder var_order;
  ValueOrder val_order;

  template <class Alloc>
  NNRoot(const Configuration<Alloc>& config, const Network& net,
    VariableOrder var_order, ValueOrder val_order)
   : config(config, ConcurrentAllocator{})
   , stats(static_cast<size_t>(net.num_neurons), static_cast<size_t>(net.num_deductions()),
           false, config.print_statistics)
   , net(net, ConcurrentAllocator{})
   , stores(ConcurrentAllocator{})
   , props(ConcurrentAllocator{})
   , cand_stores(ConcurrentAllocator{})
   , cand_props(ConcurrentAllocator{})
   , disjunct_ids(ConcurrentAllocator{})
   , bwd_root_p(ConcurrentAllocator{})
   , jac_rows(ConcurrentAllocator{})
   , jac_row_start(ConcurrentAllocator{})
   , store(bt::allocate_shared<NNStore, ConcurrentAllocator>(ConcurrentAllocator{},
             AType{0}, net.num_neurons, ConcurrentAllocator{}))
   , found_disjunct(-1)
   , best(bt::allocate_shared<NNStore, ConcurrentAllocator>(ConcurrentAllocator{},
             AType{0}, net.num_neurons, ConcurrentAllocator{}))
   , var_order(var_order)
   , val_order(val_order)
  {}

  /** The number of disjuncts searched by the current launch. */
  CUDA INLINE int num_disjuncts() const { return static_cast<int>(stores.size()); }

  /** The largest number of deductions of the propagator over the disjuncts of the launch (the
   * comparisons of a disjunct are deductions, `FastNNRelu::num_deductions`): the size of the fixpoint
   * engine of a block, which switches between the disjuncts. */
  CUDA int max_num_deductions() const {
    int m = 0;
    for(int k = 0; k < num_disjuncts(); ++k) { m = battery::max(m, props[k].size()); }
    return net.num_deductions() + m;
  }

  /** Same as `AbstractDomains::prune`, required by `must_quit` and `check_timeout`. */
  CUDA void prune() {
    stats.exhaustive = false;
  }
};

/** `-under scip` (`jet_scip.hpp`): the queue of the nodes sent to SCIP by the blocks (`scip_offer`), in
 * pinned memory mapped in the device, read by the dispatcher thread of the host
 * (`fbarebones_dive_and_solve`). The slot `s` holds `state[s]` (`SCIP_SLOT_FREE`, `SCIP_SLOT_WRITING` by
 * a block, `SCIP_SLOT_READY` for the host), the disjunct of the DNF `dnf[s]`, the sequence number `seq[s]`
 * and, at `bounds + s * 2n`, the lower then the upper bounds of the `n` neurons of the node. `head`: the
 * number of slots taken so far (a slot is `head % capacity`). The pointers are device pointers; all of
 * them are null without `-under scip`. */
constexpr int SCIP_SLOT_FREE = 0;
constexpr int SCIP_SLOT_WRITING = 1;
constexpr int SCIP_SLOT_READY = 2;
struct ScipQueueView {
  int* state = nullptr;
  int* dnf = nullptr;
  unsigned long long* seq = nullptr;
  float* bounds = nullptr;
  unsigned long long* head = nullptr;
  int capacity = 0;
  int n = 0;
  /** `-scip_mode race`, the answers of SCIP to the blocks and back. The sequence number `t` of a node sent
   * (`seq`, the value of `head` when it was taken) is its ticket. `result[t % rcap]`, written by the host,
   * is `((t + 1) << 8) | status` (`scip_under::Status`); `cancel[t % rcap] == t + 1`, written by a block,
   * asks the host not to solve the node (its subtree was searched without unknown leaf); `rec[2i]`,
   * `rec[2i+1]` (`i < *rec_count`, at most `rec_cap`) record `t + 1` and the number of unknown leaves of the
   * subtree of a node that a block left before SCIP answered, for the end of the solving. All null
   * otherwise. */
  unsigned long long* result = nullptr;
  unsigned long long* cancel = nullptr;
  unsigned long long* rec = nullptr;
  unsigned long long* rec_count = nullptr;
  int rcap = 0;
  int rec_cap = 0;
};
/** The code of `scip_under::CLOSED` in `ScipQueueView::result`. */
constexpr unsigned long long SCIP_RESULT_CLOSED = 7;

/** Data shared between CPU and GPU. */
struct UnifiedData {
  /** The root node of the problem, useful to backtrack when solving a new subproblem.
   * Also contains the shared information such as statistics and solver configuration.
   */
  NNRoot root;

  /** Stop signal from the CPU because of a timeout or CTRL-C. */
  cuda::std::atomic_flag stop;

  /** The memory configuration of each block. */
  MemoryConfig mem_config;

  /** `-jet_sym_root_only`: the number of symbolic rounds done on the root by `gpu_root_symbolic`. */
  int root_sym_rounds;

  /** `-jet_candidates`: the number of nodes where the candidates were tested, of candidate points
   * tested, and of counterexamples found by them. */
  unsigned long long cand_nodes;
  unsigned long long cand_points;
  unsigned long long cand_found;
  /** The counterexamples found by the candidates, by kind: the middle, a corner of the symbolic forms, a
   * corner of the interval Jacobian (`-jet_cand_grad`); and the nodes where the gradient corners were
   * tested. */
  unsigned long long cand_found_mid;
  unsigned long long cand_found_forms;
  unsigned long long cand_found_grad;
  unsigned long long cand_grad_nodes;

  /** `-jet_backward`: the number of passes of the backward contractor, of rows back-substituted, and of
   * multiply-adds of their pushes (an estimate, `backward::Counters`). */
  unsigned long long bwd_passes;
  unsigned long long bwd_rows;
  unsigned long long bwd_macs;

  /** `-under scip`: the queue of the nodes sent to SCIP, the number of nodes written in it, and of the
   * nodes dropped because their slot was busy. */
  ScipQueueView scip_queue;
  unsigned long long scip_sent;
  unsigned long long scip_dropped;
  /** `-scip_mode race`: the subtrees left because SCIP closed their node, the nodes cancelled because the
   * GPU searched their subtree without unknown leaf, and the subtrees left with unknown leaves before SCIP
   * answered (recorded, and lost when `rec_cap` is exceeded). */
  unsigned long long race_pruned;
  unsigned long long race_cancelled;
  unsigned long long race_recorded;
  unsigned long long race_lost;

  /** `-jet_var_select` (`jet_abs.hpp`): the number of inputs chosen by it, of those chosen as the widest
   * because the smear could not be computed (a non-finite derivative or bound), of the smears taken from
   * the symbolic forms (`-abs_grad relax`), and the sums of the numbers of candidates and of the sizes of
   * the restricted candidate lists of ABS. */
  unsigned long long abs_choices;
  unsigned long long abs_fallbacks;
  unsigned long long abs_relax;
  unsigned long long abs_candidates;
  unsigned long long abs_rcl;
  /** `-relu_order smear|abs`, `-split_on mixed` (`relu_select_split`): the number of choices, of those made
   * as the widest because a score could not be computed, of the ReLUs and of the inputs chosen, of the
   * choices with the relaxed derivatives (`-abs_grad relax`), and the sums of the numbers of candidates and
   * of the sizes of the restricted candidate lists. */
  unsigned long long rsel_choices;
  unsigned long long rsel_fallbacks;
  unsigned long long rsel_relus;
  unsigned long long rsel_inputs;
  unsigned long long rsel_relax;
  unsigned long long rsel_candidates;
  unsigned long long rsel_rcl;

  template <class Alloc>
  UnifiedData(const Configuration<Alloc>& config, const Network& net,
    VariableOrder var_order, ValueOrder val_order, MemoryConfig mem_config)
   : root(config, net, var_order, val_order)
   , stop(false)
   , mem_config(mem_config)
   , root_sym_rounds(0)
   , cand_nodes(0)
   , cand_points(0)
   , cand_found(0)
   , cand_found_mid(0)
   , cand_found_forms(0)
   , cand_found_grad(0)
   , cand_grad_nodes(0)
   , bwd_passes(0)
   , bwd_rows(0)
   , bwd_macs(0)
   , scip_queue()
   , scip_sent(0)
   , scip_dropped(0)
   , race_pruned(0)
   , race_cancelled(0)
   , race_recorded(0)
   , race_lost(0)
   , abs_choices(0)
   , abs_fallbacks(0)
   , abs_relax(0)
   , abs_candidates(0)
   , abs_rcl(0)
   , rsel_choices(0)
   , rsel_fallbacks(0)
   , rsel_relus(0)
   , rsel_inputs(0)
   , rsel_relax(0)
   , rsel_candidates(0)
   , rsel_rcl(0)
  {
    size_t num_subproblems = 1;
    num_subproblems <<= root.config.subproblems_power;
    root.stats.eps_num_subproblems = num_subproblems;
  }
};

/** The largest number of neurons of a layer (the scratch of `-jet_backward`). */
template <class Net>
CUDA INLINE int max_layer_width(const Net& net) {
  int w = 0;
  for(int l = 0; l < net.num_layers(); ++l) { w = battery::max(w, net.layers[l].size); }
  return w;
}

struct GridData;
using FStore = VStore<FItv, bt::pool_allocator>;
/** The abstract domain of a block: its own store of neurons and the ReLU propagator. */
using FProp = FastNNRelu<bt::pool_allocator, ConcurrentAllocator>;
using bound_type = typename FItv::LB::value_type;
using UB = FUB<bound_type, bt::atomic_memory_grid>;
using strategies_type = bt::vector<StrategyType<bt::global_allocator>, bt::global_allocator>;

/** Data private to a single block. */
struct BlockData {
  /** The store of variables at the root of the current subproblem. */
  abstract_ptr<VStore<FItv, bt::global_allocator>> root_store;

  // inner box
  abstract_ptr<VStore<FItv, bt::global_allocator>> inner_box;

  /** The current store of variables.
   * We use a `pool_allocator`, this allows to easily switch between global memory and shared memory, if the store of variables can fit inside.
   * */
  abstract_ptr<FStore> store;

  /** The propagators implemented as an array of bytecodes.
   * Similarly, the propagators can be stored in the global or shared memory.
   * If the propagators do not fit in shared memory, the array of propagators is shared among all blocks.
   * It is possible because the propagators are state-less, we avoid duplicating them in each block.
   * */
  abstract_ptr<FProp> iprop;

  /** The statistics of the current block. */
  Statistics<bt::global_allocator> stats;

  /** The path from the root of the disjunct `disjunct` to the current subproblem `root_store`
   * (`jet_subproblems.hpp`). */
  size_t subproblem_idx;

  /** The disjunct of the current subproblem, an index into the disjuncts of the launch
   * (`NNRoot::stores`, `NNRoot::props`). */
  int disjunct;

  /** The disjunct (of the launch) of the solution saved in `inner_box`, when `stats.solutions > 0`. */
  int solution_disjunct;

  /** The current strategy being used to split the store.
   * It is an index into `GridData::strategies`.
   */
  int current_strategy;

  /** The next unassigned variable in the current strategy.
   * It is an index into `GridData::strategies.vars`.
   */
  int next_unassigned_var;

  /** On backtracking, the value to restore `current_strategy`. */
  int snapshot_root_strategy;

  /** On backtracking, the value to restore `next_unassigned_var`. */
  int snapshot_next_unassigned_var;

  /** The decision taken when exploring the tree. */
  bt::vector<LightBranch<FItv>, bt::global_allocator> decisions;

  /** Current depth of the search tree. */
  int depth;

  /** A timer used for computing time statistics. */
  cuda::std::chrono::system_clock::time_point timer;

  /** A timer used for computing diving VS search time statistics. */
  cuda::std::chrono::system_clock::time_point dive_timer;

  /** The time at which the kernel was started, useful to compute the time of the best bound. */
  cuda::std::chrono::system_clock::time_point start_time;

  /** The depth of the first decision of the current path that underapproximates (fixes a variable to
   * a point), -1 if there is none: a failure below it is counted in `unknowns` (see `uass.hpp`). */
  int uass_depth;

  /** `-under scip`: the depth of the node of the current path sent to SCIP (`scip_offer`), -1 if none:
   * its descendants are not sent. Maintained as `uass_depth`. */
  int scip_depth;
  /** `-scip_mode race`: the ticket of the node sent (`ScipQueueView`), the number of unknown leaves and the
   * exhaustiveness of the block when it was sent (restored when SCIP closes it), and the number of unknown
   * leaves at the start of the current node. */
  unsigned long long scip_ticket;
  size_t scip_unknowns_at_send;
  bool scip_exhaustive_at_send;
  size_t unknowns_node_start;

  /** The symbolic forms of the neurons (`-jet_symbolic`), see `jet_symbolic.hpp`; empty otherwise. */
  bt::vector<bound_type, bt::global_allocator> sym_forms;
  /** `-jet_candidates` and `-jet_leaf_check strict`: the evaluation of a candidate point in `double`
   * (`candidates::evaluate`). */
  bt::vector<double, bt::global_allocator> cand_lo;
  bt::vector<double, bt::global_allocator> cand_hi;
  /** `-jet_cand_grad`: the scratch of the interval Jacobian of a corner (`glo`, `ghi`, `num_neurons` values
   * each). */
  bt::vector<bound_type, bt::global_allocator> cand_grad;
  /** `-jet_backward`: the scratch of the backward contractor (`backward::Scratch`, `jet_backward.hpp`). */
  bt::vector<bound_type, bt::global_allocator> bwd_values;
  bt::vector<int, bt::global_allocator> bwd_ints;
  /** Whether `sym_forms` are those of the current node (it ran a symbolic round), for `-abs_grad relax`. */
  bool forms_current;
  /** Whether the block dives to its subproblem (its decisions are not recorded), for `-abs_dive_count`. */
  bool diving;
  /** `-jet_var_select` (`jet_abs.hpp`): the smear of the inputs (`num_inputs`) followed, with `smear` and
   * `abs`, by the scratch of the interval Jacobian (`2 * num_neurons` values), and the number of bisections
   * of each input on the branch. */
  bt::vector<bound_type, bt::global_allocator> abs_values;
  bt::vector<int, bt::global_allocator> abs_counts;
  /** Whether the backward contractor ran on the current node (its bounds of the pre-activations are those
   * of the node), for `relu_select_split`. */
  bool bwd_current;
  /** `-relu_order smear|abs`, `-split_on mixed` with `-jet_symbolic`: the bounds of the pre-activations
   * computed by the last symbolic round (`symbolic::pass`, `pre`), `2 * num_neurons` values. */
  bt::vector<bound_type, bt::global_allocator> sym_pre;
  /** `-relu_order smear|abs`, `-split_on mixed` (`relu_select_split`): the smear of the inputs (`n0`), the
   * scratch of the reverse pass (`glo`, `ghi`), the derivatives with respect to the neurons (`gy`), the
   * bounds of the pre-activations (`plb`, `pub`), each `num_neurons` values, and the scores (`n0 +
   * num_neurons`); the counts of the variables (`n0 + num_neurons`). */
  bt::vector<bound_type, bt::global_allocator> rsel_values;
  bt::vector<int, bt::global_allocator> rsel_counts;

  __device__ BlockData()
   : subproblem_idx(0)
   , disjunct(0)
   , solution_disjunct(0)
   , current_strategy(0)
   , next_unassigned_var(0)
   , decisions(5000)
   , depth(0)
   , uass_depth(-1)
   , scip_depth(-1)
   , scip_ticket(0)
   , scip_unknowns_at_send(0)
   , scip_exhaustive_at_send(true)
   , unknowns_node_start(0)
   , forms_current(false)
   , diving(false)
   , bwd_current(false)
  {}

  __device__ void allocate(const UnifiedData& unified_data, const GridData& grid_data, unsigned char* shared_mem) {
    if(threadIdx.x == 0) {
      subproblem_idx = 0;
      disjunct = 0;
      const MemoryConfig& mem_config = unified_data.mem_config;
      const auto& u_store = *(unified_data.root.store);
      bt::pool_allocator shared_mem_pool(mem_config.make_shared_pool(shared_mem));
      bt::pool_allocator store_allocator(mem_config.make_store_pool(shared_mem_pool));
      bt::pool_allocator prop_allocator(mem_config.make_prop_pool(shared_mem_pool));
      root_store = bt::make_shared<VStore<FItv, bt::global_allocator>, bt::global_allocator>(u_store);
      inner_box = bt::make_shared<VStore<FItv, bt::global_allocator>, bt::global_allocator>(u_store);
      store = bt::allocate_shared<FStore, bt::pool_allocator>(store_allocator, u_store, store_allocator);
      /** The propagator is stateless apart from the store, so it only holds a pointer to the
       * network, which stays in managed memory and is shared by all the blocks. */
      iprop = bt::allocate_shared<FProp, bt::pool_allocator>(prop_allocator,
        &(unified_data.root.net), &(unified_data.root.props[0]), store,
        unified_data.root.config.jet_affine,
        /** `-forward_only` or `-jet_interval_prop forward`: the interval deductions project forward only. */
        unified_data.root.config.forward_only || unified_data.root.config.jet_interval_prop == JET_INTERVAL_FORWARD);
      if(unified_data.root.config.jet_symbolic) {
        sym_forms.resize(symbolic::storage_size_rows(unified_data.root.net.num_form_rows,
          unified_data.root.net.num_inputs()));
      }
      if(unified_data.root.config.jet_candidates && unified_data.root.config.jet_cand_grad) {
        cand_grad.resize(size_t{2} * static_cast<size_t>(unified_data.root.net.num_neurons));
      }
      if(unified_data.root.config.jet_candidates || unified_data.root.config.jet_leaf_check == JET_LEAF_STRICT) {
        cand_lo.resize(unified_data.root.net.num_neurons);
        cand_hi.resize(unified_data.root.net.num_neurons);
      }
      if(unified_data.root.config.jet_backward != JET_BACKWARD_OFF) {
        const auto& net = unified_data.root.net;
        const int rows = backward::rows_per_group(unified_data.root.config.jet_bwd_rows);
        bwd_values.resize(backward::scratch_values(net.num_neurons, rows));
        bwd_ints.resize(backward::scratch_ints(net.num_layers(), max_layer_width(net), rows));
      }
      if(unified_data.root.config.jet_var_select != JET_VAR_DEFAULT) {
        const auto& net = unified_data.root.net;
        const int vs = unified_data.root.config.jet_var_select;
        abs_values.resize((vs == JET_VAR_SMEAR || vs == JET_VAR_ABS ? size_t{2} * net.num_neurons : size_t{0}) + net.num_inputs());
        abs_counts.resize(net.num_inputs());
      }
      if(unified_data.root.config.jet_relu_select()) {
        const auto& net = unified_data.root.net;
        const size_t n0 = static_cast<size_t>(net.num_inputs()), N = static_cast<size_t>(net.num_neurons);
        rsel_values.resize(size_t{2} * n0 + size_t{6} * N);
        rsel_counts.resize(n0 + N);
        if(unified_data.root.config.jet_symbolic) { sym_pre.resize(size_t{2} * N); }
      }
      forms_current = false;
      bwd_current = false;
      uass::reset(*this);
    }
  }

  /** Switches the block to the disjunct `d` of the launch: its comparisons become those of the
   * propagator. By the thread 0, before the root of the subproblem is copied (`root_store_of`). */
  __device__ INLINE void select_disjunct(const UnifiedData& unified_data, int d) {
    disjunct = d;
    iprop->prop = &(unified_data.root.props[d]);
  }

  /** We must deallocate store and iprop inside the kernel because they might be initialized in shared memory. */
  __device__ void deallocate_shared_data() {
    if(threadIdx.x == 0) {
      // NOTE: .reset() does not work because it does not reset the allocator, which is itself allocated in global memory.
      store = abstract_ptr<FStore>();
      iprop = abstract_ptr<FProp>();
    }
  }

  /** Add a new decision on the `decisions` stack and increase depth.
   * \param has_changed: A Boolean in shared memory.
   * \param strategies: A sequence of strategies.
   * \precondition: We must not be on a leaf node.
   */
  __device__ INLINE void split(bool& has_changed, const strategies_type& strategies, const float epsilon) {
    using LB2 = typename FItv::LB::local_type;
    using UB2 = typename FItv::UB::local_type;
    __shared__ local::ZUB idx;
    decisions[depth].var = AVar{};
    int currentDepth = depth;
    for(int i = current_strategy; i < strategies.size(); ++i) {
      switch(strategies[i].var_order) {
        case VariableOrder::RANDOM:
        case VariableOrder::INPUT_ORDER: {
          input_order_split(has_changed, idx, strategies[i], epsilon);
          break;
        }
        case VariableOrder::FIRST_FAIL: {
          lattice_smallest_split(has_changed, idx, strategies[i], epsilon,
            [&](const FItv& u, int g_idx) { return UB2(u.width().ub().value()); });
          break;
        }
        case VariableOrder::ANTI_FIRST_FAIL: {
          lattice_smallest_split(has_changed, idx, strategies[i], epsilon,
            [&](const FItv& u, int g_idx) { return LB2(u.width().ub().value()); });
          break;
        }
        case VariableOrder::GRA_ANTI_FIRST_FAIL: {
          lattice_smallest_split(has_changed, idx, strategies[i], epsilon,
            [&](const FItv& u, int g_idx) { return LB2(u.width().ub().value()); });
          break;
        }
        case VariableOrder::LARGEST: {
          lattice_smallest_split(has_changed, idx, strategies[i], epsilon,
            [&](const FItv& u, int g_idx) { return LB2(u.ub().value()); });
          break;
        }
        case VariableOrder::SMALLEST: {
          lattice_smallest_split(has_changed, idx, strategies[i], epsilon,
            [&](const FItv& u, int g_idx) { return UB2(u.lb().value()); });
          break;
        }
        default: assert(false);
      }
      __syncthreads();
      // If we could find a variable with the current strategy, we return.
      if(!decisions[currentDepth].var.is_untyped()) {
        return;
      }
      if(threadIdx.x == 0) {
        current_strategy++;
        next_unassigned_var = 0;
      }
    }
    // `input_order_split` and `lattice_smallest_split` have a `__syncthreads()` before reading next_unassigned_var.
  }

  /** Select the next unassigned variable with a finite interval in the array `strategy.vars()` or `store` if the previous one is empty.
   * We ignore infinite variables as splitting on them do not guarantee termination.
   * \param has_changed is a Boolean in shared memory.
   * \param idx is a decreasing integer in shared memory.
   */
  __device__ INLINE void input_order_split(bool& has_changed, local::ZUB& idx, const StrategyType<bt::global_allocator>& strategy, const float epsilon)
  {
    bool split_in_store = strategy.vars.empty();
    int n = split_in_store ? store->vars() : strategy.vars.size();
    if(threadIdx.x == 0) {
      has_changed = true;
      idx = n;
    }
    __syncthreads();
    while(has_changed) {
      __syncthreads();
      // int n = idx.value();
      if(threadIdx.x == 0) {
        has_changed = false;
      }
      __syncthreads();
      for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
        const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
        if(uass::splittable(dom, epsilon)) {
          if(idx.meet(local::ZUB(split_in_store ? i : strategy.vars[i].vid()))) {
            has_changed = true;
          }
        }
      }
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      next_unassigned_var = idx.value();
      if(next_unassigned_var != n) {
        push_decision(strategy.val_order, split_in_store ? AVar{store->aty(), next_unassigned_var} : strategy.vars[next_unassigned_var], epsilon);
      }
    }
  }

  /** Given an array of variable, select the variable `x` with the smallest value `f(store[x])` where "smallest" is defined according to the lattice order of the return type of `f`.
   * \param has_changed is a Boolean in shared memory.
   * \param idx is a decreasing integer in shared memory.
   * */
  template <class F>
  __device__ INLINE void lattice_smallest_split(bool& has_changed, local::ZUB& idx,
    const StrategyType<bt::global_allocator>& strategy, const float epsilon, F f)
  {
    using T = decltype(f(FItv{},0));
    __shared__ T value;
    bool split_in_store = strategy.vars.empty();
    int n = split_in_store ? store->vars() : strategy.vars.size();
    __syncthreads();
    if(threadIdx.x == 0) {
      has_changed = true;
      value = T::top();
      idx = n;
    }
    __syncthreads();
    /** This fixpoint loop seeks for the smallest `x` according to `f(x)` and the next unassigned variable. */
    while(has_changed) {
      __syncthreads();
      if(threadIdx.x == 0) {
        has_changed = false;
      }
      __syncthreads();
      for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
        const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
        if(uass::splittable(dom, epsilon)) {
        //if (dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top()) {
          if(value.meet(f(dom, strategy.vars[i].vid()))) {
            has_changed = true;
          }
          if(idx.meet(local::ZUB(strategy.vars[i].vid()))) {
            has_changed = true;
          }
        }
      }
      __syncthreads();
    }
    /** If we found a value, we traverse again the variables' array to find its index. */
    if(!value.is_top()) {
      __syncthreads();
      if(threadIdx.x == 0) {
        next_unassigned_var = idx.value();
        idx = n;
        has_changed = true;
      }
      __syncthreads();
      // This fixpoint loop is not strictly necessary.
      // We keep it for determinism: the variable with the smallest index is selected first.
      while(has_changed) {
        // int n = idx.value();
        __syncthreads();
        has_changed = false;
        __syncthreads();
        for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
          const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
          if(uass::splittable(dom, epsilon) && f(dom,strategy.vars[i].vid()) == value) {
            if(idx.meet(local::ZUB(strategy.vars[i].vid()))) {
              has_changed = true;
            }
          }
        }
        __syncthreads();
      }
      assert(idx.value() < n);
      if(threadIdx.x == 0) {
        if(split_in_store) {
          push_decision(strategy.val_order, AVar{store->aty(), idx.value()}, epsilon);
        }
        else {
          push_decision(strategy.val_order, strategy.vars[idx.value()], epsilon);
        }
      }
      return;
    }

    /** `SPLIT` on a box of width at most `epsilon` (no variable of the strategy is wider): the
     * underapproximation tries the middle of the box, all the variables at once (`uass.hpp`).
     *   - The middles are written in the store, not in the decisions: the decision has a single child,
     *     the current domain of a variable, so that the search never comes back to this node for
     *     another child. With the two children of `push_decision(SPLIT)`, both points `[m0, m0]`, the
     *     right child was explored with the store restored from the decisions only, that is with
     *     `x0 = m0` alone (the other middles are lost): its failure proved the box empty (unsound
     *     `unsat`), and otherwise the same split was repeated forever.
     *   - The decision marks the node as underapproximated (`push_decision(SPLIT)` reset `is_uass`, so
     *     a failure of the middle was a proof).
     *   - All the variables are fixed, from the first one: those before `next_unassigned_var` are not
     *     split any more (their width is at most `epsilon`), but are not points either, and the node is
     *     a solution only when all of them are points.
     * A variable without finite bounds has no middle and is left as it is: the node is then not a
     * solution, and the next split finds nothing to branch on (the search is not exhaustive). */
    if(strategy.val_order == ValueOrder::SPLIT){
      if(threadIdx.x == 0){
        has_changed = false;
      }
      __syncthreads();
      for(int i = threadIdx.x; i < n; i += blockDim.x){
        const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
        if(uass::in_epsilon_box(dom, epsilon)){
          has_changed = true;
        }
      }
      __syncthreads();
      if(has_changed) {
        for(int i = threadIdx.x; i < n; i += blockDim.x){
          AVar var = split_in_store ? AVar{store->aty(), i} : strategy.vars[i];
          const auto& dom = (*store)[var.vid()];
          if(dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top()) {
            auto mid = battery::midpoint(dom.lb().value(), dom.ub().value());
            store->embed(var, FItv(mid, mid));
          }
        }
        __syncthreads();
        if(threadIdx.x == 0){
          AVar var = split_in_store ? AVar{store->aty(), 0} : strategy.vars[0];
          const auto& dom = (*store)[var.vid()];
          uass::push_point_decision(*this, var, FItv(dom.lb().value(), dom.ub().value()));
        }
      }
      return;
    }

    /** The underapproximation of the other value orders (`push_decision`): one variable of width at
     * most `epsilon`, chosen by the variable order among all the variables of the strategy (from the
     * first one, see `SPLIT` above). The decision is marked as an underapproximation whatever the value
     * order (e.g. `REVERSE_SPLIT`, which splits the domain in two), as before. */
    if(threadIdx.x == 0){
      has_changed = true;
    }
    __syncthreads();
    while(has_changed) {
      __syncthreads();
      if(threadIdx.x == 0){
        has_changed = false;
      }
      __syncthreads();
      for(int i = threadIdx.x; i < n; i += blockDim.x) {
        const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
        if(uass::in_epsilon_box(dom, epsilon)){
          if(value.meet(f(dom, strategy.vars[i].vid()))){
            has_changed = true;
          }
          if(idx.meet(local::ZUB(strategy.vars[i].vid()))){
            has_changed = true;
          }
        }
      }
      __syncthreads();
    }
    if(!value.is_top()) {
      __syncthreads();
      if(threadIdx.x == 0){
        next_unassigned_var = idx.value();
        idx = n;
        has_changed = true;
        uass::mark(*this);
      }
      __syncthreads();
      while(has_changed) {
        __syncthreads();
        has_changed = false;
        __syncthreads();
        for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
          const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
          if(uass::in_epsilon_box(dom, epsilon) && f(dom, strategy.vars[i].vid()) == value){
            if(idx.meet(local::ZUB(strategy.vars[i].vid()))){
              has_changed = true;
            }
          }
        }
        __syncthreads();
      }
      assert(idx.value() < n);
      if(threadIdx.x == 0){
        if(split_in_store){
          push_decision(strategy.val_order, AVar{store->aty(), idx.value()}, epsilon);
        }
        else{
          push_decision(strategy.val_order, strategy.vars[idx.value()], epsilon);
        }
      }
    }
  }

  /** Push a new decision onto the decisions stack.
   *  \precondition The domain of the variable `var` must not be empty, be a singleton or contain infinite bounds.
   *  \precondition Must be executed by thread 0 only.
  */
  __device__ INLINE void push_decision(ValueOrder val_order, AVar var, const float epsilon) {
    assert(threadIdx.x == 0);
    decisions[depth].var = var;
    decisions[depth].current_idx = -1;
    const auto& dom = store->project(decisions[depth].var);
    // printf("split on %d \n", decisions[depth].var.vid());
    // assert(dom.width().ub().value() > epsilon);
    assert(dom.width().ub().value() > 0.0);
    // auto mid = battery::add_down(dom.lb().value(), battery::div_down(battery::sub_up(dom.ub().value(), dom.lb().value()), bound_type{2.0}));
    // bound_type width = battery::sub_up(dom.ub().value(), dom.lb().value());
    // bound_type half = battery::div_up(width, bound_type{2.0});
    // bound_type mid = battery::add_up(dom.lb().value(), half);
    bound_type mid = battery::midpoint(dom.lb().value(), dom.ub().value());
    /** The underapproximation of `LB_SPLIT`, `UB_SPLIT` and `MID_SPLIT` (a domain of width at most
     * `epsilon`) tries a single point: the decision has only `children[0]`, see the ropes below. */
    bool single_child = false;
    /** Whether the children are points of the domain instead of covering it (`uass.hpp`). */
    bool underapproximates = false;
    // printf("split on %d, lb = %.20f, ub = %.20f, mid = %.20f \n", decisions[depth].var.vid(), dom.lb().value(), dom.ub().value(), mid);

    switch(val_order) {
      case ValueOrder::SPLIT: {
        decisions[depth].children[0] = FItv(dom.lb(), mid);
        decisions[depth].children[1] = FItv(mid, dom.ub());
        break;
      }
      case ValueOrder::REVERSE_SPLIT: {
        decisions[depth].children[0] = FItv(mid, dom.ub());
        decisions[depth].children[1] = FItv(dom.lb(), mid);
        break;
      }
      case ValueOrder::LB_SPLIT: {
        if(uass::in_epsilon_box(dom, epsilon)) {
          underapproximates = true;
          single_child = true;
          decisions[depth].children[0] = FItv(dom.lb().value(), dom.lb().value());
        }
        else {
          decisions[depth].children[0] = FItv(dom.lb(), mid);
          decisions[depth].children[1] = FItv(mid, dom.ub());
        }
	      break;
      }
      case ValueOrder::UB_SPLIT: {
        if(uass::in_epsilon_box(dom, epsilon)) {
          underapproximates = true;
          single_child = true;
          decisions[depth].children[0] = FItv(dom.ub().value(), dom.ub().value());
        }
        else {
          decisions[depth].children[0] = FItv(dom.lb(), mid);
          decisions[depth].children[1] = FItv(mid, dom.ub());
        }
	      break;
      }
      case ValueOrder::MID_SPLIT: {
        if(uass::in_epsilon_box(dom, epsilon)) {
          underapproximates = true;
          single_child = true;
          decisions[depth].children[0] = FItv(mid, mid);
        }
        else {
          decisions[depth].children[0] = FItv(dom.lb(), mid);
          decisions[depth].children[1] = FItv(mid, dom.ub());
        }
	      break;
      }
      case ValueOrder::MID_LB_SPLIT: {
        if(uass::in_epsilon_box(dom, epsilon)) {
          underapproximates = true;
          decisions[depth].children[0] = FItv(mid, mid);
          decisions[depth].children[1] = FItv(dom.lb().value(), dom.lb().value());
        }
        else {
          decisions[depth].children[0] = FItv(dom.lb(), mid);
          decisions[depth].children[1] = FItv(mid, dom.ub());
        }
	      break;
      }
      case ValueOrder::MID_UB_SPLIT: {
        if(uass::in_epsilon_box(dom, epsilon)) {
          underapproximates = true;
          decisions[depth].children[0] = FItv(mid, mid);
          decisions[depth].children[1] = FItv(dom.ub().value(), dom.ub().value());
        }
        else {
          decisions[depth].children[0] = FItv(dom.lb(), mid);
          decisions[depth].children[1] = FItv(mid, dom.ub());
        }
	      break;
      }
      case ValueOrder::MIX_SPLIT: {
        // TODO: not complete yet. The current version doesn't support multiple chil nodes, only binary.
        underapproximates = true;
        decisions[depth].children[0] = FItv(mid, mid);
        decisions[depth].children[1] = FItv(dom.lb().value(), dom.lb().value());
	      break;
        decisions[depth].children[2] = FItv(dom.ub().value(), dom.ub().value());
        decisions[depth].children[3] = FItv(battery::nextafter(dom.lb().value(), 1e38f), battery::nextafter(mid, -1e38f));
        decisions[depth].children[4] = FItv(battery::nextafter(mid, 1e38f), battery::nextafter(dom.ub().value(), -1e38f));
      }
      // ValueOrder::MEDIAN is not possible with interval.
      default: assert(false);
    }
    if(underapproximates) {
      uass::mark(*this);
    }
    /** Ropes are a mechanism for fast backtracking.
     * The rope of a left node is always the depth of the right node (also its depth), because after completing the exploration of the left subtree, we must visit the right subtree (rooted at the current depth).
     * The rope of the right node is inherited from its parent, we set -1 if there is no next node to visit.
     */
    decisions[depth].ropes[0] = depth + 1;
    decisions[depth].ropes[1] = depth > 0 ? decisions[depth-1].ropes[decisions[depth-1].current_idx] : -1;
    /** A decision with a single child: once the left subtree is explored, go where the right child would
     * go after its own subtree, so that the right child, which does not exist, is never visited (the
     * backtracking would otherwise embed whatever `children[1]` holds: the default `[-oo, +oo]`, or the
     * child of an older decision of this slot). The descendants inherit this rope through `ropes[0]`.
     * Such a decision is recognized by `ropes[0] == ropes[1]`, which never holds for two children since
     * `ropes[1]` is at most `depth`. */
    if(single_child) {
      decisions[depth].ropes[0] = decisions[depth].ropes[1];
    }
    ++depth;
    // printf("depth(%d), var = %d, children = [%lf, %lf] | [%lf, %lf], ropes = [%d, %d]\n",
    //   depth, decisions[depth-1].var.vid(),
    //   (bound_type)decisions[depth-1].children[0].lb().value(), (bound_type)decisions[depth-1].children[0].ub().value(),
    //   (bound_type)decisions[depth-1].children[1].lb().value(), (bound_type)decisions[depth-1].children[1].ub().value(),
    //   decisions[depth-1].ropes[0], decisions[depth-1].ropes[1]);
    // Reallocate decisions if needed.
    if(decisions.size() == depth) {
      printf("resize to %d\n", (int)decisions.size() * 2);
      decisions.resize(decisions.size() * 2);
    }
  }
};

/** Data shared among all blocks. */
struct GridData {
  /** The private data of each block. */
  bt::vector<BlockData, bt::global_allocator> blocks;

  /** We generate the subproblems lazily.
   * Suppose we generate `2^3` subproblems, we represent the first subproblem as `000`, the second as `001`, the third as `010`, and so on.
   * A `0` means to turn left in the search tree, and a `1` means to turn right.
   * Incrementing this integer will generate the path of the next subproblem.
   */
  ZLB<size_t, bt::atomic_memory_grid> next_subproblem;

  /** The subproblems of the disjuncts of the launch (`jet_subproblems.hpp`): for each disjunct, the
   * first of its paths not taken yet, and the ticket choosing the disjunct of the next subproblem. They
   * replace `next_subproblem`, which is not used by `-arch jet`. */
  bt::vector<subproblems::counter_t, bt::global_allocator> next_path;
  subproblems::counter_t ticket;

  /** Due to multithreading, we must protect `stdout` when printing.
   * The model of computation in this work is lock-free, but it seems unavoidable for printing.
  */
  cuda::binary_semaphore<cuda::thread_scope_device> print_lock;

  /** A specific strategy is used for the subproblem decomposition during the diving phase. */
  bool has_eps_strategy;

  /** The search strategy is immutable and shared among the blocks. */
  strategies_type search_strategies;

  /** The neurons with a ReLU, in the order of the network, for `-split_on relu` (`relu_split.hpp`). */
  bt::vector<int, bt::global_allocator> relu_neurons;
  /** The layer of each neuron of `relu_neurons` (`-relu_order abs`: the earliest layer first). */
  bt::vector<int, bt::global_allocator> relu_layer;

  __device__ GridData(const NNRoot& root)
   : blocks(root.stats.num_blocks)
   , next_subproblem(root.stats.num_blocks)
   , next_path(root.num_disjuncts(), subproblems::counter_t{0})
   , ticket(0)
   , print_lock(1)
   /** There is no EPS-specific strategy: the diving phase branches on the input neurons like the
    * rest of the search. */
   , has_eps_strategy(false)
   , search_strategies(1)
  {
    /** A single strategy, replacing `SplitStrategy`: we branch on the neurons of the input layer
     * only, all the others being determined by `FastNNRelu::deduce`. */
    bt::vector<AVar, bt::global_allocator> input_vars;
    const int n = root.net.num_inputs();
    for(int i = 0; i < n; ++i) {
      input_vars.push_back(AVar{root.store->aty(), i});
    }
    search_strategies[0] = StrategyType<bt::global_allocator>(
      root.var_order, root.val_order, std::move(input_vars));
    for(int l = 1; l < root.net.num_layers(); ++l) {
      const LayerInfo& L = root.net.layers[l];
      if(L.has_relu != 0) {
        for(int j = 0; j < L.size; ++j) { relu_neurons.push_back(L.base + j); relu_layer.push_back(l); }
      }
    }
  }
};

MemoryConfig configure_gpu_fbarebones(Configuration<battery::standard_allocator>&,
  Statistics<battery::standard_allocator>&, const Network&, int max_comparisons);
__global__ void initialize_global_data(UnifiedData*, bt::unique_ptr<GridData, bt::global_allocator>*);
__global__ void gpu_fbarebones_solve(UnifiedData*, GridData*);
__global__ void gpu_root_symbolic(UnifiedData*, GridData*, int);
__global__ void gpu_root_candidates(UnifiedData*, GridData*, int);
template <class FPEngine>
__device__ INLINE void propagate(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data,
   FPEngine& fp_engine, bool& stop, bool& has_changed, bool& is_leaf_node);
__global__ void reduce_blocks(UnifiedData*, GridData*);
__device__ bool scip_offer(UnifiedData&, BlockData&);
__device__ INLINE bool race_closed(const UnifiedData&, const BlockData&);
__device__ INLINE void race_leave(UnifiedData&, BlockData&);
__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>*);

/** Branch on the current node, by all the threads: on the input neurons (`BlockData::split`), or on
 * the phase of a ReLU with `-split_on relu` (`relu_split.hpp`; `y` is split at `jet_split_point`).
 * Once every ReLU has a phase, `-relu_leaf input` goes on with the input split, and `-relu_leaf point`
 * fixes the inputs to points (`relu_split::fix_inputs`). */
__device__ bool var_select_split(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data);
__device__ bool relu_select_split(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data, bool mixed);

__device__ INLINE void branch(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data, bool& has_changed) {
  const auto& config = unified_data.root.config;
  /** `-split_on mixed`: the inputs and the ReLUs together; when neither can be split, the split of the
   * strategy, which handles the epsilon-box. */
  if(config.split_on == SPLIT_ON_MIXED) {
    if(relu_select_split(unified_data, grid_data, block_data, true)) { return; }
    block_data.split(has_changed, grid_data.search_strategies, config.epsilon);
    return;
  }
  if(config.split_on == SPLIT_ON_RELU) {
    if(config.jet_relu_select()
      ? relu_select_split(unified_data, grid_data, block_data, false)
      : relu_split::split_relu(block_data, grid_data.relu_neurons.data(),
          static_cast<int>(grid_data.relu_neurons.size()), relu_split::jet_split_point<bound_type>(), config.relu_order))
    {
      return;
    }
    if(config.relu_leaf == RELU_LEAF_POINT) {
      relu_split::fix_inputs<BlockData, bound_type>(block_data, has_changed, grid_data.search_strategies[block_data.current_strategy]);
      return;
    }
  }
  /** `-jet_var_select`: the input chosen by `jet_abs.hpp`; when no input can be bisected, the split of the
   * strategy, which handles the epsilon-box. */
  if(config.jet_var_select != JET_VAR_DEFAULT && var_select_split(unified_data, grid_data, block_data)) {
    return;
  }
  block_data.split(has_changed, grid_data.search_strategies, config.epsilon);
}


/** Resolve the branching strategy from the textual options of the configuration.
 * The default mirrors the one of `AbstractDomains::interpret_default_strategy` when `WITH_NNV` is
 * defined: `anti_first_fail` on the input neurons, and `indomain_split` on their interval. */
void resolve_search_strategy(const Configuration<battery::standard_allocator>& config,
  VariableOrder& var_order, ValueOrder& val_order)
{
  var_order = VariableOrder::ANTI_FIRST_FAIL;
  val_order = ValueOrder::SPLIT;
  if(config.var_order != "default") {
    auto o = variable_order_of_string(config.var_order);
    if(o.has_value()) { var_order = *o; }
    else { printf("%% WARNING: unrecognized option `-var_order %s`, using `anti_first_fail`.\n", config.var_order.data()); }
  }
  if(config.value_order != "default") {
    /** The MiniZinc annotations are prefixed by `indomain_`, the names of `ValueOrder` are not. */
    const char* name = config.value_order.data();
    const char* prefix = "indomain_";
    if(std::strncmp(name, prefix, std::strlen(prefix)) == 0) { name += std::strlen(prefix); }
    auto o = value_order_of_string(battery::string<battery::standard_allocator>(name));
    if(o.has_value()) { val_order = *o; }
    else { printf("%% WARNING: unrecognized option `-value_order %s`, using `split`.\n", config.value_order.data()); }
  }
}

/** Print the bounds of the input and output neurons of `box`, which replaces
 * `AbstractDomains::print_solution` (there is no `VarEnv` any more, so we print the neurons by
 * their index in the layers). */
void print_neurons(const Network& net, const NNStore& box) {
  printf("%% input neurons:\n");
  for(int i = 0; i < net.num_inputs(); ++i) {
    printf("%% X_%d = [%.10f, %.10f]\n", i, (double)box[i].lb().value(), (double)box[i].ub().value());
  }
  const int out_base = net.output_base();
  printf("%% output neurons:\n");
  for(int i = out_base; i < net.num_neurons; ++i) {
    printf("%% Y_%d = [%.10f, %.10f]\n", i - out_base, (double)box[i].lb().value(), (double)box[i].ub().value());
  }
}


void fbarebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {
  if(config.print_intermediate_solutions) {
    printf("%% WARNING: -arch fbarebones is incompatible with -i and -a (it cannot print intermediate solutions).\n");
  }
  auto start = std::chrono::steady_clock::now();
  check_support_managed_memory();
  check_support_concurrent_managed_memory();

  /** I. The network is the abstract domain: we parse it directly, without building any constraint,
   * hence without ternarization and without the propagators of `pir.hpp`. */
  Network net = parse_network(config);
  net.print();
  if(net.empty()) {
    printf("%% ERROR: the network could not be parsed, or has no hidden layer.\n");
    return;
  }

  /** II. The configuration is mutated by `configure_gpu_fbarebones` (number of subproblems), and
   * the statistics of the host are only used to carry the number of blocks and the memory
   * statistics; the statistics of the solving are those of `unified_data->root`. */
  Configuration<battery::standard_allocator> hconfig(config);
  /** `-jet_linear`: which linear-relaxation contractors may run, the forward symbolic pass
   * (`-jet_symbolic`) and the backward contractor (`-jet_backward`). */
  {
    const int n0 = net.num_inputs();
    bool allow_forward = true;
    bool allow_backward = true;
    if(hconfig.jet_linear == JET_LINEAR_FORWARD) { allow_backward = false; }
    else if(hconfig.jet_linear == JET_LINEAR_BACKWARD) { allow_forward = false; }
    else if(hconfig.jet_linear == JET_LINEAR_AUTO) {
      allow_forward = n0 <= hconfig.jet_linear_n0;
      allow_backward = !allow_forward;
    }
    if(!allow_forward && hconfig.jet_symbolic) {
      printf("%% -jet_linear %s: the network has %d inputs, the symbolic pass (-jet_symbolic) is disabled.\n", name_of_jet_linear(hconfig.jet_linear), n0);
      hconfig.jet_symbolic = false;
    }
    if(!allow_backward && hconfig.jet_backward != JET_BACKWARD_OFF) {
      printf("%% -jet_linear %s: the network has %d inputs, the backward contractor (-jet_backward) is disabled.\n", name_of_jet_linear(hconfig.jet_linear), n0);
      hconfig.jet_backward = JET_BACKWARD_OFF;
    }
    if(hconfig.jet_backward != JET_BACKWARD_OFF && hconfig.jet_sym_skip_final_interval) {
      printf("%% WARNING: -jet_sym_skip_final_interval is not used with -jet_backward (a round that changed the store is always followed by the interval fixpoint).\n");
    }
    if(hconfig.jet_backward == JET_BACKWARD_OFF && (hconfig.jet_bwd_clip || hconfig.jet_bwd_et)) {
      printf("%% WARNING: -jet_bwd_clip and -jet_bwd_et need -jet_backward, they have no effect.\n");
    }
  }
  /** `-under scip`: the network must have an exact MIP encoding (no sigmoid). The candidate points of JET
   * (`-jet_candidates`) are tested first, and a node is sent to SCIP only when they found nothing. */
  if(hconfig.under == UNDER_SCIP) {
    std::string why;
    if(!scip_under::supported(net, why)) {
      printf("%% WARNING: -under scip: %s, which has no exact MIP encoding; the underapproximation of JET is used (-under jet).\n", why.c_str());
      hconfig.under = UNDER_JET;
    }
  }
  if(hconfig.jet_sym_skip_final_interval && (!hconfig.jet_symbolic || hconfig.jet_sym_rounds <= 0)) {
    printf("%% WARNING: -jet_sym_skip_final_interval needs -jet_symbolic and -jet_sym_rounds k > 0 (with -jet_sym_rounds 0, the last symbolic round changes nothing), it has no effect.\n");
  }
  Statistics<battery::standard_allocator> hstats(
    static_cast<size_t>(net.num_neurons), static_cast<size_t>(net.num_deductions()),
    false, config.print_statistics);

  VariableOrder var_order;
  ValueOrder val_order;
  resolve_search_strategy(hconfig, var_order, val_order);
  if(hconfig.split_on == SPLIT_ON_MIXED) {
    int num_relus = 0;
    for(int l = 1; l < net.num_layers(); ++l) {
      if(net.layers[l].has_relu != 0) { num_relus += net.layers[l].size; }
    }
    if(hconfig.jet_var_select != JET_VAR_SMEAR && hconfig.jet_var_select != JET_VAR_ABS) {
      printf("%% WARNING: -split_on mixed chooses with -jet_var_select smear or abs: -jet_var_select %s is replaced by abs.\n",
        name_of_jet_var_select(hconfig.jet_var_select));
      hconfig.jet_var_select = JET_VAR_ABS;
    }
    if(hconfig.verbose_solving >= 1) {
      printf("%% -split_on mixed: the search splits the %d inputs and the phase of the %d ReLU neurons, chosen by -jet_var_select %s (-relu_score %s, -abs_grad %s).\n",
        net.num_inputs(), num_relus, name_of_jet_var_select(hconfig.jet_var_select),
        hconfig.relu_score == RELU_SCORE_GAP ? "gap" : "smear", hconfig.abs_grad == ABS_GRAD_RELAX ? "relax" : "interval");
    }
  }
  if(hconfig.split_on == SPLIT_ON_RELU) {
    int num_relus = 0;
    for(int l = 1; l < net.num_layers(); ++l) {
      if(net.layers[l].has_relu != 0) { num_relus += net.layers[l].size; }
    }
    if(num_relus == 0) {
      printf("%% WARNING: -split_on relu: the network has no ReLU, the search splits the inputs.\n");
      hconfig.split_on = SPLIT_ON_INPUT;
    }
    else {
      const bool point = hconfig.relu_leaf == RELU_LEAF_POINT;
      if(point && !relu_split::is_leaf_value_order(val_order)) {
        printf("%% WARNING: -split_on relu -relu_leaf point: the value order %s does not fix the inputs of the leaves, they are fixed as with split (all at their middle).\n",
          string_of_value_order(val_order));
      }
      if(hconfig.verbose_solving >= 1) {
        printf("%% -split_on relu: the search splits the phase of %d ReLU neurons (-relu_order %s), then %s the inputs with %s and %s.\n",
          num_relus, name_of_relu_order(hconfig.relu_order),
          point ? "fixes" : "splits (-relu_leaf input)",
          string_of_variable_order(var_order), string_of_value_order(val_order));
      }
    }
  }

  /** III. The property. A disjunctive property (the `acasxu` output constraints, the robustness
   * properties of `cora_2024`, and the union of two input boxes of `prop_6`/`prop_8`) is expanded into
   * DNF: each disjunct is one box plus one conjunction of comparisons, the root of one search tree. The
   * trees are searched together (`-jet_disjuncts interleaved`, `jet_subproblems.hpp`) or one after the
   * other (`sequential`). The answer is `sat` as soon as one disjunct is `sat`, and `unsat` only when
   * all of them are. */
  DNF dnf;
  if(!load_property(hconfig, net, dnf)) {
    printf("%% ERROR: the property could not be loaded, aborting.\n");
    return;
  }
  /** The comparisons of a disjunct are deductions like the neurons, so they count in the size of
   * the fixpoint engine. We reserve for the largest disjunct. */
  int max_comparisons = 0;
  for(const Conjunction& c : dnf) {
    int n = 0;
    for(const PropAtom& a : c) { n += a.is_binary ? 1 : 0; }
    max_comparisons = std::max(max_comparisons, n);
  }

  MemoryConfig mem_config = configure_gpu_fbarebones(hconfig, hstats, net, max_comparisons);

  auto unified_data = bt::make_unique<UnifiedData, ConcurrentAllocator>(
    hconfig, net, var_order, val_order, mem_config);
  unified_data->root.stats.num_blocks = hstats.num_blocks;

  auto& uroot = unified_data->root;
  bool interrupted = false;
  int64_t time_to_kernel_start = 0;
  battery::vector<NeuronLeq> leqs;

  /** `-under scip` (`jet_scip.hpp`): the queue in pinned memory, the SCIP workers, and the dispatcher
   * thread moving the ready slots of the queue to the workers. A counterexample of SCIP stops the
   * kernel (`stop`) and is installed in `uroot.best` after the launches. */
  ScipQueueView scip_host_view;
  void* scip_pinned = nullptr;
#ifdef JET_WITH_SCIP
  std::unique_ptr<scip_under::Manager> scip_manager;
  std::thread scip_dispatcher;
  std::atomic<bool> scip_dispatch_stop{false};
  std::atomic<int64_t> scip_found_ns{0};
  /** `-scip_mode race`: the answer of every node solved, by ticket, for the end of the solving. */
  std::unordered_map<unsigned long long, int> scip_results;
  std::mutex scip_results_mutex;
  /** Moves the ready slots of the queue to the workers; returns the number of slots moved. At most
   * `capacity` nodes wait for a worker: beyond, the slots stay ready, and the blocks drop their new nodes
   * (the host memory is bounded, and the nodes solved are the oldest ones). */
  auto scip_dispatch_once = [&]() {
    int moved = 0;
    const int n = scip_host_view.n;
    for(int s = 0; s < scip_host_view.capacity; ++s) {
      if(scip_manager->pending() >= static_cast<size_t>(scip_host_view.capacity)) { break; }
      int* state = &scip_host_view.state[s];
      if(__atomic_load_n(state, __ATOMIC_ACQUIRE) != SCIP_SLOT_READY) { continue; }
      scip_under::Job job;
      job.dnf = scip_host_view.dnf[s];
      job.seq = scip_host_view.seq[s];
      const float* src = scip_host_view.bounds + static_cast<size_t>(s) * size_t{2} * static_cast<size_t>(n);
      job.lb.assign(src, src + n);
      job.ub.assign(src + n, src + 2 * n);
      __atomic_store_n(state, SCIP_SLOT_FREE, __ATOMIC_RELEASE);
      scip_manager->submit(std::move(job));
      ++moved;
    }
    return moved;
  };
  if(hconfig.under == UNDER_SCIP) {
    const size_t n = static_cast<size_t>(net.num_neurons);
    const size_t slot_bytes = n * size_t{2} * sizeof(float) + size_t{16};
    const size_t budget = size_t{256} << 20;
    const int capacity = static_cast<int>(std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(hconfig.scip_queue), budget / slot_bytes)));
    if(capacity < hconfig.scip_queue) {
      printf("%% -under scip: the queue has %d slots instead of %d (%zu bytes per node, at most 256 MB).\n", capacity, hconfig.scip_queue, slot_bytes);
    }
    const size_t cap = static_cast<size_t>(capacity);
    /** `-scip_mode race`: the rings of the answers and of the cancellations (`rcap` entries each), and the
     * records of the subtrees left with unknown leaves (`rec_cap` pairs and their counter). */
    const bool race = hconfig.scip_mode == SCIP_MODE_RACE;
    const size_t rcap = race ? size_t{4} * cap : size_t{0};
    const size_t rec_cap = race ? size_t{1} << 16 : size_t{0};
    const size_t race_words = rcap * size_t{2} + rec_cap * size_t{2} + (race ? size_t{1} : size_t{0});
    const size_t bytes = race_words * sizeof(unsigned long long)
      + cap * sizeof(int) * 2 + cap * sizeof(unsigned long long) + sizeof(unsigned long long)
      + cap * n * size_t{2} * sizeof(float) + size_t{64};
    CUDAEX(cudaHostAlloc(&scip_pinned, bytes, cudaHostAllocMapped));
    std::memset(scip_pinned, 0, bytes);
    char* base = static_cast<char*>(scip_pinned);
    /** The 8-byte values first, for their alignment: those of the race, then the queue. */
    if(race) {
      unsigned long long* w = reinterpret_cast<unsigned long long*>(base);
      scip_host_view.result = w;
      scip_host_view.cancel = w + rcap;
      scip_host_view.rec = w + 2 * rcap;
      scip_host_view.rec_count = w + 2 * rcap + 2 * rec_cap;
      scip_host_view.rcap = static_cast<int>(rcap);
      scip_host_view.rec_cap = static_cast<int>(rec_cap);
      base += race_words * sizeof(unsigned long long);
    }
    scip_host_view.head = reinterpret_cast<unsigned long long*>(base);
    scip_host_view.seq = reinterpret_cast<unsigned long long*>(base + sizeof(unsigned long long));
    scip_host_view.state = reinterpret_cast<int*>(base + sizeof(unsigned long long) * (cap + 1));
    scip_host_view.dnf = scip_host_view.state + cap;
    scip_host_view.bounds = reinterpret_cast<float*>(scip_host_view.dnf + cap);
    scip_host_view.capacity = capacity;
    scip_host_view.n = static_cast<int>(n);
    ScipQueueView dev = scip_host_view;
    auto to_device = [&](auto* p) {
      void* d = nullptr;
      CUDAEX(cudaHostGetDevicePointer(&d, static_cast<void*>(p), 0));
      return static_cast<decltype(p)>(d);
    };
    dev.head = to_device(scip_host_view.head);
    dev.seq = to_device(scip_host_view.seq);
    dev.state = to_device(scip_host_view.state);
    dev.dnf = to_device(scip_host_view.dnf);
    dev.bounds = to_device(scip_host_view.bounds);
    if(race) {
      dev.result = to_device(scip_host_view.result);
      dev.cancel = to_device(scip_host_view.cancel);
      dev.rec = to_device(scip_host_view.rec);
      dev.rec_count = to_device(scip_host_view.rec_count);
    }
    unified_data->scip_queue = dev;
    scip_under::Options opt;
    /** The time limit of a node, relative to the timeout of the solving (`-scip_time_ratio`). */
    opt.time_limit = hconfig.scip_time;
    if(hconfig.timeout_ms > 0 && hconfig.scip_time_ratio > 0) {
      const double T = static_cast<double>(hconfig.timeout_ms) / 1000.0;
      opt.time_limit = std::min(hconfig.scip_time_max, std::max(hconfig.scip_time_min, hconfig.scip_time_ratio * T));
    }
    opt.prove = hconfig.scip_mode == SCIP_MODE_RACE;
    opt.exact_cex = hconfig.scip_cex_exact;
    if(hconfig.verbose_solving >= 1) {
      printf("%% -under scip: -scip_mode %s, time limit of a node %g s.\n", name_of_scip_mode(hconfig.scip_mode), opt.time_limit);
    }
    opt.margin = hconfig.scip_margin;
    UnifiedData* ud = unified_data.get();
    scip_manager = std::make_unique<scip_under::Manager>(scip_under::HostNet(net), hconfig.scip_threads, opt,
      [ud, &scip_found_ns, start]() {
        scip_found_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        ud->stop.test_and_set();
      });
    if(race) {
      const ScipQueueView hv = scip_host_view;
      scip_manager->set_on_result([hv, &scip_results, &scip_results_mutex](unsigned long long t, scip_under::Status st) {
        __atomic_store_n(&hv.result[t % static_cast<unsigned long long>(hv.rcap)], ((t + 1ull) << 8) | static_cast<unsigned long long>(st), __ATOMIC_RELEASE);
        std::lock_guard<std::mutex> lock(scip_results_mutex);
        scip_results[t] = static_cast<int>(st);
      });
      scip_manager->set_skip([hv](unsigned long long t) {
        return __atomic_load_n(&hv.cancel[t % static_cast<unsigned long long>(hv.rcap)], __ATOMIC_ACQUIRE) == t + 1ull;
      });
    }
    if(scip_manager->num_workers() == 0) {
      printf("%% WARNING: -under scip: no SCIP worker could be started, the underapproximation of JET is used (-under jet).\n");
      scip_manager.reset();
      uroot.config.under = UNDER_JET;
      uroot.config.jet_candidates = config.jet_candidates;
    }
    else {
      if(hconfig.verbose_solving >= 1) {
        printf("%% -under scip: %d SCIP workers (exact mode), a queue of %d nodes, nodes with at most %d unstable ReLUs.\n",
          scip_manager->num_workers(), capacity, hconfig.scip_max_unstable);
      }
      scip_dispatcher = std::thread([&]() {
        while(!scip_dispatch_stop.load()) {
          if(scip_dispatch_once() == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        }
        scip_dispatch_once();
      });
    }
  }
  auto scip_found = [&]() { return scip_manager && scip_manager->found(); };
  /** Stops the dispatcher and the workers, and frees the queue, on every exit of the function. */
  struct ScipCleanup {
    std::function<void()> f;
    ~ScipCleanup() { f(); }
  } scip_cleanup{[&]() {
    scip_dispatch_stop = true;
    if(scip_dispatcher.joinable()) { scip_dispatcher.join(); }
    if(scip_manager) { scip_manager->shutdown(); }
    if(scip_pinned != nullptr) { cudaFreeHost(scip_pinned); scip_pinned = nullptr; }
  }};
#else
  auto scip_found = [&]() { return false; };
#endif

  /** Block the signal CTRL-C to notify the threads if we must exit. */
  block_signal_ctrlc();

  /** Installs the disjuncts `ids` of the DNF on the root, as the disjuncts of the next launch
   * (`NNRoot::stores`): for each one, a fresh store of neurons, all at top, with the bounds of the
   * disjunct, its comparisons, and with `-jet_candidates` its inner bounds. A disjunct inconsistent
   * before any propagation is left out. Re-allocating is simpler than resetting, and happens at most
   * twice per disjunct (with the pre-pass of the candidates).
   * \return the number of disjuncts installed. */
  auto install_disjuncts = [&](const std::vector<size_t>& ids) {
    uroot.stores.clear();
    uroot.props.clear();
    uroot.cand_stores.clear();
    uroot.cand_props.clear();
    uroot.disjunct_ids.clear();
    for(size_t d : ids) {
      auto st = bt::allocate_shared<NNStore, ConcurrentAllocator>(ConcurrentAllocator{},
        AType{0}, net.num_neurons, ConcurrentAllocator{});
      apply_disjunct(dnf[d], *st, leqs);
      if(st->is_bot()) {
        if(hconfig.verbose_solving >= 1) {
          printf("%% The disjunct %d/%d is inconsistent before any propagation, skipping it.\n", (int)d + 1, (int)dnf.size());
        }
        continue;
      }
      uroot.stores.push_back(st);
      uroot.props.push_back(Property(leqs, ConcurrentAllocator{}));
      /** The disjunct rounded inward, for the strict test (`-jet_candidates`, `-jet_leaf_check strict`). */
      {
        auto inner = bt::allocate_shared<NNStore, ConcurrentAllocator>(ConcurrentAllocator{},
          AType{0}, net.num_neurons, ConcurrentAllocator{});
        battery::vector<NeuronLeq> inner_leqs;
        apply_disjunct_inner(dnf[d], *inner, inner_leqs);
#ifdef JET_WITH_SCIP
        if(scip_manager) {
          scip_under::Region region;
          for(int i = 0; i < net.num_neurons; ++i) {
            region.lb.push_back(static_cast<double>((*inner)[i].lb().value()));
            region.ub.push_back(static_cast<double>((*inner)[i].ub().value()));
          }
          for(int q = 0; q < static_cast<int>(inner_leqs.size()); ++q) {
            region.leqs.push_back(scip_under::Leq{inner_leqs[q].x, inner_leqs[q].y, static_cast<double>(inner_leqs[q].k)});
          }
          /** The region rounded outward (the store and the comparisons of the propagation), for the MIP of
           * the proof mode (`-scip_mode race`). */
          for(int i = 0; i < net.num_neurons; ++i) {
            region.olb.push_back(static_cast<double>((*st)[i].lb().value()));
            region.oub.push_back(static_cast<double>((*st)[i].ub().value()));
          }
          for(int q = 0; q < static_cast<int>(leqs.size()); ++q) {
            region.oleqs.push_back(scip_under::Leq{leqs[q].x, leqs[q].y, static_cast<double>(leqs[q].k)});
          }
          scip_manager->add_region(static_cast<int>(d), region);
        }
#endif
        uroot.cand_stores.push_back(inner);
        uroot.cand_props.push_back(Property(inner_leqs, ConcurrentAllocator{}));
      }
      uroot.disjunct_ids.push_back(static_cast<int>(d));
    }
    /** `-jet_backward`: the bounds of the pre-activations of the roots, unknown until `gpu_root_symbolic`. */
    if(hconfig.jet_backward != JET_BACKWARD_OFF) {
      const size_t N = static_cast<size_t>(net.num_neurons);
      uroot.bwd_root_p.resize(size_t{2} * N * static_cast<size_t>(uroot.num_disjuncts()));
      for(int k = 0; k < uroot.num_disjuncts(); ++k) {
        for(size_t i = 0; i < N; ++i) {
          uroot.bwd_root_p[size_t{2} * N * k + i] = battery::limits<bound_type>::neg_inf();
          uroot.bwd_root_p[size_t{2} * N * k + N + i] = battery::limits<bound_type>::inf();
        }
      }
    }
    /** `-jet_var_select smear|abs`, `-relu_order smear|abs`, `-split_on mixed`: the rows of the Jacobian of each disjunct (`jet_jacobian.hpp`): its
     * comparisons, and its outputs with a finite bound in the region (the inner box). */
    uroot.jac_rows.clear();
    uroot.jac_row_start.clear();
    if(hconfig.jet_uses_jacobian()) {
      const auto& O = net.layers[net.output_layer];
      for(int k = 0; k < uroot.num_disjuncts(); ++k) {
        uroot.jac_row_start.push_back(static_cast<int>(uroot.jac_rows.size()));
        const auto& leqs_k = uroot.cand_props[k].leqs;
        for(int q = 0; q < static_cast<int>(leqs_k.size()); ++q) {
          uroot.jac_rows.push_back(jacobian::Row{leqs_k[q].x, leqs_k[q].y, 1.0f, -1.0f});
        }
        const auto& inner = *uroot.cand_stores[k];
        for(int j = 0; j < O.size; ++j) {
          const int o = O.base + j;
          if(inner[o].lb().value() > battery::limits<bound_type>::neg_inf() || inner[o].ub().value() < battery::limits<bound_type>::inf()) {
            uroot.jac_rows.push_back(jacobian::Row{o, -1, 1.0f, 0.0f});
          }
        }
      }
      uroot.jac_row_start.push_back(static_cast<int>(uroot.jac_rows.size()));
    }
    if(uroot.num_disjuncts() > 0) {
      uroot.store = uroot.stores[0];
      uroot.stats.eps_num_subproblems = static_cast<size_t>(uroot.num_disjuncts()) << hconfig.subproblems_power;
    }
    return uroot.num_disjuncts();
  };
  bool launched = false;
  auto record_kernel_start = [&]() {
    if(!launched) {
      launched = true;
      time_to_kernel_start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    }
  };

  /** The launches of the search (`-jet_disjuncts`): one for all the disjuncts (`interleaved`), whose
   * subproblems are taken in turn by the blocks (`jet_subproblems.hpp`), or one per disjunct
   * (`sequential`), in the order of the property. */
  const bool sequential = hconfig.jet_disjuncts == JET_DISJUNCTS_SEQUENTIAL;
  std::vector<std::vector<size_t>> launches;
  if(sequential) {
    for(size_t d = 0; d < dnf.size(); ++d) { launches.push_back(std::vector<size_t>(1, d)); }
  }
  else {
    launches.emplace_back();
    for(size_t d = 0; d < dnf.size(); ++d) { launches.back().push_back(d); }
  }

  /** `-jet_candidates` with several disjuncts searched one after the other (`sequential`): the
   * candidates of the root of every disjunct are tested first (`gpu_root_candidates`), since the search
   * of a disjunct without counterexample may take all the time. A counterexample ends the solving
   * (`sat`); otherwise the disjuncts are searched as usual (their roots are propagated again). With
   * `interleaved`, the first subproblems of all the disjuncts are taken at once, and their dives test
   * the candidates of the roots (depth 0). */
  if(hconfig.jet_candidates && sequential && dnf.size() > 1) {
    if(mem_config.shared_bytes > 0) {
      CUDAEX(cudaFuncSetAttribute((void*) gpu_root_candidates, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(mem_config.shared_bytes)));
    }
    for(size_t d = 0; d < dnf.size() && !interrupted && uroot.stats.solutions == 0; ++d) {
      if(install_disjuncts(std::vector<size_t>(1, d)) == 0) { continue; }
      auto grid_data = bt::make_unique<bt::unique_ptr<GridData, bt::global_allocator>, ConcurrentAllocator>();
      initialize_global_data<<<1,1>>>(unified_data.get(), grid_data.get());
      CUDAEX(cudaDeviceSynchronize());
      gpu_root_candidates<<<1, CUDA_THREADS_PER_BLOCK, mem_config.shared_bytes>>>(unified_data.get(), grid_data->get(), 0);
      record_kernel_start();
      interrupted = wait_solving_ends(unified_data->stop, uroot, start);
      CUDAEX(cudaDeviceSynchronize());
      deallocate_global_data<<<1,1>>>(grid_data.get());
      CUDAEX(cudaDeviceSynchronize());
      unified_data->stop.clear();
      if(hconfig.verbose_solving >= 1) {
        printf("%% -jet_candidates: root of disjunct %d/%d: %s.\n", (int)d + 1, (int)dnf.size(),
          uroot.stats.solutions > 0 ? "counterexample found" : "no counterexample");
      }
    }
  }

  for(size_t run = 0; run < launches.size() && !interrupted && uroot.stats.solutions == 0 && !scip_found(); ++run) {
    if(install_disjuncts(launches[run]) == 0) { continue; }
    if(hconfig.verbose_solving >= 1 && dnf.size() > 1) {
      if(sequential) {
        printf("%% Solving disjunct %d/%d of the property.\n", uroot.disjunct_ids[0] + 1, (int)dnf.size());
      }
      else {
        printf("%% Solving the %d disjunct(s) of the property together (-jet_disjuncts interleaved, %zu subproblems).\n",
          uroot.num_disjuncts(), uroot.stats.eps_num_subproblems);
      }
    }
    if(hconfig.verbose_solving >= 2) {
      for(int k = 0; k < uroot.num_disjuncts(); ++k) {
        printf("%% The root of disjunct %d:\n", uroot.disjunct_ids[k] + 1);
        print_neurons(net, *(uroot.stores[k]));
      }
    }

    auto grid_data = bt::make_unique<bt::unique_ptr<GridData, bt::global_allocator>, ConcurrentAllocator>();
    initialize_global_data<<<1,1>>>(unified_data.get(), grid_data.get());
    CUDAEX(cudaDeviceSynchronize());
    /** `-jet_sym_root_only` or `-jet_backward`: the symbolic rounds and the backward contractor on the root
     * of each disjunct, once, before the search. The search is launched right after, on the same stream,
     * so it starts from the roots they tightened. */
    const bool root_kernel = (hconfig.jet_symbolic && hconfig.jet_sym_root_only) || hconfig.jet_backward != JET_BACKWARD_OFF;
    if(root_kernel) {
      if(mem_config.shared_bytes > 0) {
        CUDAEX(cudaFuncSetAttribute((void*) gpu_root_symbolic, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(mem_config.shared_bytes)));
      }
      for(int k = 0; k < uroot.num_disjuncts(); ++k) {
        gpu_root_symbolic<<<1, CUDA_THREADS_PER_BLOCK, mem_config.shared_bytes>>>(unified_data.get(), grid_data->get(), k);
      }
      if(hconfig.verbose_solving >= 2) {
        CUDAEX(cudaDeviceSynchronize());
        for(int k = 0; k < uroot.num_disjuncts(); ++k) {
          printf("%% The root of disjunct %d after the symbolic rounds and the backward contractor (-jet_sym_root_only, -jet_backward):\n", uroot.disjunct_ids[k] + 1);
          print_neurons(net, *(uroot.stores[k]));
        }
      }
    }
    /** We wait that either the solving is interrupted, or that all threads have finished. */
    gpu_fbarebones_solve
      <<<static_cast<unsigned int>(uroot.stats.num_blocks),
        CUDA_THREADS_PER_BLOCK,
        mem_config.shared_bytes>>>
      (unified_data.get(), grid_data->get());
    record_kernel_start();
    interrupted = wait_solving_ends(unified_data->stop, uroot, start);
    CUDAEX(cudaDeviceSynchronize());
    reduce_blocks<<<1,1>>>(unified_data.get(), grid_data->get());
    CUDAEX(cudaDeviceSynchronize());
    deallocate_global_data<<<1,1>>>(grid_data.get());
    CUDAEX(cudaDeviceSynchronize());

    /** The blocks stop as soon as a counterexample is found, so the flag must be cleared before the
     * next disjunct is explored. */
    unified_data->stop.clear();
  }

#ifdef JET_WITH_SCIP
  /** `-under scip`: the nodes still in the queue are moved to the workers, and the workers end their nodes
   * unless JET already decided (a counterexample, or an exhaustive search without unknown leaf), until the
   * timeout. Their counterexample, if any, is the answer. */
  if(scip_manager) {
    scip_dispatch_stop = true;
    if(scip_dispatcher.joinable()) { scip_dispatcher.join(); }
    const bool jet_decided = uroot.stats.solutions > 0 || (!interrupted && uroot.stats.exhaustive && uroot.stats.unknowns == 0);
    if(!jet_decided && !scip_manager->found() && !interrupted) {
      if(hconfig.verbose_solving >= 1) {
        printf("%% -under scip: the search of the GPU ended, waiting for the SCIP workers.\n");
      }
      scip_manager->wait_idle([&]() { return !must_quit(uroot) && check_timeout(uroot, start); });
    }
    if(scip_manager->found()) {
      if(uroot.stats.solutions == 0) {
        const std::vector<double> lo = scip_manager->found_lo();
        const std::vector<double> hi = scip_manager->found_hi();
        using local_itv = typename FItv::local_type;
        uroot.best->join_top();
        for(int i = 0; i < net.num_neurons; ++i) {
          uroot.best->embed(i, local_itv(typename local_itv::LB(battery::rd_cast<bound_type>(lo[i])),
            typename local_itv::UB(battery::ru_cast<bound_type>(hi[i]))));
        }
        uroot.found_disjunct = scip_manager->found_dnf();
        uroot.stats.solutions = 1;
        uroot.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND) = scip_found_ns.load() - time_to_kernel_start;
        if(hconfig.verbose_solving >= 1) {
          printf("%% -under scip: the counterexample was found by SCIP.\n");
        }
        /** How the point was taken: exactly, or rounded to floats and checked by the strict test (with
         * `-scip_cex exact`, when the proof mode relaxed a ReLU, or the exact point was outside the inner
         * region). */
        const int path = scip_manager->found_path();
        if(path == scip_under::CEX_RELAXED || path == scip_under::CEX_OUTSIDE) {
          printf("%% -scip_cex exact: the counterexample of SCIP was rounded to floats and checked by the strict test (%s).\n",
            path == scip_under::CEX_RELAXED ? "the proof mode relaxed a ReLU of the node" : "the exact point was outside the inner region");
        }
        /** `-scip_cex exact`: the exact point (the neurons above are its values rounded outward). */
        const std::string ex = scip_manager->found_exact();
        if(!ex.empty()) {
          printf("%% -under scip: exact counterexample (rational inputs, a counterexample of the real-valued network):\n");
          size_t pos = 0;
          for(int k = 0; pos < ex.size(); ++k) {
            const size_t end = ex.find('\n', pos);
            printf("%% X_%d = %s\n", k, ex.substr(pos, end - pos).c_str());
            if(end == std::string::npos) { break; }
            pos = end + 1;
          }
        }
      }
    }
    /** `-scip_mode race`: the unknown leaves of the subtrees that the blocks left before SCIP answered are
     * not counted when SCIP closed their node. */
    if(hconfig.scip_mode == SCIP_MODE_RACE && scip_host_view.rec_count != nullptr && uroot.stats.solutions == 0) {
      const unsigned long long recorded = std::min(__atomic_load_n(scip_host_view.rec_count, __ATOMIC_ACQUIRE),
        static_cast<unsigned long long>(scip_host_view.rec_cap));
      unsigned long long removed = 0, closed = 0;
      {
        std::lock_guard<std::mutex> lock(scip_results_mutex);
        for(unsigned long long i = 0; i < recorded; ++i) {
          const unsigned long long t = scip_host_view.rec[2 * i] - 1ull;
          const auto it = scip_results.find(t);
          if(it != scip_results.end() && it->second == static_cast<int>(scip_under::CLOSED)) {
            removed += scip_host_view.rec[2 * i + 1];
            ++closed;
          }
        }
      }
      removed = std::min(removed, static_cast<unsigned long long>(uroot.stats.unknowns));
      uroot.stats.unknowns -= static_cast<size_t>(removed);
      if(config.print_statistics) {
        printf("%%%%%%nnv-stat: race_subtrees_recorded=%llu\n", recorded);
        printf("%%%%%%nnv-stat: race_subtrees_closed_after=%llu\n", closed);
        printf("%%%%%%nnv-stat: race_unknowns_removed=%llu\n", removed);
      }
    }
    if(scip_manager->found()) {
      if(jet_decided && uroot.stats.solutions == 0) {
        printf("%% WARNING: -under scip: SCIP found a counterexample that passes the strict test, but the search of JET was exhaustive.\n");
      }
    }
    scip_manager->shutdown();
  }
  if(scip_pinned != nullptr) {
    CUDAEX(cudaFreeHost(scip_pinned));
    scip_pinned = nullptr;
  }
#endif
  if(uroot.stats.solutions > 0 && hconfig.verbose_solving >= 1 && dnf.size() > 1 && uroot.found_disjunct >= 0) {
    printf("%% The counterexample satisfies disjunct %d/%d of the property.\n", uroot.found_disjunct + 1, (int)dnf.size());
  }
  if(uroot.stats.solutions > 0) {
    // We add the time before the kernel starts to the time needed to find the best bound.
    uroot.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND) += time_to_kernel_start;
    if(uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) != 0) {
      uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) += time_to_kernel_start;
    }
    print_neurons(net, *(uroot.best));
  }
  uroot.stats.print_mzn_final_separator();
  if(uroot.config.print_statistics) {
    uroot.config.print_mzn_statistics();
    uroot.stats.print_mzn_statistics(uroot.config.verbose_solving);
    if(uroot.config.jet_symbolic && uroot.config.jet_sym_root_only) {
      printf("%%%%%%nnv-stat: jet_sym_root_rounds=%d\n", unified_data->root_sym_rounds);
    }
    if(uroot.config.jet_backward != JET_BACKWARD_OFF) {
      printf("%%%%%%nnv-stat: jet_bwd_passes=%llu\n", unified_data->bwd_passes);
      printf("%%%%%%nnv-stat: jet_bwd_rows=%llu\n", unified_data->bwd_rows);
      printf("%%%%%%nnv-stat: jet_bwd_macs=%llu\n", unified_data->bwd_macs);
    }
#ifdef JET_WITH_SCIP
    if(scip_manager) {
      printf("%%%%%%nnv-stat: scip_queue_slots=%d\n", scip_host_view.capacity);
      printf("%%%%%%nnv-stat: scip_nodes_sent=%llu\n", unified_data->scip_sent);
      printf("%%%%%%nnv-stat: scip_nodes_dropped=%llu\n", unified_data->scip_dropped);
      if(uroot.config.scip_mode == SCIP_MODE_RACE) {
        printf("%%%%%%nnv-stat: race_subtrees_pruned=%llu\n", unified_data->race_pruned);
        printf("%%%%%%nnv-stat: race_nodes_cancelled=%llu\n", unified_data->race_cancelled);
        printf("%%%%%%nnv-stat: race_subtrees_left_with_unknowns=%llu\n", unified_data->race_recorded);
        printf("%%%%%%nnv-stat: race_records_lost=%llu\n", unified_data->race_lost);
      }
      scip_manager->print_statistics();
    }
#endif
    if(uroot.config.jet_relu_select()) {
      printf("%%%%%%nnv-stat: rsel_choices=%llu\n", unified_data->rsel_choices);
      printf("%%%%%%nnv-stat: rsel_fallbacks=%llu\n", unified_data->rsel_fallbacks);
      printf("%%%%%%nnv-stat: rsel_relus=%llu\n", unified_data->rsel_relus);
      printf("%%%%%%nnv-stat: rsel_inputs=%llu\n", unified_data->rsel_inputs);
      printf("%%%%%%nnv-stat: rsel_relax=%llu\n", unified_data->rsel_relax);
      if(unified_data->rsel_choices > 0) {
        printf("%%%%%%nnv-stat: rsel_mean_candidates=%g\n", static_cast<double>(unified_data->rsel_candidates) / unified_data->rsel_choices);
        printf("%%%%%%nnv-stat: rsel_mean_rcl=%g\n", static_cast<double>(unified_data->rsel_rcl) / unified_data->rsel_choices);
      }
    }
    if(uroot.config.jet_var_select != JET_VAR_DEFAULT) {
      printf("%%%%%%nnv-stat: abs_choices=%llu\n", unified_data->abs_choices);
      printf("%%%%%%nnv-stat: abs_fallbacks=%llu\n", unified_data->abs_fallbacks);
      printf("%%%%%%nnv-stat: abs_relax=%llu\n", unified_data->abs_relax);
      if(unified_data->abs_choices > 0) {
        printf("%%%%%%nnv-stat: abs_mean_candidates=%g\n", static_cast<double>(unified_data->abs_candidates) / unified_data->abs_choices);
        if(uroot.config.jet_var_select == JET_VAR_ABS) {
          printf("%%%%%%nnv-stat: abs_mean_rcl=%g\n", static_cast<double>(unified_data->abs_rcl) / unified_data->abs_choices);
        }
      }
    }
    if(uroot.config.jet_candidates) {
      printf("%%%%%%nnv-stat: jet_cand_nodes=%llu\n", unified_data->cand_nodes);
      printf("%%%%%%nnv-stat: jet_cand_points=%llu\n", unified_data->cand_points);
      printf("%%%%%%nnv-stat: jet_cand_found=%llu\n", unified_data->cand_found);
      printf("%%%%%%nnv-stat: jet_cand_found_middle=%llu\n", unified_data->cand_found_mid);
      printf("%%%%%%nnv-stat: jet_cand_found_forms_corner=%llu\n", unified_data->cand_found_forms);
      printf("%%%%%%nnv-stat: jet_cand_found_grad_corner=%llu\n", unified_data->cand_found_grad);
      printf("%%%%%%nnv-stat: jet_cand_grad_nodes=%llu\n", unified_data->cand_grad_nodes);
    }
    uroot.stats.print_mzn_end_stats();
  }
  if (uroot.stats.solutions > 0) printf("sat\n");
  else if (uroot.stats.unknowns > 0) printf("unknown\n");
  else if (interrupted) printf("timeout\n");
  /** Some subtrees were skipped (e.g. an input without finite bounds cannot be split), so the absence
   * of a counterexample proves nothing. */
  else if (!uroot.stats.exhaustive) printf("unknown\n");
  else printf("unsat\n");
}

/** We configure the GPU according to the user configuration:
 * 1) Guess the "best" number of blocks per SM, if not provided.
 * 2) Update the number of subproblems to at least "30 * B" where B is the number of blocks.
 * 3) Configure the size of the shared memory.
 * 4) Increase the global heap memory.
 * 5) Increase the stack size if requested by the user.
 */
MemoryConfig configure_gpu_fbarebones(Configuration<battery::standard_allocator>& config,
  Statistics<battery::standard_allocator>& stats, const Network& net, int max_comparisons)
{
  /** I. Number of blocks per SM. */
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int max_block_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_block_per_sm, (void*) gpu_fbarebones_solve, CUDA_THREADS_PER_BLOCK, 0);
  if(config.verbose_solving) {
    printf("%% max_blocks_per_sm=%d\n", max_block_per_sm);
  }
  if(config.or_nodes != 0) {
    stats.num_blocks = std::min(max_block_per_sm * deviceProp.multiProcessorCount, (int)config.or_nodes);
    if(config.verbose_solving >= 1 && stats.num_blocks < config.or_nodes) {
      printf("%% WARNING: -or %d is too high on your GPU architecture, it has been reduced to %d.\n", (int)config.or_nodes, stats.num_blocks);
    }
  }
  else {
    stats.num_blocks = max_block_per_sm * deviceProp.multiProcessorCount;
  }

  /** II. Number of subproblems. */
  stats.print_stat("subproblems_power", config.subproblems_power);
  if(config.subproblems_power == -1) {
    config.subproblems_power = 0;
    while((1 << config.subproblems_power) < config.subproblems_factor * stats.num_blocks) {
      config.subproblems_power++;
    }
  }

  /** III. Size of the heap global memory.
   * The estimation is very conservative, normally we should not run out of memory.
   * */
  size_t store_bytes = gpu_sizeof<FStore>() + gpu_sizeof<abstract_ptr<FStore>>() + net.num_neurons * gpu_sizeof<FItv>();
  /** The propagator has no bytecode: it is only a pointer to the network (which stays in managed
   * memory and is shared by all the blocks) and a pointer to the store of the block. */
  size_t iprop_bytes = gpu_sizeof<FProp>() + gpu_sizeof<abstract_ptr<FProp>>();
  size_t mem_per_block = gpu_sizeof<BlockData>()
    + store_bytes * size_t{3}  // current, root, inner box.
    + store_bytes * size_t{2}  // search strategies
    + iprop_bytes * size_t{2}
    + (net.num_deductions() + max_comparisons) * size_t{4} * gpu_sizeof<bound_type>()  // fixpoint engine
    + (gpu_sizeof<bound_type>() + gpu_sizeof<LightBranch<FItv>>()) * size_t{MAX_SEARCH_DEPTH};
  /** The symbolic forms of design 2, one copy per block (`-jet_symbolic`). */
  if(config.jet_symbolic) {
    const size_t sym_bytes = symbolic::storage_size_rows(net.num_form_rows, net.num_inputs()) * sizeof(bound_type);
    mem_per_block += sym_bytes;
    stats.print_memory_statistics(config.verbose_solving, "jet_symbolic_bytes_per_block", sym_bytes);
  }
  /** The scratch of the backward contractor (`-jet_backward`), one per block. */
  if(config.jet_backward != JET_BACKWARD_OFF) {
    const int rows = backward::rows_per_group(config.jet_bwd_rows);
    const size_t bwd_bytes = backward::scratch_values(net.num_neurons, rows) * sizeof(bound_type)
      + backward::scratch_ints(net.num_layers(), max_layer_width(net), rows) * sizeof(int);
    mem_per_block += bwd_bytes;
    stats.print_memory_statistics(config.verbose_solving, "jet_backward_bytes_per_block", bwd_bytes);
  }
  /** The scratch of `-jet_var_select` (`jet_abs.hpp`), one per block. */
  if(config.jet_var_select != JET_VAR_DEFAULT) {
    const bool jac = config.jet_var_select == JET_VAR_SMEAR || config.jet_var_select == JET_VAR_ABS;
    mem_per_block += ((jac ? size_t{2} * net.num_neurons : size_t{0}) + net.num_inputs()) * sizeof(bound_type)
      + net.num_inputs() * sizeof(int);
  }
  /** The scratch of `-relu_order smear|abs` and `-split_on mixed` (`relu_select_split`), one per block. */
  if(config.jet_relu_select()) {
    const size_t n0 = static_cast<size_t>(net.num_inputs()), N = static_cast<size_t>(net.num_neurons);
    mem_per_block += (size_t{2} * n0 + size_t{6} * N + (config.jet_symbolic ? size_t{2} * N : size_t{0})) * sizeof(bound_type)
      + (n0 + N) * sizeof(int);
  }
  /** The evaluation of the candidate points (`-jet_candidates`, `-jet_leaf_check strict`), one per block. */
  if(config.jet_candidates || config.jet_leaf_check == JET_LEAF_STRICT) {
    mem_per_block += size_t{2} * static_cast<size_t>(net.num_neurons) * sizeof(double);
  }
  if(config.jet_candidates && config.jet_cand_grad) {
    mem_per_block += size_t{2} * static_cast<size_t>(net.num_neurons) * sizeof(bound_type);
  }
  size_t estimated_global_mem = gpu_sizeof<UnifiedData>() + store_bytes * size_t{5} + iprop_bytes +
    gpu_sizeof<GridData>();

  size_t mem_for_blocks = deviceProp.totalGlobalMem - estimated_global_mem - (deviceProp.totalGlobalMem / 100 * 10);
  stats.num_blocks = std::max(size_t{1}, std::min(mem_for_blocks / mem_per_block, static_cast<size_t>(stats.num_blocks)));
  estimated_global_mem += stats.num_blocks * mem_per_block;
  if(estimated_global_mem > deviceProp.totalGlobalMem / 100 * 90) {
    printf("%% WARNING: The estimated global memory is larger than 90%% of the total global memory.\n\
%% It is possible to run out of memory during solving.\n");
  }
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, deviceProp.totalGlobalMem / 100 * 97));
  stats.print_memory_statistics(config.verbose_solving, "heap_memory", estimated_global_mem);
  stats.print_memory_statistics(config.verbose_solving, "mem_per_block", mem_per_block);
  stats.print_memory_statistics(config.verbose_solving, "total_global_mem_bytes", deviceProp.totalGlobalMem);

  // We still need to improve this, for some large problems, it is required to avoid running out of memory.
  stats.num_blocks = std::min(static_cast<size_t>(stats.num_blocks), size_t{200000000} / static_cast<size_t>(net.num_neurons));
  stats.print_stat("num_blocks", stats.num_blocks);

  /** IV. Increase the stack if requested by the user. */
  if(config.stack_kb != 0) {
    CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, config.stack_kb*1000));
    // The stack allocated depends on the maximum number of threads per SM, not on the actual number of threads per block.
    size_t total_stack_size = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor * config.stack_kb * 1000;
    stats.print_memory_statistics(config.verbose_solving, "stack_memory", total_stack_size);
  }

  /** V. Configure the shared memory size. */
  int blocks_per_sm = std::max(1, (stats.num_blocks + deviceProp.multiProcessorCount - 1) / deviceProp.multiProcessorCount);
  MemoryConfig mem_config;
  if(config.only_global_memory) {
    mem_config = MemoryConfig(store_bytes, iprop_bytes);
  }
  else {
    mem_config = MemoryConfig((void*) gpu_fbarebones_solve, config.verbose_solving, blocks_per_sm, store_bytes, iprop_bytes);
  }
  mem_config.print_mzn_statistics(config, stats);
  return mem_config;
}

__global__ void initialize_global_data(
  UnifiedData* unified_data,
  bt::unique_ptr<GridData, bt::global_allocator>* grid_data_ptr)
{
  *grid_data_ptr = bt::make_unique<GridData, bt::global_allocator>(unified_data->root);
}

#define TIMEPOINT(KIND) \
  if(threadIdx.x == 0) { \
    block_data.timer = block_data.stats.stop_timer(Timer::KIND, block_data.timer); \
  }

__global__ void gpu_fbarebones_solve(UnifiedData* unified_data, GridData* grid_data) {
  extern __shared__ unsigned char shared_mem[];
  auto& config = unified_data->root.config;
  BlockData& block_data = grid_data->blocks[blockIdx.x];
  if(threadIdx.x == 0 && blockIdx.x == 0 && config.verbose_solving) {
    printf("%% GPU kernel started, starting solving...\n");
  }

  /** A. Initialization the block data and the fixpoint engine. */

  block_data.allocate(*unified_data, *grid_data, shared_mem);
  __syncthreads();
  FProp& iprop = *block_data.iprop;
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
  __shared__ BlockAsynchronousFixpointGPU<true> fp_engine;
#else
  __shared__ FixpointSubsetGPU<BlockAsynchronousFixpointGPU<true>, bt::global_allocator, CUDA_THREADS_PER_BLOCK> fp_engine;
  fp_engine.init(unified_data->root.max_num_deductions());
#endif
  /** This shared variable is necessary to avoid multiple threads to read into `unified_data.stop.test()`,
   * potentially reading different values and leading to deadlock. */
  __shared__ bool stop;
  __shared__ bool has_changed;
  __shared__ bool is_leaf_node;
  __shared__ int remaining_depth;
  /** Whether the block took a subproblem (`take_subproblem`). */
  __shared__ bool has_subproblem;
  stop = false;
  auto group = cooperative_groups::this_thread_block();
  const int num_disjuncts = unified_data->root.num_disjuncts();
  const subproblems::counter_t num_paths = subproblems::counter_t{1} << config.subproblems_power;
  /** The next subproblem of the block, over the disjuncts of the launch (`jet_subproblems.hpp`), by the
   * thread 0: its disjunct is selected (`select_disjunct`) and its path is `subproblem_idx`. */
  auto take_subproblem = [&]() {
    int d = 0;
    subproblems::counter_t path = 0;
    has_subproblem = subproblems::take<subproblems::DeviceAtomics>(&grid_data->ticket,
      grid_data->next_path.data(), num_disjuncts, num_paths, d, path);
    if(has_subproblem) {
      block_data.select_disjunct(*unified_data, d);
      block_data.subproblem_idx = static_cast<size_t>(path);
    }
  };
  if(threadIdx.x == 0) {
    block_data.timer = block_data.stats.start_timer_device();
    block_data.start_time = block_data.timer;
    take_subproblem();
  }
  __syncthreads();

  /** B. Start the main dive and solve loop. */
  while(has_subproblem && !stop) {
    if(config.verbose_solving >= 2 && threadIdx.x == 0) {
      grid_data->print_lock.acquire();
      printf("%% Block %d solves subproblem num %" PRIu64 " of disjunct %d\n", blockIdx.x, block_data.subproblem_idx,
        unified_data->root.disjunct_ids[block_data.disjunct] + 1);
      grid_data->print_lock.release();
    }

    // C. Restoring the current state to the root node.

    block_data.current_strategy = 0;
    block_data.next_unassigned_var = 0;
    block_data.depth = 0;
    /** A subproblem starts with no underapproximation (`uass.hpp`). */
    if(threadIdx.x == 0) {
      uass::reset(block_data);
      block_data.scip_depth = -1;
    }
    unified_data->root.stores[block_data.disjunct]->copy_to(group, *block_data.store);
#ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
    /** The comparisons of the disjunct are deductions too (`FastNNRelu::num_deductions`). */
    fp_engine.reset(iprop.num_deductions());
#endif
    __syncthreads();

    // D. Dive into the search tree until we reach the target subproblem.
    remaining_depth = config.subproblems_power;
    if(threadIdx.x == 0) {
      block_data.dive_timer = block_data.stats.start_timer_device();
      is_leaf_node = false;
      block_data.diving = true;
    }
    __syncthreads();
    while(remaining_depth > 0 && !is_leaf_node && !stop) {
      __syncthreads();
      if(threadIdx.x == 0) { block_data.unknowns_node_start = block_data.stats.unknowns; }
      propagate(*unified_data, *grid_data, block_data, fp_engine, stop, has_changed, is_leaf_node);
      __syncthreads();
      /** `-under scip`: in the dive, only a leaf is sent (the nodes above the subproblems are covered by
       * the subproblems, whose roots are sent by the search below). */
      if(config.under == UNDER_SCIP && is_leaf_node && !stop && !iprop.is_bot() && block_data.stats.solutions == 0) {
        scip_offer(*unified_data, block_data);
      }
      if(!is_leaf_node) {
        branch(*unified_data, *grid_data, block_data, has_changed);
        __syncthreads();
        // Split was not able to split a domain. It means that the search strategy is not complete due to unsplittable infinite domains.
        // We skip the subtree, and set exhaustive to `false`.
        if(block_data.decisions[0].var.is_untyped()) {
          is_leaf_node = true;
          block_data.stats.exhaustive = false;
          if(threadIdx.x == 0 && config.verbose_solving >= 1) { printf("%% WARNING: nothing to branch on (a domain without finite bounds, or a box of width at most epsilon with -var_order input_order), search is not exhaustive\n");}
        }
        else if(threadIdx.x == 0) {
          --remaining_depth;
          // We do not record the decisions when diving.
          --block_data.depth;
          /** We commit to one of the branches depending on the current value on the path.
           * Suppose the depth is 3, the path is "010" we are currently at `remaining_depth = 1`.
           * We must extract the bit "1", and we do so by standard bitwise manipulation.
           * Whenever the branch_idx is 0 means to take the left branch, and 1 means to take the right branch.
           */
          size_t branch_idx = (block_data.subproblem_idx & (size_t{1} << remaining_depth)) >> remaining_depth;
          /** We immediately commit to the branch. */
          // printf("split on %d (", block_data.decisions[0].var.vid()); block_data.store->project(block_data.decisions[0].var).print(); printf(")\n");
          /** The right child of a decision with a single child (`push_decision`) does not exist: the
           * subproblems of this subtree are skipped, as those of a leaf. */
          if(branch_idx == 1 && block_data.decisions[0].ropes[0] == block_data.decisions[0].ropes[1]) {
            is_leaf_node = true;
          }
          else {
            block_data.store->embed(block_data.decisions[0].var, block_data.decisions[0].children[branch_idx]);
          }
        }
      }
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      block_data.stats.stop_timer(Timer::DIVE, block_data.dive_timer);
      block_data.diving = false;
    }
    // E. Skip subproblems that are not reachable.

    /** If we reached a leaf node before the subproblem was reached, then it means a whole subtree should be skipped. */
    if(is_leaf_node && !stop) {
       /** To skip all the paths of the subtree obtained, we perform bitwise operations.
       * Suppose the current path is "00" turn left two times, and the following search tree:
       *         *         depth = 0
       *        / \
       *      0    1       depth = 1
       *    / \   / \
       *   00 01 10 11     depth = 2
       *
       * If we detect a leaf node at depth 1, after only one left turn, we must skip the remaining of the subtree, in particular to avoid exploring the path "01".
       * To achieve that, we take the current path "00", shift it to the right by 1 (essentially erasing the path that has not been explored), increment it to go to the next subtree (at the same depth), and shift it back to the left to reach the first subproblem of the subtree.
       * Cool huh?
       */
      if(threadIdx.x == 0) {
        size_t next_subproblem_idx = static_cast<size_t>(subproblems::skip_target(block_data.subproblem_idx, remaining_depth));
        /** Make sure the subtree is skipped, in the disjunct of the subproblem only. */
        subproblems::skip<subproblems::DeviceAtomics>(grid_data->next_path.data(), block_data.disjunct,
          block_data.subproblem_idx, remaining_depth);
        /** It is possible that other blocks skip similar subtrees.
          * Hence, we only count the subproblems skipped by the block solving the left most subproblem. */
        if((block_data.subproblem_idx & ((size_t{1} << remaining_depth) - size_t{1})) == size_t{0}) {
          block_data.stats.eps_skipped_subproblems += next_subproblem_idx - block_data.subproblem_idx;
        }
      }
    }
    else if(!stop) {

      // F. Solve the current subproblem.

      // We skip the remaining of the EPS strategy if there is any.
      if(threadIdx.x == 0 && grid_data->has_eps_strategy) {
        block_data.current_strategy = battery::max(1, block_data.current_strategy);
        block_data.next_unassigned_var = 0;
      }

      while(!stop) {

        /** `-scip_mode race`: when SCIP closed the node sent from the current path, the block leaves its
         * subtree: the node is taken as a leaf (`depth = scip_depth`), and the unknown leaves found under it
         * are not counted. */
        if(threadIdx.x == 0) {
          has_changed = race_closed(*unified_data, block_data);
          if(has_changed) {
            block_data.depth = block_data.scip_depth;
            block_data.stats.unknowns = block_data.scip_unknowns_at_send;
            block_data.stats.exhaustive = block_data.scip_exhaustive_at_send;
            block_data.scip_depth = -1;
            atomicAdd(&unified_data->race_pruned, 1ull);
          }
          block_data.unknowns_node_start = block_data.stats.unknowns;
        }
        __syncthreads();
        if(has_changed) {
          if(threadIdx.x == 0) { is_leaf_node = true; }
          __syncthreads();
        }
        else {
        // I. Propagate the current node.
        propagate(*unified_data, *grid_data, block_data, fp_engine, stop, has_changed, is_leaf_node);
        __syncthreads();
        /** `-under scip`: the candidate points of the node are tested if they were not (`-jet_candidates`),
         * then the node is sent to SCIP if it is the first of its path with few unstable ReLUs. */
        if(config.under == UNDER_SCIP && !stop && !iprop.is_bot() && block_data.stats.solutions == 0) {
          if(scip_offer(*unified_data, block_data)) {
            if(threadIdx.x == 0) { is_leaf_node = true; }
            __syncthreads();
          }
        }
        }

        // II. Branching

        if(!is_leaf_node) {
          // If we are at the root of the current subproblem, we create a snapshot for future backtracking.
          if(block_data.depth == 0) {
            block_data.store->copy_to(group, *block_data.root_store);
            if(threadIdx.x == 0) {
              block_data.snapshot_root_strategy = block_data.current_strategy;
              block_data.snapshot_next_unassigned_var = block_data.next_unassigned_var;
            }
          }
          __syncthreads();
          /** `split` writes an untyped variable in `decisions[depth]` and pushes nothing when it finds no
           * variable to branch on: the test must be on that slot, not on `decisions[depth - 1]` (the
           * decision of the parent, already applied: `next()` would move it to its right child, and
           * `decisions[-1]` at the root). Thread 0 increments `depth` only after a `__syncthreads()`. */
          const int depth_before_branch = block_data.depth;
          branch(*unified_data, *grid_data, block_data, has_changed);
          __syncthreads();
          // Split was not able to split a domain. It means that the search strategy is not complete due to unsplittable infinite domains.
          // We trigger backtracking, and set exhaustive to `false`.
          if(block_data.decisions[depth_before_branch].var.is_untyped()) {
            is_leaf_node = true;
            block_data.stats.exhaustive = false;
            if(threadIdx.x == 0 && config.verbose_solving >= 1) { printf("%% WARNING: nothing to branch on (a domain without finite bounds, or a box of width at most epsilon with -var_order input_order), search is not exhaustive\n");}
          }
          else if(threadIdx.x == 0) {
            // Apply the decision.
            // printf("split on %d (", block_data.decisions[block_data.depth-1].var.vid()); block_data.store->project(block_data.decisions[block_data.depth-1].var).print(); printf(")\n");
            block_data.store->embed(block_data.decisions[block_data.depth-1].var, block_data.decisions[block_data.depth-1].next());
            // printf("left decision: %d [", block_data.decisions[block_data.depth - 1].var.vid()); block_data.decisions[block_data.depth - 1].current().print(); printf("]\n");
          }
        }

        // III. Backtracking

        if(is_leaf_node) {
          // Leaf node at root.
          if(block_data.depth == 0) {
            break;
          }
          if(threadIdx.x == 0) {
            block_data.depth = block_data.decisions[block_data.depth-1].ropes[block_data.decisions[block_data.depth-1].current_idx];
          }
          __syncthreads();
          // Check if there is no more node to visit.
          if(block_data.depth == -1) {
            break;
          }
          // Restore from root by copying the store and re-applying all decisions from root to block_data.depth-1.
#ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
          fp_engine.reset(iprop.num_deductions());
#endif
          block_data.root_store->copy_to(group, *block_data.store);
          // __syncthreads();
          // if(threadIdx.x == 0) {
          //   printf("%d: restoring store: ", block_data.depth); block_data.store->print(); printf("\n");
          // }
          // __syncthreads();
          if(threadIdx.x == 0) {
            has_changed = true;
          }
          __syncthreads();
          while(has_changed) {
            __syncthreads();
            if(threadIdx.x == 0) {
              has_changed = false;
            }
            __syncthreads();
            for(int i = threadIdx.x; i < block_data.depth - 1; i += blockDim.x) {
              if(block_data.store->embed(block_data.decisions[i].var, block_data.decisions[i].current())) {
                has_changed = true;
              }
            }
            __syncthreads();
          }
          if(threadIdx.x == 0) {
            block_data.store->embed(block_data.decisions[block_data.depth - 1].var, block_data.decisions[block_data.depth - 1].next());
            // printf("right decision: %d [", block_data.decisions[block_data.depth - 1].var.vid()); block_data.decisions[block_data.depth - 1].current().print(); printf("]\n");
            /** The path of the right child has the decisions `0..depth-1` (`uass.hpp`). */
            uass::backtrack(block_data);
            /** `-under scip`: the node sent to SCIP is not an ancestor of the right child any more. */
            if(block_data.scip_depth >= block_data.depth) {
              race_leave(*unified_data, block_data);
              block_data.scip_depth = -1;
            }
            block_data.current_strategy = block_data.snapshot_root_strategy;
            block_data.next_unassigned_var = block_data.snapshot_next_unassigned_var;
          }
          // __syncthreads();
          // if(threadIdx.x == 0) {
          //   printf("%d: reapplied decisions: ", block_data.depth); block_data.store->print(); printf("\n");
          // }
          // __syncthreads();
        }
      }
      /** If we didn't stop solving because of an external interruption, we increase the number of subproblems solved. */
      if(threadIdx.x == 0 && block_data.stats.nodes < config.stop_after_n_nodes
        && !unified_data->stop.test())
      {
        block_data.stats.eps_solved_subproblems += 1;
      }
    }

    // G. Move to the next subproblem.

    /** We prepare the block to solve the next problem.
     * We update the subproblem index to the next subproblem to solve. */
    if(threadIdx.x == 0 && !stop) {
      /** `-scip_mode race`: the subtree of the node sent from this subproblem, if any, was searched. */
      race_leave(*unified_data, block_data);
      block_data.scip_depth = -1;
      /** To avoid that several blocks solve the same subproblem, the paths are taken with atomic
       * post-increments (`jet_subproblems.hpp`). */
      take_subproblem();
    }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(block_data.stats.nodes < config.stop_after_n_nodes && !unified_data->stop.test()) {
      block_data.stats.num_blocks_done = 1;
    }
    block_data.stats.timers.update_timer(Timer::FIRST_BLOCK_IDLE, block_data.start_time);
    block_data.stats.cumulative_time_block = block_data.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE);
  }
  __syncthreads();
#ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
  fp_engine.destroy();
#endif
  block_data.deallocate_shared_data();
  __syncthreads();
}


/** The threads of a block, as expected by `symbolic::pass`. */
struct BlockGroup {
  __device__ INLINE int rank() const { return threadIdx.x; }
  __device__ INLINE int size() const { return blockDim.x; }
  __device__ INLINE void sync() const { __syncthreads(); }
};

/** One symbolic round of design 2 (`-jet_symbolic`) on the store of the block, by all its threads,
 * with the forms stored in the layout `layout` (`-jet_sym_layout`), and the hidden neurons clipping the
 * inputs if `hidden_clipping` (`-jet_sym_clip_hidden`). With `forward_only` (`-forward_only`), nothing
 * is clipped and the comparisons of the property are only checked (see `symbolic::pass`).
 * \return `true` in every thread if the store changed. */
__device__ bool symbolic_round(BlockData& block_data, int layout, bool hidden_clipping, bool forward_only) {
  FProp& iprop = *block_data.iprop;
  BlockGroup group;
  FProp::NeuronsAccess access{*block_data.store};
  const auto& net = *iprop.net;
  const bool changed = symbolic::pass<bound_type>(group, net, *iprop.prop, access,
    block_data.sym_forms.data(), layout, hidden_clipping, forward_only, net.form_row.data(), net.num_form_rows,
    block_data.sym_pre.size() > 0 ? block_data.sym_pre.data() : nullptr);
  return __syncthreads_or(changed) != 0;
}

/** One pass of the backward contractor (`-jet_backward`, `jet_backward.hpp`) on the store of the block, by
 * all its threads: with `full`, the rows of the unstable neurons, of the outputs and of the comparisons;
 * otherwise the rows of the outputs and of the comparisons only. The bounds of the pre-activations start
 * from those of the root of the disjunct (`NNRoot::bwd_root_p`) and, with `reuse`, from those of the
 * previous pass on the same node. With `-forward_only`, nothing is clipped. Not inlined, so that its
 * locals do not grow the stack frame of the kernels that never call it (`-jet_backward off`).
 * \return `true` in every thread if the store changed. */
__device__ __noinline__ bool backward_round(UnifiedData& unified_data, BlockData& block_data, bool full, bool reuse) {
  FProp& iprop = *block_data.iprop;
  BlockGroup group;
  FProp::NeuronsAccess access{*block_data.store};
  const auto& root = unified_data.root;
  const auto& config = root.config;
  const auto& net = *iprop.net;
  const int rows = backward::rows_per_group(config.jet_bwd_rows);
  const auto scratch = backward::make_scratch<bound_type>(block_data.bwd_values.data(), block_data.bwd_ints.data(),
    net.num_neurons, net.num_layers(), max_layer_width(net), rows);
  backward::Options opts;
  opts.neurons = full;
  opts.spec = true;
  opts.clip = config.jet_bwd_clip && !config.forward_only;
  opts.et = config.jet_bwd_et;
  opts.reuse = reuse;
  opts.rows = rows;
  const size_t per_disjunct = size_t{2} * static_cast<size_t>(net.num_neurons);
  const bound_type* root_p = (root.bwd_root_p.size() >= per_disjunct * static_cast<size_t>(block_data.disjunct + 1))
    ? root.bwd_root_p.data() + per_disjunct * static_cast<size_t>(block_data.disjunct) : nullptr;
  backward::Counters counters;
  const bool changed = backward::pass<bound_type>(group, net, *iprop.prop, access, scratch, root_p, opts, &counters);
  if(threadIdx.x == 0) {
    atomicAdd(&unified_data.bwd_passes, 1ull);
    atomicAdd(&unified_data.bwd_rows, counters.rows);
    atomicAdd(&unified_data.bwd_macs, counters.macs);
  }
  return __syncthreads_or(changed) != 0;
}

/** The fixpoint of a node, by all the threads of the block: the interval fixpoint (or, with
 * `-jet_interval_prop forward_backward`, one forward and one backward sweep), alternated, when
 * `symbolic` holds, with the symbolic rounds of design 2 until a round changes nothing (or
 * `-jet_sym_rounds` rounds when it is not 0). `rounds` receives the number of symbolic rounds.
 * `backward` (`-jet_backward`): 0, no backward pass; 1, the passes of the rows of the property
 * (`backward_round` without `full`); 2, the passes of all the rows. With `backward`, each round runs the
 * symbolic pass (if `symbolic`, within `-jet_sym_rounds`) and then the backward pass (within
 * `-jet_bwd_rounds`), and the interval fixpoint after a round that changed the store; the rounds stop
 * when a round changes nothing (`-jet_sym_skip_final_interval` is not used then).
 * \return the number of iterations of the fixpoint engine. */
template <class FPEngine>
__device__ INLINE int node_fixpoint(UnifiedData& unified_data, BlockData& block_data, FPEngine& fp_engine,
  bool symbolic, int& rounds, int backward = 0)
{
  __shared__ int warp_iterations[CUDA_THREADS_PER_BLOCK/32];
  warp_iterations[threadIdx.x / 32] = 0;
  auto& config = unified_data.root.config;
  FProp& iprop = *block_data.iprop;
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
  int num_active = iprop.num_deductions();
#else
  int num_active = fp_engine.num_active();
#endif
  /** `-jet_interval_prop forward_backward`: no fixpoint, one forward sweep (the layers in order, the
   * forward projection of each neuron only), the comparisons of the property once, then one backward
   * sweep (the layers in reverse order, the forward and backward projections of each neuron, so that a
   * layer narrowed by the next one narrows the one it reads). The neurons of a layer are deduced in
   * parallel, and a barrier separates two layers. Counted as one iteration. */
  /** Every thread reads `is_bot` before the barrier and nobody writes in between, so all the
   * threads take the same branch and reach the same barriers. */
  auto stop_sweep = [&]() {
    const bool bot = iprop.is_bot();
    __syncthreads();
    return bot;
  };
  /** The forward sweep (the layers in order, the forward projection of each neuron), then the
   * comparisons of the property once. \return `false` if the store is empty before the property. */
  auto run_forward_sweep = [&]() {
    const auto& net = *iprop.net;
    for(int l = 1; l < net.num_layers(); ++l) {
      if(stop_sweep()) { return false; }
      for(int j = threadIdx.x; j < net.layers[l].size; j += blockDim.x) {
        iprop.deduce_layer_neuron(l, j, true);
      }
      __syncthreads();
    }
    if(stop_sweep()) { return false; }
    for(int k = threadIdx.x; k < iprop.prop->size(); k += blockDim.x) {
      iprop.deduce_property(k);
    }
    __syncthreads();
    return true;
  };
  auto run_forward_backward = [&]() {
    const auto& net = *iprop.net;
    const int num_layers = net.num_layers();
    if(threadIdx.x == 0) {
      block_data.stats.num_deductions += 2 * net.num_deductions() + iprop.prop->size();
    }
    if(!run_forward_sweep()) {
      return 1;
    }
    for(int l = num_layers - 1; l >= 1; --l) {
      if(stop_sweep()) { break; }
      for(int j = threadIdx.x; j < net.layers[l].size; j += blockDim.x) {
        iprop.deduce_layer_neuron(l, j, false);
      }
      __syncthreads();
    }
    return 1;
  };
  /** The interval fixpoint, as a function so that it can be resumed after a symbolic round. */
  auto run_fixpoint = [&]() {
    if(config.jet_interval_prop == JET_INTERVAL_FORWARD_BACKWARD) {
      return run_forward_backward();
    }
    int its = 0;
    switch(config.fixpoint) {
      case FixpointKind::AC1: {
        its = fp_engine.fixpoint(
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
          iprop.num_deductions(),
#endif
          [&](int i){ return iprop.fdeduce(i, config.epsilon); },
          [&](){ return iprop.is_bot(); });
        if(threadIdx.x == 0) {
          block_data.stats.num_deductions += its * num_active;
        }
        break;
      }
      case FixpointKind::WAC1: {
        if(num_active <= config.wac1_threshold) {
          its = fp_engine.fixpoint(
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
          iprop.num_deductions(),
#endif
            [&](int i){ return iprop.fdeduce(i, config.epsilon); },
            [&](){ return iprop.is_bot(); });
          if(threadIdx.x == 0) {
            block_data.stats.num_deductions += its * num_active;
          }
        }
        else {
          its = fp_engine.fixpoint(
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
            iprop.num_deductions(),
#endif
            [&](int i){ return fwarp_fixpoint<CUDA_THREADS_PER_BLOCK>(iprop, i, warp_iterations, config.epsilon); },
            [&](){ return iprop.is_bot(); });
          if(threadIdx.x == 0) {
            for(int i = 0; i < CUDA_THREADS_PER_BLOCK/32; ++i) {
              block_data.stats.num_deductions += warp_iterations[i] * 32;
            }
          }
        }
        break;
      }
    }
    return its;
  };
  int fp_iterations = run_fixpoint();
  rounds = 0;
  /** Whether the inputs may have been narrowed after the last forward propagation of the network:
   * the backward sweep of `-jet_interval_prop forward_backward`, or the clipping of a symbolic round
   * not followed by the interval propagation (`-jet_sym_skip_final_interval`). */
  bool unchecked_inputs = config.jet_interval_prop == JET_INTERVAL_FORWARD_BACKWARD;
  if(backward != 0) {
    /** The symbolic pass and the backward contractor, then the interval fixpoint, until nothing changes.
     * A round that changed the store (the clipping of either one included) is followed by the interval
     * fixpoint, so the inputs are never left unchecked by them. */
    int bwd_rounds = 0;
    while(!iprop.is_bot()) {
      const bool sym_now = symbolic && (config.jet_sym_rounds <= 0 || rounds < config.jet_sym_rounds);
      const bool bwd_now = bwd_rounds < config.jet_bwd_rounds;
      if(!sym_now && !bwd_now) { break; }
      bool changed = false;
      if(sym_now) {
        ++rounds;
        changed = symbolic_round(block_data, config.jet_sym_layout, config.jet_sym_clip_hidden, config.forward_only) || changed;
      }
      if(bwd_now && !iprop.is_bot()) {
        ++bwd_rounds;
        changed = backward_round(unified_data, block_data, backward == 2, bwd_rounds > 1) || changed;
      }
      if(!changed) { break; }
      warp_iterations[threadIdx.x / 32] = 0;
      fp_iterations += run_fixpoint();
    }
  }
  else if(symbolic) {
    for(int round = 0; !iprop.is_bot() && (config.jet_sym_rounds <= 0 || round < config.jet_sym_rounds); ++round) {
      ++rounds;
      if(!symbolic_round(block_data, config.jet_sym_layout, config.jet_sym_clip_hidden, config.forward_only)) { break; }
      /** `-jet_sym_skip_final_interval`: no interval propagation after the last symbolic round. */
      if(config.jet_sym_skip_final_interval && config.jet_sym_rounds > 0 && round == config.jet_sym_rounds - 1) {
        unchecked_inputs = unchecked_inputs || !config.forward_only;
        break;
      }
      warp_iterations[threadIdx.x / 32] = 0;
      fp_iterations += run_fixpoint();
    }
  }
  /** The leaf check. `propagate` takes a node that is not empty and whose inputs are all points for a
   * solution. When the inputs may have been narrowed to points after the last forward propagation
   * (above), the network was never evaluated on those points: we propagate them forward once (a
   * forward sweep and the comparisons of the property), which fails the node if they are not a
   * solution. Without it, `-jet_interval_prop forward_backward` returns false `sat` (e.g. on
   * sat_relu: the backward sweep fixes the inputs to 0 or 1). */
  if(unchecked_inputs) {
    const auto& net = *iprop.net;
    const auto& store = *block_data.store;
    int points = iprop.is_bot() ? 0 : 1;
    for(int k = threadIdx.x; k < net.num_inputs(); k += blockDim.x) {
      if(store[k].lb().value() != store[k].ub().value()) { points = 0; }
    }
    if(__syncthreads_and(points) != 0) {
      run_forward_sweep();
      if(threadIdx.x == 0) {
        block_data.stats.num_deductions += net.num_deductions() + iprop.prop->size();
      }
      fp_iterations += 1;
    }
  }
  return fp_iterations;
}

/** `-jet_sym_root_only` or `-jet_backward`: the fixpoint of the root node of the disjunct `d` of the launch
 * with the symbolic rounds of design 2 (`-jet_symbolic`) and the backward contractor with all its rows
 * (`-jet_backward`), computed once, by one block, before the search (`gpu_fbarebones_solve`), and written
 * back to its root store `unified_data->root.stores[d]`, from which its subproblems start; with
 * `-jet_backward`, the bounds of the pre-activations go to `unified_data->root.bwd_root_p`, for the
 * backward passes of the nodes of the disjunct. With `-jet_sym_root_only`, the search then runs no
 * symbolic round. The root node itself (a failure, a solution) is analyzed by the search, which
 * propagates it again. Launched with one block of `CUDA_THREADS_PER_BLOCK` threads and the dynamic
 * shared memory of `gpu_fbarebones_solve`. */
__global__ void gpu_root_symbolic(UnifiedData* unified_data, GridData* grid_data, int d) {
  extern __shared__ unsigned char shared_mem[];
  BlockData& block_data = grid_data->blocks[0];
  block_data.allocate(*unified_data, *grid_data, shared_mem);
  if(threadIdx.x == 0) {
    block_data.select_disjunct(*unified_data, d);
  }
  __syncthreads();
  FProp& iprop = *block_data.iprop;
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
  __shared__ BlockAsynchronousFixpointGPU<true> fp_engine;
#else
  __shared__ FixpointSubsetGPU<BlockAsynchronousFixpointGPU<true>, bt::global_allocator, CUDA_THREADS_PER_BLOCK> fp_engine;
  fp_engine.init(unified_data->root.max_num_deductions());
#endif
  auto group = cooperative_groups::this_thread_block();
  unified_data->root.stores[d]->copy_to(group, *block_data.store);
  __syncthreads();
  int rounds = 0;
  const auto& config = unified_data->root.config;
  const bool backward = config.jet_backward != JET_BACKWARD_OFF;
  const int fp_iterations = node_fixpoint(*unified_data, block_data, fp_engine, config.jet_symbolic, rounds,
    backward ? 2 : 0);
  __syncthreads();
  block_data.store->copy_to(group, *(unified_data->root.stores[d]));
  /** The bounds of the pre-activations of the last backward pass (none if the root failed before). */
  if(backward && !iprop.is_bot()) {
    const auto& net = unified_data->root.net;
    const int rows = backward::rows_per_group(config.jet_bwd_rows);
    const auto scratch = backward::make_scratch<bound_type>(block_data.bwd_values.data(), block_data.bwd_ints.data(),
      net.num_neurons, net.num_layers(), max_layer_width(net), rows);
    const size_t N = static_cast<size_t>(net.num_neurons);
    bound_type* out = unified_data->root.bwd_root_p.data() + size_t{2} * N * static_cast<size_t>(d);
    for(int l = 1; l < net.num_layers(); ++l) {
      if(!backward::has_activation(net.layers[l])) { continue; }
      for(int j = threadIdx.x; j < net.layers[l].size; j += blockDim.x) {
        const int i = net.layers[l].base + j;
        out[i] = scratch.p_lb[i];
        out[N + i] = scratch.p_ub[i];
      }
    }
  }
  if(threadIdx.x == 0) {
    block_data.stats.fixpoint_iterations += fp_iterations;
    unified_data->root_sym_rounds += rounds;
  }
  __syncthreads();
#ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
  fp_engine.destroy();
#endif
  block_data.deallocate_shared_data();
  __syncthreads();
}

/** `-jet_candidates`: test the candidate points of the box of the current node (`jet_candidates.hpp`),
 * by all the threads of the block. Each candidate is evaluated in `double`, rounded outward
 * (`candidates::evaluate`, in `cand_lo` and `cand_hi`), and the strict test decides. The store of the
 * node is only read. If a candidate passes, `inner_box` receives the evaluated point (the
 * counterexample, rounded outward to the intervals of the store) and the function returns `true`;
 * otherwise nothing changed and the search goes on. `use_forms`: the forms of the block are those of
 * this node (it ran a symbolic round), so the corners can be tested. `count`: the test is counted in
 * the statistics of `-jet_candidates` (not for the strict test of a leaf).
 * \return the same value in every thread. */
__device__ bool check_candidates(UnifiedData& unified_data, BlockData& block_data, bool use_forms, bool count = true) {
  using local_itv = typename FItv::local_type;
  const auto& root = unified_data.root;
  const auto& net = root.net;
  const auto& inner = *root.cand_stores[block_data.disjunct];
  const auto& cand_prop = root.cand_props[block_data.disjunct];
  const auto& store = *block_data.store;
  const int n0 = net.num_inputs();
  struct Bounds {
    const NNStore& s;
    __device__ INLINE bound_type lb(int i) const { return s[i].lb().value(); }
    __device__ INLINE bound_type ub(int i) const { return s[i].ub().value(); }
  };
  const Bounds inner_bounds{inner};
  /** The range of each input: the node meets the inner box; it must be finite and not empty. */
  int bad = (inner.is_bot() || store.is_bot()) ? 1 : 0;
  for(int k = threadIdx.x; k < n0 && !bad; k += blockDim.x) {
    const bound_type lo = battery::max(store[k].lb().value(), inner[k].lb().value());
    const bound_type hi = battery::min(store[k].ub().value(), inner[k].ub().value());
    if(!(lo <= hi) || !symbolic::is_finite(lo) || !symbolic::is_finite(hi)) { bad = 1; }
  }
  if(__syncthreads_or(bad) != 0) { return false; }
  const int layout = root.config.jet_sym_layout;
  const bound_type* forms = block_data.sym_forms.data();
  const int rows = net.num_form_rows;
  auto coef = [&](int r, bool upper, int k) {
    return candidates::coef_mid<bound_type>(forms, layout, n0, rows, r, upper, k);
  };
  const bool forms_ok = use_forms && root.config.jet_symbolic && block_data.sym_forms.size() > 0;
  /** `-jet_cand_grad`: without the forms of the node, the corners follow the signs of the middle of the
   * interval Jacobian of their rows (`jet_jacobian.hpp`): a comparison `x - y <= k` minimizes `x - y`, a
   * bound `o <= c` minimizes `o`, a bound `o >= c` maximizes it. */
  const bool grad_ok = !forms_ok && root.config.jet_cand_grad && block_data.cand_grad.size() > 0;
  struct StoreBounds {
    const FStore& s;
    __device__ INLINE bound_type lb(int i) const { return s[i].lb().value(); }
    __device__ INLINE bound_type ub(int i) const { return s[i].ub().value(); }
  };
  const StoreBounds store_bounds{store};
  bound_type* glo = grad_ok ? block_data.cand_grad.data() : nullptr;
  bound_type* ghi = grad_ok ? block_data.cand_grad.data() + net.num_neurons : nullptr;
  const auto& O = net.layers[net.output_layer];
  auto grad_applicable = [&](int id) {
    if(id == 0) { return true; }
    if(id <= cand_prop.size()) { return true; }
    const int r = id - 1 - cand_prop.size();
    if(r / 2 >= O.size) { return false; }
    const int o = O.base + r / 2;
    return (r % 2 == 0) ? inner[o].lb().value() > battery::limits<bound_type>::neg_inf()
                        : inner[o].ub().value() < battery::limits<bound_type>::inf();
  };
  int found_id = -1;
  double* lo = block_data.cand_lo.data();
  double* hi = block_data.cand_hi.data();
  BlockGroup group;
  const int num_ids = candidates::num_candidates(cand_prop.size(), net.layers[net.output_layer].size);
  int tested = 0;
  bool found = false;
  for(int id = 0; id < num_ids && !found; ++id) {
    if(grad_ok ? !grad_applicable(id)
      : !candidates::applicable<bound_type>(id, net, cand_prop, inner_bounds, net.form_row.data(), forms_ok)) { continue; }
    ++tested;
    /** The sign of the coefficient of each input: minimized `sign * row`. */
    bound_type sign = bound_type{1};
    if(grad_ok && id > 0) {
      jacobian::Row row{0, -1, 1.0f, 0.0f};
      if(id <= cand_prop.size()) { row = jacobian::Row{cand_prop.leqs[id - 1].x, cand_prop.leqs[id - 1].y, 1.0f, -1.0f}; }
      else {
        const int r = id - 1 - cand_prop.size();
        row = jacobian::Row{O.base + r / 2, -1, 1.0f, 0.0f};
        if(r % 2 == 0) { sign = bound_type{-1}; }
      }
      jacobian::reverse_row<bound_type>(BlockGroup{}, net, store_bounds, row, glo, ghi);
    }
    for(int k = threadIdx.x; k < n0; k += blockDim.x) {
      const bound_type range_lo = battery::max(store[k].lb().value(), inner[k].lb().value());
      const bound_type range_hi = battery::min(store[k].ub().value(), inner[k].ub().value());
      const bound_type v = (grad_ok && id > 0)
        ? candidates::minimizer<bound_type>(sign * (glo[k] / bound_type{2} + ghi[k] / bound_type{2}), range_lo, range_hi)
        : candidates::value<bound_type>(id, k, range_lo, range_hi, net, cand_prop, net.form_row.data(), coef);
      lo[k] = static_cast<double>(v);
      hi[k] = static_cast<double>(v);
    }
    __syncthreads();
    candidates::evaluate(group, net, lo, hi);
    /** The strict test. */
    int ok = 1;
    for(int i = threadIdx.x; i < net.num_neurons && ok; i += blockDim.x) {
      if(!candidates::neuron_ok(i, lo, hi, inner_bounds)) { ok = 0; }
    }
    for(int q = threadIdx.x; q < cand_prop.size() && ok; q += blockDim.x) {
      if(!candidates::leq_ok(cand_prop.leqs[q], lo, hi)) { ok = 0; }
    }
    found = __syncthreads_and(ok) != 0;
    if(found) { found_id = id; }
  }
  if(found) {
    /** The counterexample in `inner_box`, where a leaf solution is saved. */
    auto& solution = *block_data.inner_box;
    if(threadIdx.x == 0) { solution.join_top(); }
    __syncthreads();
    for(int i = threadIdx.x; i < net.num_neurons; i += blockDim.x) {
      solution.embed(i, local_itv(typename local_itv::LB(battery::rd_cast<bound_type>(lo[i])),
        typename local_itv::UB(battery::ru_cast<bound_type>(hi[i]))));
    }
    __syncthreads();
  }
  if(threadIdx.x == 0 && count) {
    atomicAdd(&unified_data.cand_nodes, 1ull);
    atomicAdd(&unified_data.cand_points, static_cast<unsigned long long>(tested));
    if(found) {
      atomicAdd(&unified_data.cand_found, 1ull);
      atomicAdd(found_id == 0 ? &unified_data.cand_found_mid : (grad_ok ? &unified_data.cand_found_grad : &unified_data.cand_found_forms), 1ull);
    }
    if(grad_ok) { atomicAdd(&unified_data.cand_grad_nodes, 1ull); }
  }
  return found;
}

/** `-jet_var_select` (`jet_abs.hpp`): chooses the input to bisect among those that can be bisected
 * (`uass::splittable`) and pushes the decision (the value order of the strategy), by all the threads of
 * the block. The smear (`smear`, `abs`) is computed by the whole block: from the symbolic forms of the node
 * with `-abs_grad relax` when they are those of the node and every row of the disjunct reads output
 * neurons, from the interval Jacobian (`jet_jacobian.hpp`) otherwise; when it is not finite, the widest
 * input is chosen. The bisections of an input on the branch are those of the decisions of the path, plus,
 * with `-abs_dive_count on`, those of the dive estimated from the widths (`abs::dive_bisections`): from the
 * root of the disjunct to the root of the subproblem, or to the current node during the dive.
 * \return `false` (in every thread) when no input can be bisected: the caller then runs the split of the
 * strategy, which handles the epsilon-box. */
__device__ bool var_select_split(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data) {
  const auto& root = unified_data.root;
  const auto& config = root.config;
  const auto& net = root.net;
  const auto& store = *block_data.store;
  const auto& strat = grid_data.search_strategies[block_data.current_strategy];
  const int n0 = net.num_inputs();
  const double epsilon = config.epsilon;
  const int start = block_data.next_unassigned_var;
  const int mode = config.jet_var_select;
  __shared__ int bad;
  __shared__ int relax_ok;
  int local = 0;
  for(int k = start + threadIdx.x; k < n0; k += blockDim.x) {
    if(uass::splittable(store[k], epsilon)) { local = 1; }
  }
  if(__syncthreads_or(local) == 0) { return false; }
  bound_type* smear = block_data.abs_values.data();
  bool smear_ok = true;
  bool used_relax = false;
  if(mode == JET_VAR_SMEAR || mode == JET_VAR_ABS) {
    const int d = block_data.disjunct;
    const jacobian::Row* rows = root.jac_rows.data() + root.jac_row_start[d];
    const int num_rows = root.jac_row_start[d + 1] - root.jac_row_start[d];
    smear_ok = false;
    /** `-abs_grad relax`: `|c_k| w_k`, `c_k` the coefficient of the input `k` (the middle of its interval) in
     * the lower form of `x` minus the upper form of `y` for a comparison `x - y <= c`, and the larger of
     * the two forms of an output. */
    if(config.abs_grad == ABS_GRAD_RELAX && config.jet_symbolic && block_data.forms_current && block_data.sym_forms.size() > 0) {
      if(threadIdx.x == 0) {
        relax_ok = 1;
        for(int r = 0; r < num_rows; ++r) {
          if(candidates::output_row(net, net.form_row.data(), rows[r].x) < 0
            || (rows[r].y >= 0 && candidates::output_row(net, net.form_row.data(), rows[r].y) < 0)) { relax_ok = 0; }
        }
        bad = 0;
      }
      __syncthreads();
      if(relax_ok) {
        const int layout = config.jet_sym_layout;
        const bound_type* forms = block_data.sym_forms.data();
        const int frows = net.num_form_rows;
        for(int k = threadIdx.x; k < n0; k += blockDim.x) {
          bound_type m = bound_type{0};
          for(int r = 0; r < num_rows; ++r) {
            const int rx = candidates::output_row(net, net.form_row.data(), rows[r].x);
            bound_type c;
            if(rows[r].y >= 0) {
              const int ry = candidates::output_row(net, net.form_row.data(), rows[r].y);
              c = candidates::coef_mid<bound_type>(forms, layout, n0, frows, rx, false, k)
                - candidates::coef_mid<bound_type>(forms, layout, n0, frows, ry, true, k);
              c = c < bound_type{0} ? -c : c;
            }
            else {
              const bound_type a = candidates::coef_mid<bound_type>(forms, layout, n0, frows, rx, false, k);
              const bound_type b = candidates::coef_mid<bound_type>(forms, layout, n0, frows, rx, true, k);
              c = battery::max(a < bound_type{0} ? -a : a, b < bound_type{0} ? -b : b);
            }
            m = battery::max(m, c);
          }
          const bound_type v = m * (store[k].ub().value() - store[k].lb().value());
          if(!jacobian::finite(v)) { bad = 1; }
          smear[k] = v;
        }
        __syncthreads();
        smear_ok = bad == 0;
        used_relax = smear_ok;
      }
      __syncthreads();
    }
    if(!smear_ok) {
      struct Bounds {
        const FStore& s;
        __device__ INLINE bound_type lb(int i) const { return s[i].lb().value(); }
        __device__ INLINE bound_type ub(int i) const { return s[i].ub().value(); }
      };
      const Bounds bounds{store};
      bound_type* glo = smear + n0;
      bound_type* ghi = glo + net.num_neurons;
      smear_ok = jacobian::smear<bound_type>(BlockGroup{}, net, bounds, rows, num_rows, glo, ghi, smear, &bad);
    }
  }
  if(threadIdx.x == 0) {
    int* counts = block_data.abs_counts.data();
    if(mode == JET_VAR_RR || mode == JET_VAR_ABS) {
      for(int k = 0; k < n0; ++k) { counts[k] = 0; }
      for(int i = 0; i < block_data.depth; ++i) {
        const AVar v = block_data.decisions[i].var;
        if(!v.is_untyped() && v.vid() < n0) { ++counts[v.vid()]; }
      }
      if(config.abs_dive_count) {
        const auto& droot = *root.stores[block_data.disjunct];
        const auto& subroot = *block_data.root_store;
        for(int k = 0; k < n0; ++k) {
          const double w_ref = block_data.diving
            ? static_cast<double>(store[k].ub().value()) - static_cast<double>(store[k].lb().value())
            : static_cast<double>(subroot[k].ub().value()) - static_cast<double>(subroot[k].lb().value());
          counts[k] += abs::dive_bisections(
            static_cast<double>(droot[k].ub().value()) - static_cast<double>(droot[k].lb().value()), w_ref);
        }
      }
    }
    const bool fallback = (mode == JET_VAR_SMEAR || mode == JET_VAR_ABS) && !smear_ok;
    const int m = fallback ? JET_VAR_MAXDOM : mode;
    abs::Info info;
    const int k = abs::select(m, config.abs_beta, config.abs_psi, n0, start,
      [&](int i) { return uass::splittable(store[i], epsilon); },
      [&](int i) { return static_cast<double>(store[i].ub().value()) - static_cast<double>(store[i].lb().value()); },
      [&](int i) { return static_cast<double>(smear[i]); },
      [&](int i) { return counts[i]; },
      &info);
    assert(k < n0);
    /** The inputs before the first candidate stay narrower than epsilon on this path (`split`). */
    int first = start;
    while(first < n0 && !uass::splittable(store[first], epsilon)) { ++first; }
    block_data.next_unassigned_var = first;
    block_data.push_decision(strat.val_order, strat.vars[k], static_cast<float>(epsilon));
    atomicAdd(&unified_data.abs_choices, 1ull);
    if(fallback) { atomicAdd(&unified_data.abs_fallbacks, 1ull); }
    if(used_relax) { atomicAdd(&unified_data.abs_relax, 1ull); }
    atomicAdd(&unified_data.abs_candidates, static_cast<unsigned long long>(info.candidates));
    if(m == JET_VAR_ABS) { atomicAdd(&unified_data.abs_rcl, static_cast<unsigned long long>(info.rcl)); }
  }
  __syncthreads();
  return true;
}

/** `-relu_order smear|abs` (`mixed == false`) and `-split_on mixed` (`mixed == true`): chooses the ReLU, or
 * with `mixed` the ReLU or the input, to split, and pushes the decision, by all the threads of the block.
 *
 * The variables. The ReLUs of unknown phase (`relu_split::is_candidate` at `jet_split_point`) and, with
 * `mixed`, the inputs that can be bisected (`uass::splittable`, from `next_unassigned_var`).
 *
 * The scores (`jet_jacobian.hpp`). One reverse pass over the rows of the property gives, for every neuron
 * `y_n`, `g_n = max_r |d row_r / d y_n|`, and for every input its smear. With `-abs_grad interval`, the
 * derivatives are those of the interval Jacobian; with `-abs_grad relax`, those of the linear relaxation
 * back-substituted from the rows (CROWN) on the bounds of the pre-activations of the node. The score of a
 * ReLU is `g_n * w(y_n)` (`-relu_score smear`), or `g_n` times the gap of its relaxation at the node
 * (`-relu_score gap`: `relu_gap(l, u)` when the node ran the symbolic pass or the backward contractor, the
 * width of `y_n` otherwise). The score of an input is half its smear: a bisection halves its width, so it
 * removes about half of its first-order contribution to the rows, where a ReLU split removes its gap.
 * The bounds of the pre-activations are those of the forward deduction from the store, met with those of
 * the last symbolic round and of the backward contractor when they ran on the node.
 *
 * The choice (`abs::select`): `-relu_order smear` (or `-jet_var_select smear` with `mixed`) the maximum
 * score; `abs` the adaptive bisection of Granvilliers over the scores, and in the restricted candidate list:
 *   * ReLUs only: the earliest layer (a ReLU is split at most once on a branch, so the fewest bisections
 *     of the paper would not choose), then the highest score, then the earliest ReLU;
 *   * `mixed`: the fewest bisections on the branch (an input: as `var_select_split`, with
 *     `-abs_dive_count`; a ReLU: 0, it was never split), then the highest score, then the inputs first.
 * When a score is not finite, the widest variable is chosen (half its width for an input).
 * \return `false` (in every thread) when no variable can be split. */
__device__ bool relu_select_split(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data, bool mixed) {
  const auto& root = unified_data.root;
  const auto& config = root.config;
  const auto& net = root.net;
  const auto& store = *block_data.store;
  const auto& strat = grid_data.search_strategies[block_data.current_strategy];
  const int n0 = net.num_inputs();
  const int N = net.num_neurons;
  const int R = static_cast<int>(grid_data.relu_neurons.size());
  const int* relus = grid_data.relu_neurons.data();
  const double epsilon = config.epsilon;
  const int start = block_data.next_unassigned_var;
  const bound_type tau = relu_split::jet_split_point<bound_type>();
  __shared__ int bad;
  /** 1. Is there a variable to split? */
  int local = 0;
  for(int i = threadIdx.x; i < R; i += blockDim.x) {
    if(relu_split::is_candidate(store[relus[i]].lb().value(), store[relus[i]].ub().value(), tau)) { local = 1; }
  }
  if(mixed) {
    for(int k = start + threadIdx.x; k < n0; k += blockDim.x) {
      if(uass::splittable(store[k], epsilon)) { local = 1; }
    }
  }
  if(__syncthreads_or(local) == 0) { return false; }
  /** 2. The scores. */
  bound_type* smear = block_data.rsel_values.data();
  bound_type* glo = smear + n0;
  bound_type* ghi = glo + N;
  bound_type* gy = ghi + N;
  bound_type* plb = gy + N;
  bound_type* pub = plb + N;
  bound_type* score = pub + N;
  struct Bounds {
    const FStore& s;
    __device__ INLINE bound_type lb(int i) const { return s[i].lb().value(); }
    __device__ INLINE bound_type ub(int i) const { return s[i].ub().value(); }
  };
  const Bounds bounds{store};
  const bool relax = config.abs_grad == ABS_GRAD_RELAX;
  const bool sym_now = config.jet_symbolic && block_data.forms_current && block_data.sym_pre.size() > 0;
  const bool bwd_now = block_data.bwd_current && block_data.bwd_values.size() > 0;
  const bool linear_node = sym_now || bwd_now;
  const bool gap = config.relu_score == RELU_SCORE_GAP && linear_node;
  if(relax || gap) {
    const bound_type* s_lb = sym_now ? block_data.sym_pre.data() : nullptr;
    const bound_type* s_ub = sym_now ? block_data.sym_pre.data() + N : nullptr;
    const bound_type* b_lb = nullptr;
    const bound_type* b_ub = nullptr;
    if(bwd_now) {
      const auto scratch = backward::make_scratch<bound_type>(block_data.bwd_values.data(), block_data.bwd_ints.data(),
        net.num_neurons, net.num_layers(), max_layer_width(net), backward::rows_per_group(config.jet_bwd_rows));
      b_lb = scratch.p_lb;
      b_ub = scratch.p_ub;
    }
    jacobian::preactivation<bound_type>(BlockGroup{}, net, bounds, plb, pub, s_lb, s_ub, b_lb, b_ub);
  }
  const int d = block_data.disjunct;
  const jacobian::Row* rows = root.jac_rows.data() + root.jac_row_start[d];
  const int num_rows = root.jac_row_start[d + 1] - root.jac_row_start[d];
  const bool ok = jacobian::smear<bound_type>(BlockGroup{}, net, bounds, rows, num_rows, glo, ghi, smear, &bad,
    gy, relax ? plb : nullptr, relax ? pub : nullptr);
  /** 3. The choice, by the thread 0. */
  if(threadIdx.x == 0) {
    const int off = mixed ? n0 : 0;
    const int n = off + R;
    int* counts = block_data.rsel_counts.data();
    bool finite_scores = ok;
    for(int i = 0; i < R; ++i) {
      const int v = relus[i];
      const bound_type lb = store[v].lb().value(), ub = store[v].ub().value();
      counts[off + i] = mixed ? 0 : grid_data.relu_layer[i];
      if(!relu_split::is_candidate(lb, ub, tau)) { score[off + i] = bound_type{0}; continue; }
      const bound_type m = gap ? jacobian::relu_gap<bound_type>(plb[v], battery::min(pub[v], ub)) : ub - lb;
      score[off + i] = gy[v] * m;
      if(!jacobian::finite(score[off + i])) { finite_scores = false; }
    }
    if(mixed) {
      for(int k = 0; k < n0; ++k) {
        score[k] = smear[k] / bound_type{2};
        counts[k] = 0;
      }
      for(int i = 0; i < block_data.depth; ++i) {
        const AVar v = block_data.decisions[i].var;
        if(!v.is_untyped() && v.vid() < n0) { ++counts[v.vid()]; }
      }
      if(config.abs_dive_count) {
        const auto& droot = *root.stores[block_data.disjunct];
        const auto& subroot = *block_data.root_store;
        for(int k = 0; k < n0; ++k) {
          const double w_ref = block_data.diving
            ? static_cast<double>(store[k].ub().value()) - static_cast<double>(store[k].lb().value())
            : static_cast<double>(subroot[k].ub().value()) - static_cast<double>(subroot[k].lb().value());
          counts[k] += abs::dive_bisections(
            static_cast<double>(droot[k].ub().value()) - static_cast<double>(droot[k].lb().value()), w_ref);
        }
      }
    }
    const int rule = mixed ? (config.jet_var_select == JET_VAR_SMEAR ? JET_VAR_SMEAR : JET_VAR_ABS)
      : (config.relu_order == RELU_ORDER_SMEAR ? JET_VAR_SMEAR : JET_VAR_ABS);
    const int m = finite_scores ? rule : JET_VAR_MAXDOM;
    auto candidate = [&](int i) {
      if(i < off) { return i >= start && uass::splittable(store[i], epsilon); }
      const int v = relus[i - off];
      return relu_split::is_candidate(store[v].lb().value(), store[v].ub().value(), tau);
    };
    auto width = [&](int i) {
      const int v = i < off ? i : relus[i - off];
      const double w = static_cast<double>(store[v].ub().value()) - static_cast<double>(store[v].lb().value());
      return i < off ? w / 2 : w;
    };
    abs::Info info;
    const int k = abs::select(m, config.abs_beta, config.abs_psi, n, 0, candidate, width,
      [&](int i) { return static_cast<double>(score[i]); }, [&](int i) { return counts[i]; }, &info);
    assert(k < n);
    if(k < off) {
      int first = start;
      while(first < n0 && !uass::splittable(store[first], epsilon)) { ++first; }
      block_data.next_unassigned_var = first;
      block_data.push_decision(strat.val_order, strat.vars[k], static_cast<float>(epsilon));
      atomicAdd(&unified_data.rsel_inputs, 1ull);
    }
    else {
      const int v = relus[k - off];
      relu_split::push_relu_decision(block_data, AVar{store.aty(), v},
        bound_type(store[v].lb().value()), bound_type(store[v].ub().value()), tau);
      atomicAdd(&unified_data.rsel_relus, 1ull);
    }
    atomicAdd(&unified_data.rsel_choices, 1ull);
    if(!finite_scores) { atomicAdd(&unified_data.rsel_fallbacks, 1ull); }
    if(relax) { atomicAdd(&unified_data.rsel_relax, 1ull); }
    atomicAdd(&unified_data.rsel_candidates, static_cast<unsigned long long>(info.candidates));
    atomicAdd(&unified_data.rsel_rcl, static_cast<unsigned long long>(m == JET_VAR_ABS ? info.rcl : 1));
  }
  __syncthreads();
  return true;
}

/** `-scip_mode race`: whether SCIP closed the node sent by the block (`scip_depth >= 0`), read in
 * `ScipQueueView::result`. Thread 0 only. */
__device__ INLINE bool race_closed(const UnifiedData& unified_data, const BlockData& block_data) {
  const ScipQueueView& q = unified_data.scip_queue;
  if(q.result == nullptr || block_data.scip_depth < 0) { return false; }
  const unsigned long long t = block_data.scip_ticket;
  const unsigned long long w = *reinterpret_cast<volatile const unsigned long long*>(&q.result[t % static_cast<unsigned long long>(q.rcap)]);
  return (w >> 8) == t + 1 && (w & 0xffull) == SCIP_RESULT_CLOSED;
}

/** `-scip_mode race`: the block leaves the subtree of the node it sent (`scip_depth` is reset). Without
 * unknown leaf in the subtree, the node is cancelled (SCIP need not solve it); otherwise its ticket and
 * the number of unknown leaves are recorded, for the end of the solving (they are not counted when SCIP
 * closed the node). Thread 0 only. */
__device__ INLINE void race_leave(UnifiedData& unified_data, BlockData& block_data) {
  const ScipQueueView& q = unified_data.scip_queue;
  if(q.result == nullptr || block_data.scip_depth < 0) { return; }
  const unsigned long long t = block_data.scip_ticket;
  const size_t delta = block_data.stats.unknowns - block_data.scip_unknowns_at_send;
  if(delta == 0) {
    *reinterpret_cast<volatile unsigned long long*>(&q.cancel[t % static_cast<unsigned long long>(q.rcap)]) = t + 1;
    __threadfence_system();
    atomicAdd(&unified_data.race_cancelled, 1ull);
  }
  else {
    const unsigned long long i = atomicAdd_system(q.rec_count, 1ull);
    if(i < static_cast<unsigned long long>(q.rec_cap)) {
      q.rec[2 * i] = t + 1;
      q.rec[2 * i + 1] = static_cast<unsigned long long>(delta);
      __threadfence_system();
      atomicAdd(&unified_data.race_recorded, 1ull);
    }
    else { atomicAdd(&unified_data.race_lost, 1ull); }
  }
}

/** `-under scip` (`jet_scip.hpp`): by all the threads of the block, after the propagation of a node that
 * did not fail. When no ancestor of the node on the current path was sent (`scip_depth`) and the node has at
 * most `-scip_max_unstable` ReLUs of unknown phase (`relu_split::is_candidate` at `jet_split_point`), the
 * bounds of its neurons are copied to a free slot of the queue (`ScipQueueView`) and the node becomes the
 * sent node of the path. The slot is taken by the thread 0 (an atomic increment of `head`, then the slot
 * must be free); a busy slot drops the node (`scip_dropped`), and a descendant may be sent later. The
 * store is only read: the search of the node goes on. */
__device__ bool scip_offer(UnifiedData& unified_data, BlockData& block_data) {
  __shared__ int unstable;
  __shared__ int slot;
  const auto& config = unified_data.root.config;
  const auto& net = unified_data.root.net;
  const auto& store = *block_data.store;
  __syncthreads();
  if(block_data.scip_depth >= 0) { return false; }
  /** `-scip_mode race`: a node below an underapproximating decision (`uass.hpp`, e.g. an epsilon-box fixed
   * to its middle) is not sent: SCIP would close the point, not the box whose unknown leaf it stands for. */
  if(config.scip_mode == SCIP_MODE_RACE && uass::is_uass(block_data)) { return false; }
  if(threadIdx.x == 0) { unstable = 0; }
  __syncthreads();
  const bound_type tau = relu_split::jet_split_point<bound_type>();
  int local = 0;
  for(int l = 1; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    if(L.has_relu == 0) { continue; }
    for(int j = threadIdx.x; j < L.size; j += blockDim.x) {
      const int v = L.base + j;
      local += relu_split::is_candidate(store[v].lb().value(), store[v].ub().value(), tau) ? 1 : 0;
    }
  }
  if(local > 0) { atomicAdd(&unstable, local); }
  __syncthreads();
  if(unstable > config.scip_max_unstable) { return false; }
  /** The candidate points first (`-jet_candidates`), when the node did not test them (its depth is not a
   * multiple of `-jet_cand_every`): SCIP is only for the nodes where they found nothing. */
  if(config.jet_candidates && !(block_data.depth == 0 || (config.jet_cand_every > 0 && block_data.depth % config.jet_cand_every == 0))) {
    if(check_candidates(unified_data, block_data, block_data.forms_current)) {
      if(threadIdx.x == 0) {
        block_data.stats.timers.update_timer(Timer::LATEST_BEST_OBJ_FOUND, block_data.start_time);
        block_data.solution_disjunct = block_data.disjunct;
        block_data.stats.solutions++;
        unified_data.stop.test_and_set();
      }
      __syncthreads();
      return true;
    }
  }
  ScipQueueView& q = unified_data.scip_queue;
  if(threadIdx.x == 0) {
    const unsigned long long idx = atomicAdd_system(q.head, 1ull);
    const int s = static_cast<int>(idx % static_cast<unsigned long long>(q.capacity));
    if(atomicCAS_system(&q.state[s], SCIP_SLOT_FREE, SCIP_SLOT_WRITING) == SCIP_SLOT_FREE) {
      slot = s;
      q.seq[s] = idx;
      q.dnf[s] = unified_data.root.disjunct_ids[block_data.disjunct];
    }
    else {
      slot = -1;
      atomicAdd(&unified_data.scip_dropped, 1ull);
    }
  }
  __syncthreads();
  if(slot < 0) { return false; }
  const int n = q.n;
  float* dst = q.bounds + static_cast<size_t>(slot) * size_t{2} * static_cast<size_t>(n);
  for(int i = threadIdx.x; i < n; i += blockDim.x) {
    dst[i] = static_cast<float>(store[i].lb().value());
    dst[n + i] = static_cast<float>(store[i].ub().value());
  }
  /** The bounds are visible to the host before the slot is marked ready. */
  __threadfence_system();
  __syncthreads();
  if(threadIdx.x == 0) {
    atomicExch_system(&q.state[slot], SCIP_SLOT_READY);
    block_data.scip_depth = block_data.depth;
    block_data.scip_ticket = q.seq[slot];
    block_data.scip_unknowns_at_send = block_data.unknowns_node_start;
    block_data.scip_exhaustive_at_send = block_data.stats.exhaustive;
    atomicAdd(&unified_data.scip_sent, 1ull);
  }
  __syncthreads();
  return false;
}

/** `-jet_candidates` with `-jet_disjuncts sequential` and a disjunctive property: the candidate points
 * of the root of the disjunct `d` of the launch, tested before any disjunct is searched
 * (`fbarebones_dive_and_solve`), so that a counterexample of a later disjunct is not delayed by the
 * search of an earlier one (with `interleaved`, the first subproblems of all the disjuncts test them).
 * One block computes the fixpoint of the root (with the symbolic rounds when `-jet_symbolic`, also with
 * `-jet_sym_root_only`, so that the corners can be tested) on a copy of `unified_data->root.stores[d]`,
 * which is not changed, and tests the candidates if the root is not empty. A counterexample is saved in
 * `unified_data->root.best` and counted in `unified_data->root.stats.solutions`. Launched with one block
 * of `CUDA_THREADS_PER_BLOCK` threads and the dynamic shared memory of `gpu_fbarebones_solve`. */
__global__ void gpu_root_candidates(UnifiedData* unified_data, GridData* grid_data, int d) {
  extern __shared__ unsigned char shared_mem[];
  auto& root = unified_data->root;
  BlockData& block_data = grid_data->blocks[0];
  block_data.allocate(*unified_data, *grid_data, shared_mem);
  if(threadIdx.x == 0) {
    block_data.select_disjunct(*unified_data, d);
  }
  __syncthreads();
  FProp& iprop = *block_data.iprop;
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
  __shared__ BlockAsynchronousFixpointGPU<true> fp_engine;
#else
  __shared__ FixpointSubsetGPU<BlockAsynchronousFixpointGPU<true>, bt::global_allocator, CUDA_THREADS_PER_BLOCK> fp_engine;
  fp_engine.init(unified_data->root.max_num_deductions());
#endif
  auto group = cooperative_groups::this_thread_block();
  if(threadIdx.x == 0) {
    block_data.timer = block_data.stats.start_timer_device();
    block_data.start_time = block_data.timer;
    block_data.depth = 0;
  }
  root.stores[d]->copy_to(group, *block_data.store);
  __syncthreads();
  int rounds = 0;
  const int fp_iterations = node_fixpoint(*unified_data, block_data, fp_engine, root.config.jet_symbolic, rounds);
  __syncthreads();
  bool found = false;
  if(!iprop.is_bot()) {
    found = check_candidates(*unified_data, block_data, rounds > 0);
  }
  if(threadIdx.x == 0) {
    root.stats.fixpoint_iterations += fp_iterations;
    if(found) {
      block_data.inner_box->extract(*root.best);
      block_data.stats.timers.update_timer(Timer::LATEST_BEST_OBJ_FOUND, block_data.start_time);
      root.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND) = block_data.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND);
      root.found_disjunct = root.disjunct_ids[d];
      root.stats.solutions++;
    }
  }
  __syncthreads();
#ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
  fp_engine.destroy();
#endif
  block_data.deallocate_shared_data();
  __syncthreads();
}

template <class FPEngine>
__device__ INLINE void propagate(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data,
   FPEngine& fp_engine, bool& stop, bool& has_changed, bool& is_leaf_node)
{
  auto& config = unified_data.root.config;
  FProp& iprop = *block_data.iprop;
  auto group = cooperative_groups::this_thread_block();

  TIMEPOINT(SEARCH);
  if(threadIdx.x == 0) {
    is_leaf_node = false;
  }

  // II. Compute the fixpoint of the current node. With `-jet_sym_root_only`, the symbolic rounds were
  // done once on the root before the search (`gpu_root_symbolic`), and the search runs the interval
  // fixpoint only.
  int sym_rounds = 0;
  /** `-jet_backward spec`: the rows of the property in every node; `all`: all the rows in the nodes of
   * depth 0 (the dive) or multiple of `-jet_bwd_every`, the rows of the property in the others; `root`:
   * the root of each disjunct only (`gpu_root_symbolic`). */
  int backward = 0;
  if(config.jet_backward == JET_BACKWARD_SPEC) {
    backward = 1;
  }
  else if(config.jet_backward == JET_BACKWARD_ALL) {
    backward = (block_data.depth == 0 || (config.jet_bwd_every > 0 && block_data.depth % config.jet_bwd_every == 0)) ? 2 : 1;
  }
  const int fp_iterations = node_fixpoint(unified_data, block_data, fp_engine,
    config.jet_symbolic && !config.jet_sym_root_only, sym_rounds, backward);
  if(threadIdx.x == 0) {
    block_data.forms_current = sym_rounds > 0;
    block_data.bwd_current = backward != 0;
  }
  int num_active;
  TIMEPOINT(FIXPOINT);

  const auto current_strat = block_data.current_strategy;
  const auto& strat = grid_data.search_strategies[current_strat];
  const auto& store = *block_data.store;

  /** `-jet_candidates`: the candidate points of the node, at the depths multiple of `-jet_cand_every`
   * (at depth 0 only when it is 0). A counterexample found ends the search as a leaf solution; otherwise
   * the node is unchanged. */
  bool candidate_found = false;
  if(config.jet_candidates && !iprop.is_bot()
    && (block_data.depth == 0 || (config.jet_cand_every > 0 && block_data.depth % config.jet_cand_every == 0)))
  {
    candidate_found = check_candidates(unified_data, block_data, sym_rounds > 0);
  }

  // III. Analyze the result of propagation
  if(candidate_found) {
    if(threadIdx.x == 0) {
      is_leaf_node = true;
      block_data.stats.timers.update_timer(Timer::LATEST_BEST_OBJ_FOUND, block_data.start_time);
      block_data.solution_disjunct = block_data.disjunct;
      block_data.stats.solutions++;
      unified_data.stop.test_and_set();
    }
    __syncthreads();
    TIMEPOINT(SELECT_FP_FUNCTIONS);
  }
  else if(!iprop.is_bot()) {
    if(threadIdx.x == 0) {
      has_changed = false;
    }
    __syncthreads();
    // This is an underapproximation caes.
    for(int i = (int)group.thread_rank(); i < strat.vars.size(); i+=group.num_threads()){
      if(store[strat.vars[i].vid()].lb().value() != store[strat.vars[i].vid()].ub().value()){
        has_changed = true;
        break;
      }
    }
    __syncthreads();
    num_active = has_changed ? 1 : 0;
    TIMEPOINT(SELECT_FP_FUNCTIONS);
    if (num_active == 0) {
      is_leaf_node = true;
      /** `-jet_leaf_check strict` (default): the point of the leaf (its middle within the inner
       * bounds, the point itself when the inputs are points) is a solution only if it passes the
       * strict test (`check_candidates` without the corners), which saves it in `inner_box`.
       * Otherwise the leaf is unknown: its propagation is not empty, but that only says that the
       * outward-rounded intervals meet the region. `lenient`: the store of the leaf is the solution. */
      const bool strict = config.jet_leaf_check == JET_LEAF_STRICT;
      const bool certified = !strict || check_candidates(unified_data, block_data, false, false);
      if(certified) {
        if(threadIdx.x == 0) {
          block_data.stats.timers.update_timer(Timer::LATEST_BEST_OBJ_FOUND, block_data.start_time);
        }
        if(!strict) {
          block_data.store->copy_to(group, *block_data.inner_box);
        }
        if(threadIdx.x == 0) {
          block_data.solution_disjunct = block_data.disjunct;
          block_data.stats.solutions++;
          unified_data.stop.test_and_set();
        }
      }
      else if(threadIdx.x == 0) {
        block_data.stats.timers.update_timer(Timer::LATEST_BEST_OBJ_FOUND, block_data.start_time);
        block_data.stats.unknowns++;
      }
      __syncthreads();
    }
  }
  else {
    // This is unknown checking.
    // If is_bot() is true /\ exists at least 1 the width == 0.0, then it is identified as an unknown box.
    //   -> It can be simplified to check if it uses UASS or not.
    //   -> If we have applied UASS, it implies that there exists at least 1 variable is assigned.
    // If is_bot() is true /\ all the widths != 0.0, then it is pruned by propagation, not underapproximation.
    is_leaf_node = true;
    if (threadIdx.x == 0) {
      has_changed = uass::is_uass(block_data);
    }
    __syncthreads();
    num_active = has_changed ? 0 : 1;
    TIMEPOINT(SELECT_FP_FUNCTIONS);
    if (num_active == 0) {
      if(threadIdx.x == 0) {
        block_data.stats.timers.update_timer(Timer::LATEST_BEST_OBJ_FOUND, block_data.start_time);
        block_data.stats.unknowns++;
      }
    }
  }

  if(threadIdx.x == 0) {
    block_data.stats.fixpoint_iterations += fp_iterations;
    block_data.stats.nodes++;
    block_data.stats.fails += (iprop.is_bot() ? 1 : 0);
    block_data.stats.depth_max = battery::max(block_data.stats.depth_max, block_data.depth);

    // IV. Checking stopping conditions.

    if(block_data.stats.nodes >= config.stop_after_n_nodes
      || unified_data.stop.test()
      || block_data.stats.solutions != 0)
    {
      block_data.stats.exhaustive = false;
      stop = true;
    }
  }
}

__global__ void reduce_blocks(UnifiedData* unified_data, GridData* grid_data) {
  auto& root = unified_data->root;
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    root.stats.meet(grid_data->blocks[i].stats);
    int64_t& grid_first_block_idle = root.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE);
    int64_t block_idle = grid_data->blocks[i].stats.timers.time_of(Timer::FIRST_BLOCK_IDLE);
    if(grid_first_block_idle > block_idle) {
      grid_first_block_idle = block_idle;
    }
  }
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    auto& block = grid_data->blocks[i];
    if(block.stats.solutions > 0) {
      // FIXME: We might have more than one solution to remember.
      block.inner_box->extract(*root.best);
      root.found_disjunct = root.disjunct_ids[block.solution_disjunct];
      break;
    }
  }
}

__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>* grid_data) {
  grid_data->reset();
}

#endif // TURBO_IPC_ABSTRACT_DOMAIN
#endif // __CUDACC__

#if defined(TURBO_IPC_ABSTRACT_DOMAIN) || !defined(__CUDACC__)

void fbarebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {
#ifdef TURBO_IPC_ABSTRACT_DOMAIN
  std::cerr << "-arch fbarebones does not support IPC abstract domain." << std::endl;
#else
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
#endif
}

#endif

} // namespace fbarebones

#endif // TURBO_FASTFBAREBONES_DIVE_AND_SOLVE_HPP
