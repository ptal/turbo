// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_FASTFBAREBONES_DIVE_AND_SOLVE_HPP
#define TURBO_FASTFBAREBONES_DIVE_AND_SOLVE_HPP

#include "battery/allocator.hpp"
#include "common_solving.hpp"
#include "memory_gpu.hpp"
#include "lala/light_branch.hpp"
#include <mutex>
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

/** `ConcurrentAllocator` allocates memory available both on CPU and GPU. For non-Linux systems such as Windows pinned memory must be used (see PR #19). */
#ifdef NO_CONCURRENT_MANAGED_MEMORY
  using ConcurrentAllocator = bt::pinned_allocator;
#else
  using ConcurrentAllocator = bt::managed_allocator;
#endif

using ::FItv;
/** The read-only description of the neural network: its topology, weights and biases.
 * It is immutable during solving, so a single copy in managed memory is shared by every block;
 * only the store of neurons (`FastNNRelu::store`) is duplicated per block.
 *
 * Layout conventions:
 *  * The neurons of all the layers are numbered consecutively, `acc_layers[k]` being the index of
 *    the first neuron of the layer `k`, and the layer `k` having `acc_layers[k+1] - acc_layers[k]`
 *    neurons (the last layer ending at `num_neurons`).
 *  * `weights` stores the layers consecutively and, within a layer, column-major: the weight of the
 *    connection `(c, j)` of the layer `l` sits at `wbase(l) + c * layer_size(l) + j`.
 *  * `biases` has no entry for the input layer, hence the bias of the neuron `acc_layers[1] + i`
 *    is `biases[i]`.
 */
template <class Alloc>
struct FastNNReluNetwork {
  using allocator_type = Alloc;

  int num_neurons;
  bt::vector<int, Alloc> acc_layers;
  bt::vector<float, Alloc> weights;
  bt::vector<float, Alloc> biases;

  /** `has_relu[l]` tells whether the affine layer `l` is followed by a ReLU. It is indexed like
   * `acc_layers`, so the entry `0` (the input layer) is unused.
   * A ReLU cannot be assumed on every layer: in `tllverifybench` only every other layer has one
   * (the pattern is `..R.R.R...`), and applying a ReLU where the graph has none would cut off the
   * negative values of that layer and make the deduction unsound. */
  bt::vector<int, Alloc> has_relu;

  FastNNReluNetwork(const Alloc& alloc = Alloc{})
   : num_neurons(0), acc_layers(alloc), weights(alloc), biases(alloc), has_relu(alloc)
  {}

  template <class Alloc2>
  FastNNReluNetwork(int num_neurons,
    const bt::vector<int, Alloc2>& acc_layers,
    const bt::vector<float, Alloc2>& weights,
    const bt::vector<float, Alloc2>& biases,
    const bt::vector<int, Alloc2>& has_relu,
    const Alloc& alloc = Alloc{})
   : num_neurons(num_neurons)
   , acc_layers(acc_layers, alloc)
   , weights(weights, alloc)
   , biases(biases, alloc)
   , has_relu(has_relu, alloc)
  {}

  template <class Alloc2>
  FastNNReluNetwork(const FastNNReluNetwork<Alloc2>& other, const Alloc& alloc = Alloc{})
   : FastNNReluNetwork(other.num_neurons, other.acc_layers, other.weights, other.biases,
       other.has_relu, alloc)
  {}

  /** Whether the affine layer `l` is followed by a ReLU. */
  CUDA INLINE bool relu_at(int l) const {
    return l >= 0 && l < static_cast<int>(has_relu.size()) && has_relu[l] != 0;
  }

  FastNNReluNetwork(const FastNNReluNetwork&) = default;
  FastNNReluNetwork(FastNNReluNetwork&&) = default;

  /** The number of neurons of the input layer, the only ones we branch on. */
  CUDA INLINE int num_inputs() const {
    return acc_layers.size() >= 2 ? acc_layers[1] : num_neurons;
  }

  /** The index of the first neuron of the output layer.
   * `acc_layers` holds the *base* of each layer, so its last element is the base of the output
   * layer and is therefore strictly smaller than `num_neurons`:
   * \f$ acc\_layers.back() = num\_neurons - num\_outputs() \f$. */
  CUDA INLINE int output_base() const {
    return acc_layers.size() >= 2 ? acc_layers[acc_layers.size() - 1] : 0;
  }

  /** The number of neurons of the output layer. */
  CUDA INLINE int num_outputs() const {
    return num_neurons - output_base();
  }

  /** The number of neurons that have a deduction, that is every neuron but the input ones. */
  CUDA INLINE int num_deductions() const {
    return num_neurons - num_inputs();
  }

  CUDA INLINE int num_layers() const {
    return static_cast<int>(acc_layers.size());
  }

  /** A network we failed to parse, or with no hidden layer, has nothing to propagate. */
  CUDA bool empty() const {
    return num_neurons == 0 || acc_layers.size() < 2;
  }

  void print() const {
    printf("%% In total, we have %d neurons in the network (%d layers, %d inputs, %d outputs, %d deductions)\n",
      num_neurons, num_layers(), num_inputs(), num_outputs(), num_deductions());
    printf("%% ReLU per layer: ");
    for(int l = 1; l < num_layers(); ++l) { printf("%c", relu_at(l) ? 'R' : '.'); }
    printf("\n");
  }
};

/** A binary constraint of the property: \f$ neurons[x] - neurons[y] \leq k \f$.
 * The output constraints of `acasxu_2023` and `safenlp_2024` are comparisons between two output
 * neurons (`(assert (<= Y_0 Y_1))`), which is `x = Y_0, y = Y_1, k = 0`. A unary bound never
 * reaches here: it is embedded directly in the store of neurons at the root.
 */
struct NeuronLeq {
  int x;
  int y;
  float k;
};

/** The non-unary part of the property, shared read-only by all the blocks like the network.
 * It holds one conjunction of comparisons: a disjunctive property is expanded into DNF on the host
 * and each disjunct is solved as a separate run (see `load_property`).
 */
template <class Alloc>
struct FastProperty {
  using allocator_type = Alloc;

  bt::vector<NeuronLeq, Alloc> leqs;

  FastProperty(const Alloc& alloc = Alloc{}): leqs(alloc) {}

  template <class Alloc2>
  FastProperty(const bt::vector<NeuronLeq, Alloc2>& leqs, const Alloc& alloc = Alloc{})
   : leqs(leqs, alloc) {}

  template <class Alloc2>
  FastProperty(const FastProperty<Alloc2>& other, const Alloc& alloc = Alloc{})
   : leqs(other.leqs, alloc) {}

  FastProperty(const FastProperty&) = default;
  FastProperty(FastProperty&&) = default;
  FastProperty& operator=(const FastProperty&) = default;
  FastProperty& operator=(FastProperty&&) = default;

  CUDA INLINE int size() const { return static_cast<int>(leqs.size()); }
};

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

  CUDA FastNNRelu(const network_type* net, const property_type* prop, abstract_ptr<NStore> store)
   : net(net), prop(prop), store(store)
  {}

public:
  // `i` designates the target neuron among those that have a deduction, that is, every neuron but
  // those of the input layer: `i` ranges over `[0, num_deductions())` and updates the neuron
  // `neurons[acc_layers[1] + i]`. Consecutive `i` are therefore consecutive neurons of the same
  // layer, except across a layer boundary.
  // One thread handles one neuron: it reads the intervals of the neurons of the previous layer,
  // multiplies them by the weights of the connections into the target, adds its bias, applies the
  // ReLU (except on the output layer, which is affine only), and merges the result into the store
  // with a meet. The affine part and the ReLU are fused,
  // so the pre-activation never leaves the registers and a layer is updated in one deduction per
  // neuron, without any intra-warp reduction.
  // The sizes of the layers are read off `acc_layers` alone: the layer `k` has
  // `acc_layers[k+1] - acc_layers[k]` neurons.
  CUDA bool deduce_neuron(int i) {
    /** Local aliases so the deduction reads exactly as if the store and the network parameters were
     * members of this class. */
    auto& neurons = *store;
    const auto& acc_layers = net->acc_layers;
    const auto& weights = net->weights;
    const auto& biases = net->biases;
    const int num_neurons = net->num_neurons;
    using bound_type = typename FItv::LB::value_type;
    /** The interval of lala-interval, held in registers. Only the final result of the neuron is
     * merged back into `neurons`, which stores the shared `FItv` of the solver. */
    using RItv = FInterval<bound_type>;
    using local_itv = typename FItv::local_type;

    const int num_layers = static_cast<int>(acc_layers.size());
    assert(num_layers >= 2);
    assert(i >= 0 && i < num_neurons - (acc_layers[1] - acc_layers[0]));
    const int target = acc_layers[1] + i;  /**< the neuron of `neurons` that this deduction updates. */

    /** Locate the layer of `target`, accumulating the weight offset on the way: `l` is the last
     * layer whose first neuron is at or before `target`, and each layer `k < l` contributes
     * `layers[k] * layers[k-1]` weights before the block of the layer `l`. */
    int l = 1;
    int wbase = 0;
    while(l + 1 < num_layers && acc_layers[l+1] <= target) {
      wbase += (acc_layers[l+1] - acc_layers[l]) * (acc_layers[l] - acc_layers[l-1]);
      ++l;
    }

    const int out_base = acc_layers[l];
    const int layer_size = ((l + 1 < num_layers) ? acc_layers[l+1] : num_neurons) - out_base;
    const int prev_base = acc_layers[l-1];
    const int fan_in = out_base - prev_base;
    const int j = target - out_base;  /**< index of the target within its own layer. */
    assert(j >= 0 && j < layer_size);
    /** Weight of the connection `(c, j)`: `c * layer_size + j` within the block of the layer `l`,
     * the weights being stored column-major (see the layout conventions above). */
    wbase += j;

    RItv sum(bound_type{0});  /**< running pre-activation without the bias. */
    RItv r1,r2,r3;

    // STEP 1: compute the pre-activation \f$ s = \sum_c w_{jc} * x_c + b_j \f$ and store it in `sum`.

    /** Pre-activation \f$ s = \sum_c w_{jc} * x_c + b_j \f$. Each result is a variable that we just
     * reset to top, so we use the forward projections instead of the relational propagators: the
     * backward passes would only narrow operands that are already exact. */
    for(int c = 0; c < fan_in; ++c) {
      // Temporarily necessary to convert between the two kinds of interval (in lala-core and lala-interval).
      r1 = RItv(neurons[prev_base + c].lb().value(), neurons[prev_base + c].ub().value());
      r2 = RItv(static_cast<bound_type>(weights[wbase + c * layer_size]));
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

    /** `biases` has no entry for the input layer, hence the shift by the size of the layer 0,
     * which lands exactly on `i` when the neurons are numbered from `acc_layers[0] == 0`. */
    r1 = RItv(static_cast<bound_type>(biases[acc_layers[0] + i]));
    r2.join_top();

    /** Forward projection rather than `tell::fadd`: `add` takes its operands by value, so `sum`
     * provably keeps the plain forward accumulation that STEP 3 needs to undo. */
    r2.add(sum, r1);  // Pre-activation + bias.

    /** Whether the layer `l` is followed by a ReLU is read off the graph rather than assumed: the
     * output layer is usually affine only, but in `tllverifybench` every other hidden layer is
     * affine too. Applying a ReLU where the graph has none would cut off the negative values of the
     * layer, hence unsound conclusions on the property. */
    const bool apply_relu = net->relu_at(l);

    RItv zero(bound_type{0});
    r3 = RItv(neurons[target].lb().value(), neurons[target].ub().value());
    if(apply_relu) {
      tell::fmax(r3, r2, zero);  // ReLU. Its backward pass narrows `r2` from the domain of neuron `j`.
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

    /** `r2` is the pre-activation narrowed by the ReLU above, and `r1` still holds the bias. The
     * bias is a singleton, so pairing opposite bounds costs nothing and `sub` is exact here: `nsum`
     * is the narrowed value of \f$ \sum_c w_{jc} * x_c \f$, to be confronted with the forward
     * `sum`, which is left untouched by STEP 2 and still holds the plain forward accumulation. */
    RItv nsum;
    nsum.sub(r2, r1);
    if(!nsum.is_bot()) {
      RItv partial;  /**< the sum of all the *other* terms. */
      for(int c = 0; c < fan_in; ++c) {
        r1 = RItv(neurons[prev_base + c].lb().value(), neurons[prev_base + c].ub().value());
        r2 = RItv(static_cast<bound_type>(weights[wbase + c * layer_size]));
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
        has_changed |= neurons.embed(prev_base + c,
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

using NNStore = VStore<FItv, ConcurrentAllocator>;
using Network = FastNNReluNetwork<ConcurrentAllocator>;
using Property = FastProperty<ConcurrentAllocator>;

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

  /** The non-unary constraints of the disjunct being solved, shared by all the blocks.
   * A disjunctive property is expanded into DNF on the host and solved one disjunct per kernel
   * launch, so this is rewritten between two runs (see `fbarebones_dive_and_solve`). */
  Property prop;

  /** The bounds of the neurons at the root of the search: the input box and, the property being
   * unary in this application, the bounds of the output neurons expressing it. */
  abstract_ptr<NNStore> store;

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
   , prop(ConcurrentAllocator{})
   , store(bt::allocate_shared<NNStore, ConcurrentAllocator>(ConcurrentAllocator{},
             AType{0}, net.num_neurons, ConcurrentAllocator{}))
   , best(bt::allocate_shared<NNStore, ConcurrentAllocator>(ConcurrentAllocator{},
             AType{0}, net.num_neurons, ConcurrentAllocator{}))
   , var_order(var_order)
   , val_order(val_order)
  {}

  /** Same as `AbstractDomains::prune`, required by `must_quit` and `check_timeout`. */
  CUDA void prune() {
    stats.exhaustive = false;
  }
};

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

  template <class Alloc>
  UnifiedData(const Configuration<Alloc>& config, const Network& net,
    VariableOrder var_order, ValueOrder val_order, MemoryConfig mem_config)
   : root(config, net, var_order, val_order)
   , stop(false)
   , mem_config(mem_config)
  {
    size_t num_subproblems = 1;
    num_subproblems <<= root.config.subproblems_power;
    root.stats.eps_num_subproblems = num_subproblems;
  }
};

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

  /** The path from `UnifiedData::root` to the current subproblem `root_store`. */
  size_t subproblem_idx;

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

  /* For underapproximation search strategy. */
  bool is_uass;

  __device__ BlockData()
   : subproblem_idx(0)
   , current_strategy(0)
   , next_unassigned_var(0)
   , decisions(5000)
   , depth(0)
   , is_uass(false)
  {}

  __device__ void allocate(const UnifiedData& unified_data, const GridData& grid_data, unsigned char* shared_mem) {
    if(threadIdx.x == 0) {
      subproblem_idx = blockIdx.x;
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
        &(unified_data.root.net), &(unified_data.root.prop), store);
      is_uass = false;
    }
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
        if(dom.width().ub().value() > epsilon && !dom.lb().is_top() && !dom.ub().is_top()) {
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
        if(dom.width().ub().value() > epsilon && !dom.lb().is_top() && !dom.ub().is_top()) {
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
	      is_uass = false;
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
          if(dom.width().ub().value() > epsilon && !dom.lb().is_top() && !dom.ub().is_top() && f(dom,strategy.vars[i].vid()) == value) {
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

    /*
      Original search stategy. When using SPLIT && all widths <= epsilon, we check the solution with midpoints.
    */
    if(strategy.val_order == ValueOrder::SPLIT){
      if(threadIdx.x == 0){
        has_changed = false;
      }
      __syncthreads();
      for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x){
        const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
        if(dom.width().ub().value() <= epsilon && dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top()){
          has_changed = true;
        }
      }
      __syncthreads();
      if(has_changed) {
        for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x){
          AVar var = split_in_store ? AVar{store->aty(), i} : strategy.vars[i];
          const auto& dom = (*store)[var.vid()];
          auto mid = battery::midpoint(dom.lb().value(), dom.ub().value());
          store->embed(var, FItv(mid, mid));
        }
        __syncthreads();
        if(threadIdx.x == 0){
          is_uass = true;
          push_decision(strategy.val_order, strategy.vars[0], epsilon);
        }
      }
      return;
    }

    /*
    Underapproximation search strategy.
    */
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
      for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
        const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
        if(dom.width().ub().value() <= epsilon && dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top()){
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
	      is_uass = true;
      }
      __syncthreads();
      while(has_changed) {
        __syncthreads();
        has_changed = false;
        __syncthreads();
        for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
          const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
          if(dom.width().ub().value() <= epsilon && dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top() && f(dom, strategy.vars[i].vid()) == value){
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
    // printf("split on %d, lb = %.20f, ub = %.20f, mid = %.20f \n", decisions[depth].var.vid(), dom.lb().value(), dom.ub().value(), mid);

    switch(val_order) {
      case ValueOrder::SPLIT: {
				is_uass = false;
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
        if(dom.lb().value() != dom.ub().value() && dom.width().ub().value() <= epsilon) {
		      is_uass = true;
          decisions[depth].children[0] = FItv(dom.lb().value(), dom.lb().value());
        }
        else {
		      is_uass = false;
          decisions[depth].children[0] = FItv(dom.lb(), mid);
          decisions[depth].children[1] = FItv(mid, dom.ub());
        }
	      break;
      }
      case ValueOrder::UB_SPLIT: {
        if(dom.lb().value() != dom.ub().value() && dom.width().ub().value() <= epsilon) {
		      is_uass = true;
          decisions[depth].children[0] = FItv(dom.ub().value(), dom.ub().value());
        }
        else {
		      is_uass = false;
          decisions[depth].children[0] = FItv(dom.lb(), mid);
          decisions[depth].children[1] = FItv(mid, dom.ub());
        }
	      break;
      }
      case ValueOrder::MID_SPLIT: {
        if(dom.lb().value() != dom.ub().value() && dom.width().ub().value() <= epsilon) {
		      is_uass = true;
          decisions[depth].children[0] = FItv(mid, mid);
        }
        else {
		      is_uass = false;
          decisions[depth].children[0] = FItv(dom.lb(), mid);
          decisions[depth].children[1] = FItv(mid, dom.ub());
        }
	      break;
      }
      case ValueOrder::MID_LB_SPLIT: {
        if(dom.lb().value() != dom.ub().value() && dom.width().ub().value() <= epsilon) {
		      is_uass = true;
          decisions[depth].children[0] = FItv(mid, mid);
          decisions[depth].children[1] = FItv(dom.lb().value(), dom.lb().value());
        }
        else {
		      is_uass = false;
          decisions[depth].children[0] = FItv(dom.lb(), mid);
          decisions[depth].children[1] = FItv(mid, dom.ub());
        }
	      break;
      }
      case ValueOrder::MID_UB_SPLIT: {
        if(dom.lb().value() != dom.ub().value() && dom.width().ub().value() <= epsilon) {
		      is_uass = true;
          decisions[depth].children[0] = FItv(mid, mid);
          decisions[depth].children[1] = FItv(dom.ub().value(), dom.ub().value());
        }
        else {
		      is_uass = false;
          decisions[depth].children[0] = FItv(dom.lb(), mid);
          decisions[depth].children[1] = FItv(mid, dom.ub());
        }
	      break;
      }
      case ValueOrder::MIX_SPLIT: {
        // TODO: not complete yet. The current version doesn't support multiple chil nodes, only binary.
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
    /** Ropes are a mechanism for fast backtracking.
     * The rope of a left node is always the depth of the right node (also its depth), because after completing the exploration of the left subtree, we must visit the right subtree (rooted at the current depth).
     * The rope of the right node is inherited from its parent, we set -1 if there is no next node to visit.
     */
    decisions[depth].ropes[0] = depth + 1;
    decisions[depth].ropes[1] = depth > 0 ? decisions[depth-1].ropes[decisions[depth-1].current_idx] : -1;
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

  /** Due to multithreading, we must protect `stdout` when printing.
   * The model of computation in this work is lock-free, but it seems unavoidable for printing.
  */
  cuda::binary_semaphore<cuda::thread_scope_device> print_lock;

  /** A specific strategy is used for the subproblem decomposition during the diving phase. */
  bool has_eps_strategy;

  /** The search strategy is immutable and shared among the blocks. */
  strategies_type search_strategies;

  __device__ GridData(const NNRoot& root)
   : blocks(root.stats.num_blocks)
   , next_subproblem(root.stats.num_blocks)
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
  }
};

MemoryConfig configure_gpu_fbarebones(Configuration<battery::standard_allocator>&,
  Statistics<battery::standard_allocator>&, const Network&, int max_comparisons);
__global__ void initialize_global_data(UnifiedData*, bt::unique_ptr<GridData, bt::global_allocator>*);
__global__ void gpu_fbarebones_solve(UnifiedData*, GridData*);
template <class FPEngine>
__device__ INLINE void propagate(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data,
   FPEngine& fp_engine, bool& stop, bool& has_changed, bool& is_leaf_node);
__global__ void reduce_blocks(UnifiedData*, GridData*);
__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>*);

/** Read a float initializer, whether it is stored in `float_data` or in `raw_data`.
 * Returns false if the tensor is not a float tensor. */
static bool read_float_tensor(const onnx::TensorProto& tensor, battery::vector<float>& out) {
  if(tensor.data_type() != onnx::TensorProto::FLOAT) {
    return false;
  }
  if(tensor.float_data_size() > 0) {
    for(const auto& v : tensor.float_data()) {
      out.push_back(v);
    }
    return true;
  }
  /** Most exported models store the initializers in `raw_data` instead of `float_data`. */
  if(!tensor.raw_data().empty()) {
    const std::string& raw = tensor.raw_data();
    if(raw.size() % sizeof(float) != 0) {
      return false;
    }
    size_t n = raw.size() / sizeof(float);
    for(size_t i = 0; i < n; ++i) {
      float v;
      std::memcpy(&v, raw.data() + i * sizeof(float), sizeof(float));
      out.push_back(v);
    }
    return true;
  }
  return false;
}

Network parse_network(const Configuration<battery::standard_allocator>& config) {
  std::ifstream input(config.onnx_path.data(), std::ios::in | std::ios::binary);
  onnx::ModelProto network;

  if (!network.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse onnx file." << std::endl;
    return Network();
  }

  const onnx::GraphProto& graph = network.graph();
  std::unordered_map<std::string, onnx::TensorProto> tensor_map;
  for (const auto& tensor : graph.initializer()) {
    tensor_map[tensor.name()] = tensor;
  }

  battery::vector<int> acc_layers;
  battery::vector<float> weights;
  battery::vector<float> biases;
  battery::vector<int> has_relu;
  int total_neurons = 0;

  /** The name of the tensor that currently holds the value of the last layer added. It advances
   * through the bias `Add` of the layer, and through a `Relu` consuming it, which is how we detect
   * that the layer is followed by a ReLU. */
  std::string current_output;

  if(graph.input_size() == 0) {
    std::cerr << "The onnx graph has no input." << std::endl;
    return Network();
  }

  // number of input neurons
  const onnx::ValueInfoProto& graph_input = graph.input(0);
  const auto& input_shape = graph_input.type().tensor_type().shape();  // <batch_size, num_input_channels, input_H, input_W>;
  int64_t batch_size = input_shape.dim(0).dim_value() != 0 ? input_shape.dim(0).dim_value() : 1;
  int64_t input_channels = input_shape.dim().size() > 1 ? input_shape.dim(1).dim_value() : 1;
  int64_t input_height = input_shape.dim().size() > 2 ? input_shape.dim(2).dim_value() : 1;
  int64_t input_width = input_shape.dim().size() > 3 ? input_shape.dim(3).dim_value() : 1;
  int64_t input_dimensions = batch_size * input_channels * input_height * input_width;  // number of input neurons.
  acc_layers.push_back(0);
  has_relu.push_back(0);  /**< the input layer has no activation. */
  total_neurons += static_cast<int>(input_dimensions);

  /** The number of neurons of the last layer added so far, needed to check that the weight matrix
   * of the next layer has a matching number of inputs. It is NOT `node.input()`'s index: a node
   * carries at most one weight matrix, whatever its position among the inputs of the node. */
  int prev_layer_size = static_cast<int>(input_dimensions);

  for (const auto& node : graph.node()) {
    std::cout << "Node: " << node.output()[0] << "| OpType: " << node.op_type() << std::endl;

    if (node.op_type() == "Constant") { continue; }

    // Whether the weight matrix of this node is stored transposed, i.e. `[out_features, in_features]`.
    bool transB = false;
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "transB") { transB = attr.i(); }
    }

    /** We look for the initializers of this node in two passes, the weight matrix first, because
     * the bias of a layer must be appended after the layer itself has been added.
     * We scan the inputs from `0` and not from `1`: while `Gemm` takes the data first (`X, W, B`),
     * a `MatMul` exported as `MatMul(W, X)` carries its weights at the index `0`, and skipping it
     * silently dropped the whole layer. */
    const onnx::TensorProto* weight_tensor = nullptr;
    const onnx::TensorProto* bias_tensor = nullptr;
    int weight_index = -1;
    for (int i = 0; i < node.input().size(); ++i) {
      auto it = tensor_map.find(node.input()[i]);
      if (it == tensor_map.end()) { continue; }
      const auto& tensor = it->second;
      /** Only 1D and 2D initializers describe a layer. Anything else (e.g. the 4D normalization
       * constant of the `Sub` node of the ACAS-Xu models) is not part of the network we build. */
      if (tensor.dims().size() == 1 && bias_tensor == nullptr) { bias_tensor = &tensor; }
      else if (tensor.dims().size() == 2 && weight_tensor == nullptr) {
        weight_tensor = &tensor;
        weight_index = i;
      }
    }

    if (weight_tensor != nullptr) {
      const auto& tensor = *weight_tensor;
      /** Orientation of the weight matrix. `Gemm` computes \f$ X \cdot W \f$ with `W` given as
       * `[in_features, out_features]`, or transposed when `transB` is set. A `MatMul` exported as
       * `MatMul(W, X)` instead computes \f$ W \cdot X \f$, so a weight matrix found at the input
       * index `0` is transposed as well. */
      transB = transB || weight_index == 0;
      battery::vector<float> tmp_weights;
      if(!read_float_tensor(tensor, tmp_weights)) {
        std::cerr << "ERROR: The weights of `" << tensor.name() << "` are not stored as floats.\n";
        return Network();
      }

      /** `out_features` is the number of neurons of the new layer, `in_features` must match
       * the number of neurons of the previous layer. */
      int64_t out_features = transB ? tensor.dims(0) : tensor.dims(1);
      int64_t in_features = transB ? tensor.dims(1) : tensor.dims(0);

      /** `FastNNReluNetwork` represents a sequential feed-forward network, so each layer must take
       * its inputs from the previous one. A mismatch means the graph is a DAG with branches or skip
       * connections (e.g. the `cart_pole`/`quadrotor` models of VNN-COMP), which this
       * representation cannot express. */
      if(in_features != static_cast<int64_t>(prev_layer_size)) {
        std::cerr << "ERROR: The onnx graph is not a sequential feed-forward network: the weight matrix of `"
                  << tensor.name() << "` expects " << in_features
                  << " inputs but the previous layer has " << prev_layer_size << " neurons.\n";
        return Network();
      }
      if(static_cast<int64_t>(tmp_weights.size()) != out_features * in_features) {
        std::cerr << "ERROR: The weight matrix of `" << tensor.name() << "` has "
                  << tmp_weights.size() << " entries instead of " << out_features * in_features << ".\n";
        return Network();
      }

      /** We always store the weights column-major, all the output neurons of a given input
       * being contiguous (see the layout conventions of `FastNNReluNetwork`). The ONNX layout
       * `[in_features, out_features]` is already in that order, so only a matrix given
       * transposed, as `[out_features, in_features]`, has to be rearranged. */
      if(transB) {
        for(int64_t c = 0; c < in_features; ++c) {
          for(int64_t r = 0; r < out_features; ++r) {
            weights.push_back(tmp_weights[r * in_features + c]);
          }
        }
      }
      else {
        for(int64_t k = 0; k < out_features * in_features; ++k) {
          weights.push_back(tmp_weights[k]);
        }
      }

      // add the new layer
      acc_layers.push_back(total_neurons);
      has_relu.push_back(0);
      total_neurons += static_cast<int>(out_features);
      prev_layer_size = static_cast<int>(out_features);
      current_output = node.output_size() > 0 ? node.output()[0] : std::string();
    }

    if (bias_tensor != nullptr) {
      /** A bias belongs to the last layer added, whether that layer comes from this node (`Gemm`)
       * or from a previous one (`MatMul` followed by `Add`). We pad with zeros the layers of the
       * models that carry no bias at all, since `FastNNRelu::deduce` reads `biases[i]` for every
       * deduced neuron. */
      int deduced_so_far = total_neurons - static_cast<int>(input_dimensions);
      int base_of_last_layer = deduced_so_far - prev_layer_size;
      if(base_of_last_layer < 0) {
        std::cerr << "ERROR: The bias `" << bias_tensor->name() << "` comes before any layer.\n";
        return Network();
      }
      while(biases.size() < static_cast<size_t>(base_of_last_layer)) {
        biases.push_back(0.0f);
      }
      if(!read_float_tensor(*bias_tensor, biases)) {
        std::cerr << "ERROR: The biases of `" << bias_tensor->name() << "` are not stored as floats.\n";
        return Network();
      }
      /** The bias `Add` of a `MatMul`/`Add` pair produces the value of the layer. */
      if(node.output_size() > 0) { current_output = node.output()[0]; }
    }

    /** A `Relu` consuming the value of the last layer is the activation of that layer. We do not
     * assume one on every layer: `tllverifybench` alternates affine and ReLU layers, and
     * `acasxu`/`safenlp` have an affine output layer. */
    if(weight_tensor == nullptr && bias_tensor == nullptr && node.op_type() == "Relu"
       && acc_layers.size() >= 2 && !current_output.empty())
    {
      for (int i = 0; i < node.input().size(); ++i) {
        if(node.input()[i] == current_output) {
          has_relu[has_relu.size() - 1] = 1;
          if(node.output_size() > 0) { current_output = node.output()[0]; }
          break;
        }
      }
    }
  }

  if(acc_layers.size() < 2) {
    std::cerr << "ERROR: No layer could be read from the onnx graph.\n";
    return Network();
  }

  /** `acc_layers` holds the base index of each layer, so its last element is the base of the output
   * layer: it must be strictly smaller than `total_neurons`, the difference being the size of the
   * output layer. `total_neurons` itself counts every neuron of the network. */
  if(acc_layers[acc_layers.size()-1] >= total_neurons) {
    std::cerr << "ERROR: the last layer of the onnx graph is empty (acc_layers.back()="
              << acc_layers[acc_layers.size()-1] << ", total_neurons=" << total_neurons << ").\n";
    return Network();
  }

  /** `deduce` reads `biases[i]` for every deduced neuron, so the models that omit some (or all) of
   * their biases are padded with zeros rather than read out of bounds. */
  size_t expected_biases = static_cast<size_t>(total_neurons - input_dimensions);
  if(biases.size() < expected_biases) {
    std::cerr << "% WARNING: The onnx graph declares " << biases.size() << " biases instead of "
              << expected_biases << ", the missing ones are set to 0.\n";
    while(biases.size() < expected_biases) { biases.push_back(0.0f); }
  }
  else if(biases.size() > expected_biases) {
    std::cerr << "ERROR: The onnx graph declares " << biases.size() << " biases instead of "
              << expected_biases << ".\n";
    return Network();
  }

  return Network(total_neurons, acc_layers, weights, biases, has_relu);
}

/** A minimal s-expression: an atom when `children` is empty, a list otherwise.
 * It is enough for the subset of the vnnlib format we support (see `load_vnnlib`). */
struct SExpr {
  std::string atom;
  std::vector<SExpr> children;
  bool is_atom() const { return children.empty(); }
};

/** Split a vnnlib file into parentheses and atoms, dropping the `;` comments. */
static std::vector<std::string> vnnlib_tokenize(std::istream& in) {
  std::vector<std::string> tokens;
  std::string line;
  while(std::getline(in, line)) {
    size_t comment = line.find(';');
    if(comment != std::string::npos) { line.resize(comment); }
    std::string atom;
    for(char c : line) {
      if(c == '(' || c == ')') {
        if(!atom.empty()) { tokens.push_back(atom); atom.clear(); }
        tokens.push_back(std::string(1, c));
      }
      else if(std::isspace(static_cast<unsigned char>(c))) {
        if(!atom.empty()) { tokens.push_back(atom); atom.clear(); }
      }
      else { atom.push_back(c); }
    }
    if(!atom.empty()) { tokens.push_back(atom); }
  }
  return tokens;
}

/** Read one s-expression starting at `i`, which is advanced past it. */
static bool vnnlib_parse(const std::vector<std::string>& tokens, size_t& i, SExpr& out) {
  if(i >= tokens.size() || tokens[i] == ")") { return false; }
  if(tokens[i] != "(") { out.atom = tokens[i++]; return true; }
  ++i;  // consume '('
  while(i < tokens.size() && tokens[i] != ")") {
    SExpr child;
    if(!vnnlib_parse(tokens, i, child)) { return false; }
    out.children.push_back(std::move(child));
  }
  if(i >= tokens.size()) { return false; }  // missing ')'
  ++i;  // consume ')'
  /** An empty list would be indistinguishable from an atom, and we never expect one. */
  return !out.children.empty();
}

/** Convert a decimal literal into the tightest pair of doubles enclosing it, by reading it twice
 * with the rounding mode set outward. This is the same technique as `impl::string_to_real` of
 * `lala-parsing/flatzinc_parser.hpp`, and it is necessary because a decimal such as
 * `-0.303531156` is not representable: rounding it to the nearest double could move a bound inward
 * and cut off a part of the property. The weights and the biases need no such care, they are
 * already stored as floats in the onnx file. */
static void vnnlib_string_to_real(const std::string& str, double& lb, double& ub) {
  #if !defined(__GNUC__) && !defined(_MSC_VER)
    #pragma STDC FENV_ACCESS ON
  #endif
  int old_round = std::fegetround();
  std::fesetround(FE_DOWNWARD);
  lb = std::strtod(str.c_str(), nullptr);
  std::fesetround(FE_UPWARD);
  ub = std::strtod(str.c_str(), nullptr);
  std::fesetround(old_round);
}

/** A term of an assertion: either a neuron of the store, or a numeric literal.
 * A literal is kept as the enclosure \f$ [lb, ub] \f$ of its exact decimal value, so that each
 * bound can later be rounded in the direction that widens the property. */
struct VnnTerm {
  bool is_var;
  int var;    /**< index of the neuron in the store, when `is_var`. */
  double lb;  /**< the literal rounded downward, otherwise. */
  double ub;  /**< the literal rounded upward, otherwise. */
};

/** `X_i` is the input neuron `i` and `Y_j` the output neuron `j`, which lives at
 * `net.output_base() + j` in the store since all the neurons share a single numbering. */
static bool vnnlib_term(const SExpr& e, const Network& net, VnnTerm& out) {
  if(e.is_atom()) {
    const std::string& a = e.atom;
    if(a.size() > 2 && (a[0] == 'X' || a[0] == 'Y') && a[1] == '_') {
      char* endp = nullptr;
      long idx = std::strtol(a.c_str() + 2, &endp, 10);
      if(endp == a.c_str() + 2 || *endp != '\0' || idx < 0) {
        std::cerr << "ERROR: `" << a << "` is not a valid neuron name.\n";
        return false;
      }
      const int limit = (a[0] == 'X') ? net.num_inputs() : net.num_outputs();
      if(idx >= limit) {
        std::cerr << "ERROR: `" << a << "` is out of range, the network has " << limit
                  << (a[0] == 'X' ? " input" : " output") << " neurons.\n";
        return false;
      }
      const int base = (a[0] == 'X') ? 0 : net.output_base();
      out = VnnTerm{true, base + static_cast<int>(idx), 0.0, 0.0};
      return true;
    }
    char* endp = nullptr;
    std::strtod(a.c_str(), &endp);  /**< only to validate that the whole atom is a number. */
    if(endp == a.c_str() || *endp != '\0') {
      std::cerr << "ERROR: `" << a << "` is neither a neuron nor a number.\n";
      return false;
    }
    double lb, ub;
    vnnlib_string_to_real(a, lb, ub);
    out = VnnTerm{false, -1, lb, ub};
    return true;
  }
  /** Some vnnlib files write the negative literals as `(- 1.5)`. Negation is exact in floating
   * point, but it swaps the two ends of the enclosure. */
  if(e.children.size() == 2 && e.children[0].is_atom() && e.children[0].atom == "-") {
    VnnTerm t;
    if(!vnnlib_term(e.children[1], net, t) || t.is_var) { return false; }
    out = VnnTerm{false, -1, -t.ub, -t.lb};
    return true;
  }
  return false;
}

/** One atom of the property, once interpreted.
 *  * unary:  `neurons[x] <= k` when `upper`, `neurons[x] >= k` otherwise. It is embedded directly
 *    in the store of neurons at the root, and costs no propagator.
 *  * binary: `neurons[x] - neurons[y] <= k`, which becomes a `NeuronLeq` propagated by
 *    `FastNNRelu::deduce_property`.
 */
struct PropAtom {
  bool is_binary;
  int x;
  int y;
  /** The enclosure of the constant. `apply_disjunct` picks the end that widens the property:
   * `ub` for an upper bound, `lb` for a lower bound. */
  double k_lb;
  double k_ub;
  bool upper;
};

/** A property in disjunctive normal form: a disjunction of conjunctions of atoms.
 * `acasxu_2023` needs it, both for its output constraints
 * (`(assert (or (and (<= Y_1 Y_0)) (and (<= Y_2 Y_0)) ...))`) and, for `prop_6` and `prop_8`, for
 * its input box, which is a union of two boxes. Each disjunct is a box plus a conjunction of
 * comparisons, which is exactly what one run of the solver handles, so we solve the disjuncts one
 * after the other. */
using Conjunction = std::vector<PropAtom>;
using DNF = std::vector<Conjunction>;

/** Guard against the combinatorial explosion of the DNF expansion. The benchmarks we target stay
 * far below it (`acasxu`'s `prop_6` is the largest with 2 * 4 = 8 disjuncts). */
#define MAX_PROPERTY_DISJUNCTS 4096

/** `out = a /\ b` in DNF, i.e. the pairwise union of their conjunctions. */
static bool dnf_and(const DNF& a, const DNF& b, DNF& out) {
  if(a.size() * b.size() > MAX_PROPERTY_DISJUNCTS) {
    std::cerr << "ERROR: the property has more than " << MAX_PROPERTY_DISJUNCTS
              << " disjuncts once expanded into DNF.\n";
    return false;
  }
  out.clear();
  out.reserve(a.size() * b.size());
  for(const Conjunction& ca : a) {
    for(const Conjunction& cb : b) {
      Conjunction c(ca);
      c.insert(c.end(), cb.begin(), cb.end());
      out.push_back(std::move(c));
    }
  }
  return true;
}

/** Interpret one formula into a DNF over the neurons.
 * An empty DNF means `false` (no disjunct can be satisfied), while a DNF holding one empty
 * conjunction means `true` (no constraint at all).
 */
static bool vnnlib_dnf(const SExpr& e, const Network& net, DNF& out) {
  out.clear();
  if(e.is_atom()) {
    /** `true` and `false` are the only atoms that can stand as a formula by themselves. */
    if(e.atom == "true") { out.push_back(Conjunction{}); return true; }
    if(e.atom == "false") { return true; }  // empty DNF
    std::cerr << "ERROR: unexpected formula `" << e.atom << "`.\n";
    return false;
  }
  if(!e.children[0].is_atom()) {
    std::cerr << "ERROR: the head of a formula must be an operator.\n";
    return false;
  }
  const std::string& op = e.children[0].atom;

  if(op == "and") {
    out.push_back(Conjunction{});  // neutral element
    for(size_t k = 1; k < e.children.size(); ++k) {
      DNF child;
      if(!vnnlib_dnf(e.children[k], net, child)) { return false; }
      DNF merged;
      if(!dnf_and(out, child, merged)) { return false; }
      out = std::move(merged);
    }
    return true;
  }
  if(op == "or") {
    for(size_t k = 1; k < e.children.size(); ++k) {
      DNF child;
      if(!vnnlib_dnf(e.children[k], net, child)) { return false; }
      if(out.size() + child.size() > MAX_PROPERTY_DISJUNCTS) {
        std::cerr << "ERROR: the property has more than " << MAX_PROPERTY_DISJUNCTS << " disjuncts.\n";
        return false;
      }
      for(Conjunction& c : child) { out.push_back(std::move(c)); }
    }
    return true;
  }

  const bool is_le = (op == "<=" || op == "<");
  const bool is_ge = (op == ">=" || op == ">");
  const bool is_eq = (op == "=");
  if(!is_le && !is_ge && !is_eq) {
    std::cerr << "ERROR: unsupported operator `" << op << "` in the vnnlib file.\n";
    return false;
  }
  if(e.children.size() != 3) {
    std::cerr << "ERROR: `" << op << "` expects two arguments.\n";
    return false;
  }
  if(op == "<" || op == ">") {
    printf("%% WARNING: the strict comparison `%s` is relaxed into its non-strict version, which\n\
%% over-approximates the property.\n", op.c_str());
  }

  VnnTerm lhs, rhs;
  if(!vnnlib_term(e.children[1], net, lhs) || !vnnlib_term(e.children[2], net, rhs)) {
    return false;
  }

  Conjunction conj;
  if(lhs.is_var && rhs.is_var) {
    /** `x <= y` is `x - y <= 0`, and `x >= y` is `y - x <= 0`. This is the shape of the output
     * constraints of `acasxu_2023` and `safenlp_2024`. The constant `0` is exact. */
    if(is_le || is_eq) { conj.push_back(PropAtom{true, lhs.var, rhs.var, 0.0, 0.0, true}); }
    if(is_ge || is_eq) { conj.push_back(PropAtom{true, rhs.var, lhs.var, 0.0, 0.0, true}); }
  }
  else if(!lhs.is_var && !rhs.is_var) {
    /** A comparison between two constants is either vacuous or unsatisfiable. We keep it when it
     * *can* hold on the enclosures, so that a rounding never removes a disjunct. */
    bool holds = is_eq ? (lhs.lb <= rhs.ub && rhs.lb <= lhs.ub)
               : (is_le ? (lhs.lb <= rhs.ub) : (lhs.ub >= rhs.lb));
    if(!holds) { return true; }  // empty DNF, i.e. `false`
  }
  else {
    /** We normalize to `var op constant`, flipping the operator when the neuron is on the right. */
    const int var = lhs.is_var ? lhs.var : rhs.var;
    const VnnTerm& c = lhs.is_var ? rhs : lhs;
    const bool upper = lhs.is_var ? is_le : is_ge;
    if(is_eq) {
      conj.push_back(PropAtom{false, var, -1, c.lb, c.ub, true});
      conj.push_back(PropAtom{false, var, -1, c.lb, c.ub, false});
    }
    else {
      conj.push_back(PropAtom{false, var, -1, c.lb, c.ub, upper});
    }
  }
  out.push_back(std::move(conj));
  return true;
}

/** Load the property of the vnnlib file as a DNF over the neurons.
 *
 * `X_i` is the input neuron `i` and `Y_j` the output neuron `net.output_base() + j`. Following the
 * vnnlib convention, the assertions describe the region we are looking for: a point of that region
 * is a counterexample (`sat`), and an empty region is a proof that the property holds (`unsat`).
 * The assertions of a file are conjoined, and the result is expanded into DNF because a single run
 * of the solver explores one box with one conjunction of comparisons.
 *
 * Returns `false` on a malformed or unsupported file, in which case the search must not start.
 */
bool load_property(const Configuration<battery::standard_allocator>& config,
  const Network& net, DNF& dnf)
{
  dnf.clear();
  dnf.push_back(Conjunction{});  /**< `true`: a single disjunct with no constraint. */
  if(config.vnnlib_path.size() == 0) {
    printf("%% WARNING: no vnnlib file given (-vnnlib_path), every neuron is left unbounded, hence\n\
%% the input box is unbounded and the property is trivially satisfiable.\n");
    return true;
  }
  std::ifstream in(config.vnnlib_path.data());
  if(!in) {
    std::cerr << "ERROR: cannot open the vnnlib file `" << config.vnnlib_path.data() << "`.\n";
    return false;
  }
  std::vector<std::string> tokens = vnnlib_tokenize(in);
  size_t i = 0;
  int num_assertions = 0;
  while(i < tokens.size()) {
    SExpr e;
    if(!vnnlib_parse(tokens, i, e)) {
      std::cerr << "ERROR: syntax error in the vnnlib file `" << config.vnnlib_path.data() << "`.\n";
      return false;
    }
    /** `declare-const`, `set-logic`, ... carry no constraint, we skip them. */
    if(e.is_atom() || !e.children[0].is_atom() || e.children[0].atom != "assert") { continue; }
    if(e.children.size() != 2) {
      std::cerr << "ERROR: `assert` expects a single formula.\n";
      return false;
    }
    DNF child;
    if(!vnnlib_dnf(e.children[1], net, child)) { return false; }
    DNF merged;
    if(!dnf_and(dnf, child, merged)) { return false; }
    dnf = std::move(merged);
    ++num_assertions;
  }
  if(config.verbose_solving >= 1) {
    printf("%% Loaded %d assertions from `%s`, expanded into %d disjunct(s).\n",
      num_assertions, config.vnnlib_path.data(), (int)dnf.size());
  }
  return true;
}

/** Install one disjunct: its unary atoms become the bounds of the neurons at the root, and its
 * binary atoms become the comparisons propagated by `FastNNRelu::deduce_property`.
 * `neurons` must arrive with every neuron at top, i.e. `]-oo, +oo[`; any neuron left untouched
 * keeps that value.
 */
void apply_disjunct(const Conjunction& conj, NNStore& neurons,
  battery::vector<NeuronLeq>& leqs)
{
  using local_itv = typename NNStore::universe_type::local_type;
  using LB2 = typename local_itv::LB;
  using UB2 = typename local_itv::UB;
  using bound_t = typename LB2::value_type;
  leqs.clear();
  for(const PropAtom& a : conj) {
    /** The cast from `double` to `bound_t` (a `float` unless `TURBO_ITV_BITS == 64`) is a second
     * rounding, so it is directed too: `ru_cast` for an upper bound and `rd_cast` for a lower one.
     * Both widen the property, which never cuts off a counterexample. */
    if(a.is_binary) {
      /** `neurons[x] - neurons[y] <= k`: the largest `k` is the weakest constraint. */
      leqs.push_back(NeuronLeq{a.x, a.y, battery::ru_cast<float>(a.k_ub)});
    }
    else if(a.upper) {
      neurons.embed(a.x, local_itv(LB2::top(), UB2(battery::ru_cast<bound_t>(a.k_ub))));
    }
    else {
      neurons.embed(a.x, local_itv(LB2(battery::rd_cast<bound_t>(a.k_lb)), UB2::top()));
    }
  }
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
  Statistics<battery::standard_allocator> hstats(
    static_cast<size_t>(net.num_neurons), static_cast<size_t>(net.num_deductions()),
    false, config.print_statistics);

  VariableOrder var_order;
  ValueOrder val_order;
  resolve_search_strategy(hconfig, var_order, val_order);

  /** III. The property. A disjunctive property (the `acasxu` output constraints, and the union of
   * two input boxes of `prop_6`/`prop_8`) is expanded into DNF: each disjunct is one box plus one
   * conjunction of comparisons, which is exactly what a run of the solver explores. The answer is
   * `sat` as soon as one disjunct is `sat`, and `unsat` only when all of them are. */
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

  /** Block the signal CTRL-C to notify the threads if we must exit. */
  block_signal_ctrlc();

  for(size_t d = 0; d < dnf.size() && !interrupted && uroot.stats.solutions == 0; ++d) {
    if(hconfig.verbose_solving >= 1 && dnf.size() > 1) {
      printf("%% Solving disjunct %d/%d of the property.\n", (int)d + 1, (int)dnf.size());
    }

    /** A fresh store of neurons, all at top, on which the disjunct is installed. Re-allocating is
     * simpler than resetting, and happens at most once per disjunct. */
    uroot.store = bt::allocate_shared<NNStore, ConcurrentAllocator>(ConcurrentAllocator{},
      AType{0}, net.num_neurons, ConcurrentAllocator{});
    apply_disjunct(dnf[d], *(uroot.store), leqs);
    uroot.prop = Property(leqs, ConcurrentAllocator{});

    if(uroot.store->is_bot()) {
      if(hconfig.verbose_solving >= 1) {
        printf("%% The disjunct is inconsistent before any propagation, skipping it.\n");
      }
      continue;
    }
    if(hconfig.verbose_solving >= 2) {
      print_neurons(net, *(uroot.store));
    }

    auto grid_data = bt::make_unique<bt::unique_ptr<GridData, bt::global_allocator>, ConcurrentAllocator>();
    initialize_global_data<<<1,1>>>(unified_data.get(), grid_data.get());
    CUDAEX(cudaDeviceSynchronize());
    /** We wait that either the solving is interrupted, or that all threads have finished. */
    gpu_fbarebones_solve
      <<<static_cast<unsigned int>(uroot.stats.num_blocks),
        CUDA_THREADS_PER_BLOCK,
        mem_config.shared_bytes>>>
      (unified_data.get(), grid_data->get());
    auto now = std::chrono::steady_clock::now();
    if(d == 0) {
      time_to_kernel_start = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
    }
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
    uroot.stats.print_mzn_end_stats();
  }
  if (uroot.stats.solutions > 0) printf("sat\n");
  else if (uroot.stats.unknowns > 0) printf("unknown\n");
  else if (interrupted) printf("timeout\n");
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
  fp_engine.init(iprop.num_deductions());
#endif
  /** This shared variable is necessary to avoid multiple threads to read into `unified_data.stop.test()`,
   * potentially reading different values and leading to deadlock. */
  __shared__ bool stop;
  __shared__ bool has_changed;
  __shared__ bool is_leaf_node;
  __shared__ int remaining_depth;
  stop = false;
  auto group = cooperative_groups::this_thread_block();
  if(threadIdx.x == 0) {
    block_data.timer = block_data.stats.start_timer_device();
    block_data.start_time = block_data.timer;
  }
  __syncthreads();

  /** B. Start the main dive and solve loop. */
  size_t num_subproblems = unified_data->root.stats.eps_num_subproblems;
  while(block_data.subproblem_idx < num_subproblems && !stop) {
    if(config.verbose_solving >= 2 && threadIdx.x == 0) {
      grid_data->print_lock.acquire();
      printf("%% Block %d solves subproblem num %" PRIu64 "\n", blockIdx.x, block_data.subproblem_idx);
      grid_data->print_lock.release();
    }

    // C. Restoring the current state to the root node.

    block_data.current_strategy = 0;
    block_data.next_unassigned_var = 0;
    block_data.depth = 0;
    unified_data->root.store->copy_to(group, *block_data.store);
#ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
    fp_engine.reset(iprop.num_deductions());
#endif
    __syncthreads();

    // D. Dive into the search tree until we reach the target subproblem.
    remaining_depth = config.subproblems_power;
    if(threadIdx.x == 0) {
      block_data.dive_timer = block_data.stats.start_timer_device();
      is_leaf_node = false;
    }
    __syncthreads();
    while(remaining_depth > 0 && !is_leaf_node && !stop) {
      __syncthreads();
      propagate(*unified_data, *grid_data, block_data, fp_engine, stop, has_changed, is_leaf_node);
      __syncthreads();
      if(!is_leaf_node) {
        block_data.split(has_changed, grid_data->search_strategies, config.epsilon);
        __syncthreads();
        // Split was not able to split a domain. It means that the search strategy is not complete due to unsplittable infinite domains.
        // We skip the subtree, and set exhaustive to `false`.
        if(block_data.decisions[0].var.is_untyped()) {
          is_leaf_node = true;
          block_data.stats.exhaustive = false;
          if(threadIdx.x == 0 && config.verbose_solving >= 1) { printf("%% WARNING: infinite element detected during branching, search is not exhaustive\n");}
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
          block_data.store->embed(block_data.decisions[0].var, block_data.decisions[0].children[branch_idx]);
        }
      }
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      block_data.stats.stop_timer(Timer::DIVE, block_data.dive_timer);
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
        size_t next_subproblem_idx = ((block_data.subproblem_idx >> remaining_depth) + size_t{1}) << remaining_depth;
        // Make sure the subtree is skipped.
        while(grid_data->next_subproblem.meet(ZLB<size_t, bt::local_memory>(next_subproblem_idx))) {}
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

        // I. Propagate the current node.
        propagate(*unified_data, *grid_data, block_data, fp_engine, stop, has_changed, is_leaf_node);
        __syncthreads();

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
          block_data.split(has_changed, grid_data->search_strategies, config.epsilon);
          __syncthreads();
          // Split was not able to split a domain. It means that the search strategy is not complete due to unsplittable infinite domains.
          // We trigger backtracking, and set exhaustive to `false`.
          if(block_data.decisions[block_data.depth - 1].var.is_untyped()) {
            is_leaf_node = true;
            block_data.stats.exhaustive = false;
            if(threadIdx.x == 0 && config.verbose_solving >= 1) { printf("%% WARNING: infinite element detected during branching, search is not exhaustive\n");}
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
      /** To avoid that several blocks solve the same subproblem, we use an atomic post-increment. */
      block_data.subproblem_idx = grid_data->next_subproblem.atomic()++;
      /** The following commented code is completely valid and does not use atomic post-increment.
       * But honestly, we kinda need more performance so... let's avoid reexploring subproblems. */
      // subproblem_idx = grid_data->next_subproblem.value();
      // grid_data->next_subproblem.meet(FLB<size_t, bt::local_memory>(subproblem_idx + size_t{1}));
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


template <class FPEngine>
__device__ INLINE void propagate(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data,
   FPEngine& fp_engine, bool& stop, bool& has_changed, bool& is_leaf_node)
{
  __shared__ int warp_iterations[CUDA_THREADS_PER_BLOCK/32];
  warp_iterations[threadIdx.x / 32] = 0;
  auto& config = unified_data.root.config;
  FProp& iprop = *block_data.iprop;
  auto group = cooperative_groups::this_thread_block();

  TIMEPOINT(SEARCH);
  if(threadIdx.x == 0) {
    is_leaf_node = false;
  }

  // II. Compute the fixpoint of the current node.
  int fp_iterations;
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
  int num_active = iprop.num_deductions();
#else
  int num_active = fp_engine.num_active();
#endif
  switch(config.fixpoint) {
    case FixpointKind::AC1: {
      fp_iterations = fp_engine.fixpoint(
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
        iprop.num_deductions(),
#endif
        [&](int i){ return iprop.fdeduce(i, config.epsilon); },
        [&](){ return iprop.is_bot(); });
      if(threadIdx.x == 0) {
        block_data.stats.num_deductions += fp_iterations * num_active;
      }
      break;
    }
    case FixpointKind::WAC1: {
      if(num_active <= config.wac1_threshold) {
        fp_iterations = fp_engine.fixpoint(
#ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
        iprop.num_deductions(),
#endif
          [&](int i){ return iprop.fdeduce(i, config.epsilon); },
          [&](){ return iprop.is_bot(); });
        if(threadIdx.x == 0) {
          block_data.stats.num_deductions += fp_iterations * num_active;
        }
      }
      else {
        fp_iterations = fp_engine.fixpoint(
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
  TIMEPOINT(FIXPOINT);

  const auto current_strat = block_data.current_strategy;
  const auto& strat = grid_data.search_strategies[current_strat];
  const auto& store = *block_data.store;

  // III. Analyze the result of propagation
  if(!iprop.is_bot()) {
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
      if(threadIdx.x == 0) {
        block_data.stats.timers.update_timer(Timer::LATEST_BEST_OBJ_FOUND, block_data.start_time);
      }
      block_data.store->copy_to(group, *block_data.inner_box);
      if(threadIdx.x == 0) {
        block_data.stats.solutions++;
        unified_data.stop.test_and_set();
      }
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
      has_changed = block_data.is_uass;
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
