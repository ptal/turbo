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

namespace fbarebones {

#ifdef __CUDACC__
#ifndef TURBO_IPC_ABSTRACT_DOMAIN

/** `ConcurrentAllocator` allocates memory available both on CPU and GPU. For non-Linux systems such as Windows pinned memory must be used (see PR #19). */
#ifdef NO_CONCURRENT_MANAGED_MEMORY
  using ConcurrentAllocator = bt::pinned_allocator;
#else
  using ConcurrentAllocator = bt::managed_allocator;
#endif

using ::FItv;
using GridCP = AbstractDomains<FItv,
  bt::statistics_allocator<ConcurrentAllocator>,
  bt::statistics_allocator<UniqueLightAlloc<ConcurrentAllocator, 0>>,
  bt::statistics_allocator<UniqueLightAlloc<ConcurrentAllocator, 1>>>;

/** Fast neural network verification design on GPU. */
struct FastNNRelu {
  using NStore = VStore<FItv, bt::pool_allocator>;

  /** Declared BEFORE `neurons` so it is constructing first. */
  bt::pool_allocator neurons_pool;
  NStore neurons;

  int num_neurons;
  bt::vector<int> acc_layers;
  bt::vector<float> weights;
  bt::vector<float> biases;

  /** The pool is backed by managed memory so the store is reachable from host and device.
   * `pool_allocator` does not own the buffer: it stays alive as long as we do not free it. */
  static bt::pool_allocator make_neurons_pool(int num_neurons) {
    size_t bytes = size_t(num_neurons) * sizeof(FItv) + alignof(FItv);
    void* mem = bt::managed_allocator{}.allocate(bytes);   // returns nullptr when bytes == 0
    return bt::pool_allocator(static_cast<unsigned char*>(mem), bytes);
  }

  FastNNRelu()
    : neurons_pool(make_neurons_pool(0))
    , neurons(0, 0, neurons_pool)
    , num_neurons(0) 
    {}

  FastNNRelu(const int num_neurons, const bt::vector<int>& acc_layers,
             const bt::vector<float>& weights, const bt::vector<float>& biases, AType atype = 0)
    : neurons_pool(make_neurons_pool(num_neurons))
    , neurons(atype, num_neurons, neurons_pool)
    , num_neurons(num_neurons)
    , acc_layers(acc_layers)
    , weights(weights)
    , biases(biases) 
    {}

public:
  // `i` designates the target neuron among those that have a deduction, that is, every neuron but
  // those of the input layer: `i` ranges over `[0, num_deductions())` and updates the neuron
  // `neurons[acc_layers[1] + i]`. Consecutive `i` are therefore consecutive neurons of the same
  // layer, except across a layer boundary.
  // One thread handles one neuron: it reads the intervals of the neurons of the previous layer,
  // multiplies them by the weights of the connections into the target, adds its bias, applies the
  // ReLU, and merges the result into the store with a meet. The affine part and the ReLU are fused,
  // so the pre-activation never leaves the registers and a layer is updated in one deduction per
  // neuron, without any intra-warp reduction.
  // The sizes of the layers are read off `acc_layers` alone: the layer `k` has
  // `acc_layers[k+1] - acc_layers[k]` neurons.
  CUDA bool deduce(int i) {
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

    RItv zero(bound_type{0});
    r3 = RItv(neurons[target].lb().value(), neurons[target].ub().value());
    tell::fmax(r3, r2, zero);  // ReLU. Its backward pass narrows `r2` from the domain of neuron `j`.

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

  CUDA int num_deductions() const {
    return neurons.vars() - layers[0];  /**< the input layer has no deduction. */
  }

  CUDA void print() const {
    printf("In total, we have %d neurons in the network\n", (int)neurons.vars());
  }
};

/** Data shared between CPU and GPU. */
struct UnifiedData {
  /** The root node of the problem, useful to backtrack when solving a new subproblem.
   * Also contains the shared information such as statistics and solver configuration.
   */
  GridCP root;

  /** Stop signal from the CPU because of a timeout or CTRL-C. */
  cuda::std::atomic_flag stop;

  /** The memory configuration of each block. */
  MemoryConfig mem_config;

  UnifiedData(const CP<FItv>& cp, MemoryConfig mem_config)
   : root(GridCP::tag_gpu_block_copy{}, false, cp)
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
using FProp = PIR<FStore, bt::pool_allocator>;
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

  /** The gradients from the neural network. */
  // float* h_gradients;
  // float* h_mid_gradients;
  // float* h_lb_gradients;
  // float* h_ub_gradients;
  // int num_h_gradients;

  /* For underapproximation search strategy. */
  bool is_uass;

  __device__ BlockData()
   : subproblem_idx(0)
   , current_strategy(0)
   , next_unassigned_var(0)
   , decisions(5000)
   , depth(0)
  //  , h_gradients(nullptr)
  //  , h_mid_gradients(nullptr)
  //  , h_lb_gradients(nullptr)
  //  , h_ub_gradients(nullptr)
  //  , num_h_gradients(0)
   , is_uass(false)
  {}

  __device__ void allocate(const UnifiedData& unified_data, const GridData& grid_data, unsigned char* shared_mem) {
    if(threadIdx.x == 0) {
      subproblem_idx = blockIdx.x;
      const MemoryConfig& mem_config = unified_data.mem_config;
      const auto& u_store = *(unified_data.root.store);
      const auto& u_iprop = *(unified_data.root.iprop);
      bt::pool_allocator shared_mem_pool(mem_config.make_shared_pool(shared_mem));
      bt::pool_allocator store_allocator(mem_config.make_store_pool(shared_mem_pool));
      bt::pool_allocator prop_allocator(mem_config.make_prop_pool(shared_mem_pool));
      root_store = bt::make_shared<VStore<FItv, bt::global_allocator>, bt::global_allocator>(u_store);
      inner_box = bt::make_shared<VStore<FItv, bt::global_allocator>, bt::global_allocator>(u_store);
      store = bt::allocate_shared<FStore, bt::pool_allocator>(store_allocator, u_store, store_allocator);
      iprop = bt::allocate_shared<FProp, bt::pool_allocator>(prop_allocator, u_iprop, store, prop_allocator);

      // num_h_gradients = u_store.vars(); // only take input neurons.
      // size_t gradient_bytes = sizeof(float) * static_cast<size_t>(num_h_gradients) * 4;
      // void* gradient_mem = bt::global_allocator{}.allocate(gradient_bytes);
      // h_gradients = static_cast<float*>(gradient_mem);
      // h_mid_gradients = h_gradients + num_h_gradients;
      // h_lb_gradients = h_mid_gradients + num_h_gradients;
      // h_ub_gradients = h_lb_gradients + num_h_gradients;
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

  /**
   *
   * TODO: this function should be implemented later. It is for improving the performance.
   * */
  __device__ INLINE void floating_split(bool& has_changed, local::ZUB& idx,
    const StrategyType<bt::global_allocator>& strategy, const float epsilon)
  {
    bool split_in_store = strategy.vars.empty();
    int n = split_in_store ? store->vars() : strategy.vars.size();
    if(threadIdx.x == 0) {
      has_changed = true;
      idx = n;
    }
    __syncthreads();
    bt::vector<bool> E(n, false);
    for(int i = threadIdx.x; i < iprop->num_deductions(); i += blockDim.x) {
      if(!iprop->is_fsolution(i, epsilon)) {
        has_changed = true;
        E[iprop->load_deduce(i).x.vid()] = true;
        E[iprop->load_deduce(i).y.vid()] = true;
        E[iprop->load_deduce(i).z.vid()] = true;
      }
    }
    __syncthreads();
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
      const int dom_id = split_in_store ? i : strategy.vars[i].vid();
      const auto& dom = (*store)[dom_id];
      if(dom.width().ub().value() > epsilon && E[dom_id] && !dom.lb().is_top() && !dom.ub().is_top()) {
        if(idx.meet(local::ZUB(i))) {
          has_changed = true;
          break;
        }
      }
    }
    __syncthreads();
    if(threadIdx.x == 0) {
      next_unassigned_var = idx.value();
      if(next_unassigned_var != n) {
        push_decision(strategy.val_order, split_in_store ? AVar{store->aty(), next_unassigned_var} : strategy.vars[next_unassigned_var], epsilon);
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

  /** The objective variable to minimize.
   * Maximization problem are transformed into minimization problems by negating the objective variable.
   * Equal to -1 if the problem is a satisfaction problem.
   */
  AVar obj_var;

  __device__ GridData(const GridCP& root)
   : blocks(root.stats.num_blocks)
   , next_subproblem(root.stats.num_blocks)
   , print_lock(1)
   , has_eps_strategy(root.config.eps_var_order != "default")
   , search_strategies(root.split->strategies_())
   , obj_var(root.minimize_obj_var)
  {}
};

MemoryConfig configure_gpu_fbarebones(CP<FItv>&);
__global__ void initialize_global_data(UnifiedData*, bt::unique_ptr<GridData, bt::global_allocator>*);
__global__ void gpu_fbarebones_solve(UnifiedData*, GridData*);
template <class FPEngine>
__device__ INLINE void propagate(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data,
   FPEngine& fp_engine, bool& stop, bool& has_changed, bool& is_leaf_node);
// __device__ INLINE void back_propagation(BlockData& block_data, Ort::Session& session);
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

FastNNRelu parse_network(const Configuration<battery::standard_allocator>& config) {
  std::ifstream input(config.onnx_path.data(), std::ios::in | std::ios::binary);
  onnx::ModelProto network;

  if (!network.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse onnx file." << std::endl;
    return FastNNRelu();
  }

  const onnx::GraphProto& graph = network.graph();
  std::unordered_map<std::string, onnx::TensorProto> tensor_map;
  for (const auto& tensor : graph.initializer()) {
    tensor_map[tensor.name()] = tensor;
  }

  battery::vector<int> acc_layers;
  battery::vector<float> weights;
  battery::vector<float> biases;
  int total_neurons = 0;

  if(graph.input_size() == 0) {
    std::cerr << "The onnx graph has no input." << std::endl;
    return FastNNRelu();
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
  total_neurons += static_cast<int>(input_dimensions);

  for (const auto& node : graph.node()) {
    std::cout << "Node: " << node.output()[0] << "| OpType: " << node.op_type() << std::endl;

    if (node.op_type() == "Constant") { continue; }

    // Whether the weight matrix of this node is stored transposed, i.e. `[out_features, in_features]`.
    bool transB = false;
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "transB") { transB = attr.i(); }
    }

    for (int i = 1; i < node.input().size(); ++i) {
      const std::string input_name = node.input()[i];
      if (tensor_map.find(input_name) != tensor_map.end()) {
        const auto& tensor = tensor_map[input_name];
        if (tensor.dims().size() == 1) {
          // bias 1d tensor
          if(!read_float_tensor(tensor, biases)) {
            std::cerr << "ERROR: The biases of `" << input_name << "` are not stored as floats.\n";
            return FastNNRelu();
          }
        }
        else if (tensor.dims().size() == 2) {
          battery::vector<float> tmp_weights;
          if(!read_float_tensor(tensor, tmp_weights)) {
            std::cerr << "ERROR: The weights of `" << input_name << "` are not stored as floats.\n";
            return FastNNRelu();
          }

          /** `out_features` is the number of neurons of the new layer, `in_features` must match
           * the number of neurons of the previous layer. */
          int64_t out_features = transB ? tensor.dims(0) : tensor.dims(1);
          int64_t in_features = transB ? tensor.dims(1) : tensor.dims(0);

          /** We always store the weights column-major, all the output neurons of a given input
           * being contiguous (see the layout conventions of `FastNNRelu`). The ONNX layout
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
          total_neurons += static_cast<int>(out_features);

	  if(in_features != static_cast<int64_t>(acc_layers[i]-acc_layers[i-1])) {
            std::cerr << "ERROR: The weight matrix of `" << input_name << "` expects " << in_features
                      << " inputs but the previous layer has " << acc_layers[i]-acc_layers[i-1] << " neurons.\n";
            return FastNNRelu();
          }
          if(static_cast<int64_t>(tmp_weights.size()) != out_features * in_features) {
            std::cerr << "ERROR: The weight matrix of `" << input_name << "` has "
                      << tmp_weights.size() << " entries instead of " << out_features * in_features << ".\n";
            return FastNNRelu();
          }
        }
      }
    }
  }

  FastNNRelu fast_network(total_neurons, acc_layers, weights, biases);
  return fast_network;
}


void fbarebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {
  if(config.print_intermediate_solutions) {
    printf("%% WARNING: -arch fbarebones is incompatible with -i and -a (it cannot print intermediate solutions).\n");
  }
  auto start = std::chrono::steady_clock::now();
  check_support_managed_memory();
  check_support_concurrent_managed_memory();


  FastNNRelu fast_network = parse_network(config);
  fast_network.print();

  // /** We start with some preprocessing to reduce the number of variables and constraints. */
  CP<FItv> cp(config);
  cp.preprocess();
  if(cp.iprop->is_bot()) {
     cp.print_final_solution();
     cp.print_mzn_statistics();
     return;
   }

  MemoryConfig mem_config = configure_gpu_fbarebones(cp);
  auto unified_data = bt::make_unique<UnifiedData, ConcurrentAllocator>(cp, mem_config);
  auto grid_data = bt::make_unique<bt::unique_ptr<GridData, bt::global_allocator>, ConcurrentAllocator>();
  initialize_global_data<<<1,1>>>(unified_data.get(), grid_data.get());
  // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nnv");
  // Ort::SessionOptions session_options;
  // OrtCUDAProviderOptions cuda_options;
  // cuda_options.device_id = 0;
  // session_options.AppendExecutionProvider_CUDA(cuda_options);
  // Ort::Session session(env, config.onnx_path.data(), session_options);
  CUDAEX(cudaDeviceSynchronize());
  /** We wait that either the solving is interrupted, or that all threads have finished. */
  /** Block the signal CTRL-C to notify the threads if we must exit. */
  block_signal_ctrlc();
  gpu_fbarebones_solve
    <<<static_cast<unsigned int>(cp.stats.num_blocks),
      CUDA_THREADS_PER_BLOCK,
      mem_config.shared_bytes>>>
    (unified_data.get(), grid_data->get());
  auto now = std::chrono::steady_clock::now();
  int64_t time_to_kernel_start = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
  bool interrupted = wait_solving_ends(unified_data->stop, unified_data->root, start);
  CUDAEX(cudaDeviceSynchronize());
  reduce_blocks<<<1,1>>>(unified_data.get(), grid_data->get());
  CUDAEX(cudaDeviceSynchronize());
  auto& uroot = unified_data->root;
  if(uroot.stats.solutions > 0) {
    // We add the time before the kernel starts to the time needed to find the best bound.
    uroot.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND) += time_to_kernel_start;
    if(uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) != 0) {
      uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) += time_to_kernel_start;
    }
    cp.print_solution(*uroot.best);
  }
  uroot.stats.print_mzn_final_separator();
  if(uroot.config.print_statistics) {
    uroot.config.print_mzn_statistics();
    uroot.stats.print_mzn_statistics(uroot.config.verbose_solving);
    if(uroot.bab->is_optimization() && uroot.stats.solutions > 0) {
      uroot.stats.print_mzn_objective(uroot.best->project(uroot.bab->objective_var()), uroot.bab->is_minimization());
    }
    unified_data->root.stats.print_mzn_end_stats();
  }
  if (uroot.stats.solutions > 0) printf("sat\n");
  else if (uroot.stats.unknowns > 0) printf("unknown\n");
  else if (interrupted) printf("timeout\n");
  else printf("unsat\n");
  deallocate_global_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
}

/** We configure the GPU according to the user configuration:
 * 1) Guess the "best" number of blocks per SM, if not provided.
 * 2) Update the number of subproblems to at least "30 * B" where B is the number of blocks.
 * 3) Configure the size of the shared memory.
 * 4) Increase the global heap memory.
 * 5) Increase the stack size if requested by the user.
 */
MemoryConfig configure_gpu_fbarebones(CP<FItv>& cp) {
  auto& config = cp.config;

  /** I. Number of blocks per SM. */
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int max_block_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_block_per_sm, (void*) gpu_fbarebones_solve, CUDA_THREADS_PER_BLOCK, 0);
  if(cp.config.verbose_solving) {
    printf("%% max_blocks_per_sm=%d\n", max_block_per_sm);
  }
  if(cp.config.or_nodes != 0) {
    cp.stats.num_blocks = std::min(max_block_per_sm * deviceProp.multiProcessorCount, (int)cp.config.or_nodes);
    if(cp.config.verbose_solving >= 1 && cp.stats.num_blocks < cp.config.or_nodes) {
      printf("%% WARNING: -or %d is too high on your GPU architecture, it has been reduced to %d.\n", (int)cp.config.or_nodes, cp.stats.num_blocks);
    }
  }
  else {
    cp.stats.num_blocks = max_block_per_sm * deviceProp.multiProcessorCount;
  }

  /** II. Number of subproblems. */
  cp.stats.print_stat("subproblems_power", cp.config.subproblems_power);
  if(cp.config.subproblems_power == -1) {
    cp.config.subproblems_power = 0;
    while((1 << cp.config.subproblems_power) < cp.config.subproblems_factor * cp.stats.num_blocks) {
      cp.config.subproblems_power++;
    }
  }

  /** III. Size of the heap global memory.
   * The estimation is very conservative, normally we should not run out of memory.
   * */
  size_t store_bytes = gpu_sizeof<FStore>() + gpu_sizeof<abstract_ptr<FStore>>() + cp.store->vars() * gpu_sizeof<FItv>();
  size_t iprop_bytes = gpu_sizeof<FProp>() + gpu_sizeof<abstract_ptr<FProp>>() + cp.iprop->num_deductions() * gpu_sizeof<bytecode_type>() + gpu_sizeof<typename FProp::bytecodes_type>();
  // size_t gradient_bytes = sizeof(float) * static_cast<size_t>(cp.store->vars()) * 4;
  size_t mem_per_block = gpu_sizeof<BlockData>()
    + store_bytes * size_t{3}  // current, root, best.
    + store_bytes * size_t{2}  // search strategies
    + iprop_bytes * size_t{2}
    + cp.iprop->num_deductions() * size_t{4} * gpu_sizeof<bound_type>()  // fixpoint engine
    // + gradient_bytes
    + (gpu_sizeof<bound_type>() + gpu_sizeof<LightBranch<FItv>>()) * size_t{MAX_SEARCH_DEPTH};
  // size_t estimated_global_mem = gpu_sizeof<UnifiedData>() + store_bytes * size_t{5} + iprop_bytes + gradient_bytes +
  //   gpu_sizeof<GridData>();
  size_t estimated_global_mem = gpu_sizeof<UnifiedData>() + store_bytes * size_t{5} + iprop_bytes +
    gpu_sizeof<GridData>();

  size_t mem_for_blocks = deviceProp.totalGlobalMem - estimated_global_mem - (deviceProp.totalGlobalMem / 100 * 10);
  cp.stats.num_blocks = std::max(size_t{1}, std::min(mem_for_blocks / mem_per_block, static_cast<size_t>(cp.stats.num_blocks)));
  estimated_global_mem += cp.stats.num_blocks * mem_per_block;
  if(estimated_global_mem > deviceProp.totalGlobalMem / 100 * 90) {
    printf("%% WARNING: The estimated global memory is larger than 90%% of the total global memory.\n\
%% It is possible to run out of memory during solving.\n");
  }
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, deviceProp.totalGlobalMem / 100 * 97));
  cp.stats.print_memory_statistics(cp.config.verbose_solving, "heap_memory", estimated_global_mem);
  cp.stats.print_memory_statistics(cp.config.verbose_solving, "mem_per_block", mem_per_block);
  cp.stats.print_memory_statistics(cp.config.verbose_solving, "total_global_mem_bytes", deviceProp.totalGlobalMem);

  // We still need to improve this, for some large problems, it is required to avoid running out of memory.
  cp.stats.num_blocks = std::min(cp.stats.num_blocks, 200000000 / cp.store->vars());
  cp.stats.print_stat("num_blocks", cp.stats.num_blocks);

  /** IV. Increase the stack if requested by the user. */
  if(config.stack_kb != 0) {
    CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, config.stack_kb*1000));
    // The stack allocated depends on the maximum number of threads per SM, not on the actual number of threads per block.
    size_t total_stack_size = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor * config.stack_kb * 1000;
    cp.stats.print_memory_statistics(cp.config.verbose_solving, "stack_memory", total_stack_size);
  }

  /** V. Configure the shared memory size. */
  int blocks_per_sm = std::max(1, (cp.stats.num_blocks + deviceProp.multiProcessorCount - 1) / deviceProp.multiProcessorCount);
  MemoryConfig mem_config;
  if(config.only_global_memory) {
    mem_config = MemoryConfig(store_bytes, iprop_bytes);
  }
  else {
    mem_config = MemoryConfig((void*) gpu_fbarebones_solve, config.verbose_solving, blocks_per_sm, store_bytes, iprop_bytes);
  }
  mem_config.print_mzn_statistics(config, cp.stats);
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

void back_propagation(BlockData& block_data) {
  // // Step 1.
  // Ort::MemoryInfo cuda_mem_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);

  // // Step 2.
  // Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
  // battery::vector<int64_t> input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
  // size_t total_elements = 1;
  // for (size_t i = 0; i < input_dims.size(); ++i) {
  //   total_elements *= input_dims[i];
  // }

  // Ort::Value input_mid_tensor = Ort::Value::CreateTensor<float>(
  //   cuda_mem_info,
  //   block_data.h_mid_gradients,
  //   total_elements,
  //   input_dims.data(),
  //   input_dims.size()
  // );
  // Ort::Value input_lb_tensor = Ort::Value::CreateTensor<float>(
  //   cuda_mem_info,
  //   block_data.h_lb_gradients,
  //   total_elements,
  //   input_dims.data(),
  //   input_dims.size()
  // );
  // Ort::Value input_ub_tensor = Ort::Value::CreateTensor<float>(
  //   cuda_mem_info,
  //   block_data.h_ub_gradients,
  //   total_elements,
  //   input_dims.data(),
  //   input_dims.size()
  // );

  // Step 3.
  // Ort::RunOptions run_opts;
  // Ort::AllocatorWithDefaultOptions ort_allocator;
  // Ort::AllocatedStringPtr input_name_alloc = session.GetInputNameAllocated(0, ort_allocator);
  // std::string real_input_name = input_name_alloc.get();
  // Ort::AllocatedStringPtr output_name_alloc = session.GetOutputNameAllocated(0, ort_allocator);
  // std::string real_output_name = output_name_alloc.get();

  // const char* input_names[] = { real_input_name.c_str() };
  // const char* output_names[] = { real_output_name.c_str() };
  // const char* const* input_names_ptr = input_names;
  // const char* const* output_names_ptr = output_names;

  // Step 4.
  // std::vector<Ort::Value> output_mid_tensors;
  // std::vector<Ort::Value> output_lb_tensors;
  // std::vector<Ort::Value> output_ub_tensors;
  // output_mid_tensors = session.Run(
  //   run_opts,
  //   input_names_ptr,
  //   &input_mid_tensor,
  //   1,
  //   output_names_ptr,
  //   1
  // );
  // output_lb_tensors = session.Run(
  //   run_opts,
  //   input_names_ptr,
  //   &input_lb_tensor,
  //   1,
  //   output_names_ptr,
  //   1
  // );
  // output_ub_tensors = session.Run(
  //   run_opts,
  //   input_names_ptr,
  //   &input_ub_tensor,
  //   1,
  //   output_names_ptr,
  //   1
  // );

  // Step 5.
  // block_data.h_mid_gradients = output_mid_tensors[0].GetTensorMutableData<float>();
  // block_data.h_lb_gradients = output_lb_tensors[0].GetTensorMutableData<float>();
  // block_data.h_ub_gradients = output_ub_tensors[0].GetTensorMutableData<float>();

  // TODO: combine these gradients together. just use average
  // we also have to consider floating-point errors.
  // to simplicitly, we use only upper-towards rounding function
  // for(size_t i = 0; i < block_data.num_h_gradients; ++i){
    // block_data.h_gradients[i] = battery::div_up(battery::add_up(battery::add_up(block_data.h_mid_gradients[i], block_data.h_lb_gradients[i]), block_data.h_ub_gradients[i]), float{3.0});
    // block_data.h_gradients[i] = block_data.h_mid_gradients[i];
    // block_data.h_gradients[i] = block_data.h_ub_gradients[i];
  // }

  // cudaDeviceSynchronize();
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
    // else if(strat.var_order == VariableOrder::GRA_ANTI_FIRST_FAIL){
    //   // Obtain graident from the network by lbs, midpoints, and ubs.
    //   // We need to split this node.
    //   // By applying back_propagation(), we can have the latest gradient information.
    //   // This gradient information might not work. The code itself is correct, but it might not effective.
    //   for(int i = (int)group.thread_rank(); i < strat.vars.size(); i += group.num_threads()){
    //     block_data.h_mid_gradients[i] = battery::midpoint(store[i].lb().value(), store[i].ub().value());
    //     block_data.h_lb_gradients[i] = store[i].lb().value();
    //     block_data.h_ub_gradients[i] = store[i].ub().value();
    //   }
    //   __syncthreads();
    //   if(threadIdx.x == 0) {
    //     back_propagation(block_data, session);
    //   }
    // }
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
    // for (int i = (int)group.thread_rank(); i < strat.vars.size(); i += group.num_threads()){
    //   if(store[i].lb().value() == store[i].ub().value()){
    //     has_changed = true;
    //   }
    // }
    // __syncthreads();
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
      if(root.bab->is_satisfaction()) {
        // FIXME: We might have more than one solution to remember.
        block.inner_box->extract(*root.best);
        // for(int j = 0; j < block.inner_boxes.size(); ++j) {
        //   block.inner_boxes[j].extract(*root.best);
        //   root.inner_boxes.push_back(*root.best);
        //   if (j >= 10) break;
        // }
        break;
      }
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
