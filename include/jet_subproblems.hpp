// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_SUBPROBLEMS_HPP
#define TURBO_JET_SUBPROBLEMS_HPP

#include "battery/utility.hpp"

/** The subproblems of `-arch jet` over the disjuncts of a property (`-jet_disjuncts interleaved`).
 *
 * The search tree of a disjunct is decomposed into `P = 2^p` subproblems (`-sub`, `-subfactor`): the
 * subproblem `path` of a disjunct is the node reached from its root by the decisions of the bits of
 * `path`, from the most significant one (`0` the left child, `1` the right one). Each disjunct has its
 * own counter `next_path[d]`, the first of its paths not taken yet: a block takes a path with an atomic
 * post-increment, and a leaf met while diving at `remaining_depth` skips the paths of its subtree by
 * raising the counter of its disjunct to `skip_target` (an atomic max), exactly as the single counter
 * of one disjunct did. Since the counters are per disjunct, skipping a subtree of one disjunct never
 * skips a path of another one.
 *
 * The disjunct of a new subproblem is chosen by a ticket, an atomic post-increment shared by all the
 * blocks: the ticket `t` tries the disjuncts `t mod D`, `t + 1 mod D`, ... and takes the first one with a
 * path left. The blocks hence take the disjuncts in turn, and a disjunct whose paths are all taken (or
 * skipped) no longer receives any block. There is no subproblem left when every disjunct is exhausted.
 *
 * Every path of every disjunct is either taken by exactly one block (the post-increments return
 * distinct values) or skipped because it lies in the subtree of a leaf, which is the invariant of the
 * single-disjunct decomposition, for each disjunct separately (`tests/disjunct_schedule_test.cpp`).
 *
 * `Atomics` provides `add(unsigned long long*, v)` (returns the old value), `raise(unsigned long
 * long*, v)` (atomic max) and `load`: `DeviceAtomics` in the kernel, host atomics in the tests. */
namespace jet {
namespace subproblems {

using counter_t = unsigned long long;

#ifdef __CUDACC__
struct DeviceAtomics {
  __device__ INLINE static counter_t add(counter_t* p, counter_t v) { return atomicAdd(p, v); }
  __device__ INLINE static void raise(counter_t* p, counter_t v) { atomicMax(p, v); }
  __device__ INLINE static counter_t load(const counter_t* p) { return *static_cast<const volatile counter_t*>(p); }
};
#endif

/** Takes the next subproblem: its disjunct `d` and its `path`, or returns `false` when every disjunct
 * among the `num_disjuncts` is exhausted (`num_paths` paths each). */
template <class Atomics>
CUDA bool take(counter_t* ticket, counter_t* next_path, int num_disjuncts, counter_t num_paths,
  int& d, counter_t& path)
{
  const counter_t t = Atomics::add(ticket, counter_t{1});
  for(int k = 0; k < num_disjuncts; ++k) {
    const int dd = static_cast<int>((t + static_cast<counter_t>(k)) % static_cast<counter_t>(num_disjuncts));
    /** A cheap check before the increment, so that an exhausted counter is not increased forever. */
    if(Atomics::load(&next_path[dd]) >= num_paths) { continue; }
    const counter_t p = Atomics::add(&next_path[dd], counter_t{1});
    if(p < num_paths) {
      d = dd;
      path = p;
      return true;
    }
  }
  return false;
}

/** The first path after the subtree of the node reached by the decisions of `path` with
 * `remaining_depth` decisions still to take (the leaf met while diving). */
CUDA INLINE counter_t skip_target(counter_t path, int remaining_depth) {
  return ((path >> remaining_depth) + counter_t{1}) << remaining_depth;
}

/** Skips the paths of the subtree of a leaf met while diving (`skip_target`) in the disjunct `d`. */
template <class Atomics>
CUDA void skip(counter_t* next_path, int d, counter_t path, int remaining_depth) {
  Atomics::raise(&next_path[d], skip_target(path, remaining_depth));
}

} // namespace subproblems
} // namespace jet

#endif
