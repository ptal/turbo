// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_UASS_HPP
#define TURBO_UASS_HPP

#include <cstdio>
#include "battery/utility.hpp"
#include "lala/logic/logic.hpp"

/** The underapproximation of the search of `-arch jet` and `-arch fbarebones` (UASS).
 *
 * A decision underapproximates when its children do not cover the domain of its variable: they are
 * points of it. It is the case of `LB_SPLIT`, `UB_SPLIT`, `MID_SPLIT`, `MID_LB_SPLIT` and
 * `MID_UB_SPLIT` on a domain of width at most `epsilon`, of `SPLIT` on a box of width at most
 * `epsilon` (all the variables at their middle), of `MIX_SPLIT`, and of the leaves of the ReLU split
 * (`relu_split.hpp`). A node is underapproximated when a decision of its path is (and only then):
 * its failure refutes the points it tries, not its box, so it is counted in `unknowns` instead of
 * proving the box empty (`propagate`).
 *
 * `Block::uass_depth` is the depth of the first decision of the current path that underapproximates,
 * -1 if there is none; the node is underapproximated when `uass_depth >= 0` (`is_uass`). It is a
 * property of the path, so it follows the backtracking:
 *   - `mark`: a decision that underapproximates is pushed at `b.depth` (before `++b.depth`);
 *   - `backtrack`: the search goes to the right child of the decision `b.depth - 1` (`b.depth` is the
 *     rope), whose path has the decisions `0..b.depth-1`: if the first underapproximation is not one
 *     of them, the node is not underapproximated any more;
 *   - `reset`: at the root of a subproblem (and when a block is allocated).
 * The dive to a subproblem pushes its decisions at depth 0 without recording them: one that
 * underapproximates gives `uass_depth = 0`, which no backtracking of the subproblem resets (they go to
 * a depth of at least 1), since every node of the subproblem is below it.
 *
 * A single flag set by the last decision would be wrong after a backtracking: the right child of an
 * ordinary decision would still be counted unknown, and the right child of `MID_LB_SPLIT` would not
 * be if a decision had reset the flag in between.
 */
namespace uass {

template <class Block>
CUDA INLINE bool is_uass(const Block& b) { return b.uass_depth >= 0; }

template <class Block>
CUDA INLINE void reset(Block& b) { b.uass_depth = -1; }

template <class Block>
CUDA INLINE void mark(Block& b) {
  if(b.uass_depth < 0) { b.uass_depth = b.depth; }
}

template <class Block>
CUDA INLINE void backtrack(Block& b) {
  if(b.uass_depth >= b.depth) { b.uass_depth = -1; }
}

/** The ropes of the decision at `b.depth` (see `push_decision` of the kernels), then `++b.depth`.
 * `Block` is the `BlockData` of the kernel (or a host replay with the same members). */
template <class Block>
CUDA INLINE void push_ropes(Block& b, bool single_child) {
  auto& d = b.decisions;
  d[b.depth].ropes[0] = b.depth + 1;
  d[b.depth].ropes[1] = b.depth > 0 ? d[b.depth-1].ropes[d[b.depth-1].current_idx] : -1;
  if(single_child) {
    d[b.depth].ropes[0] = d[b.depth].ropes[1];
  }
  ++b.depth;
  if(d.size() == b.depth) {
    printf("resize to %d\n", (int)d.size() * 2);
    d.resize(d.size() * 2);
  }
}

/** A decision that underapproximates with the single child `var in child`. Thread 0 only. */
template <class Block, class Itv>
CUDA INLINE void push_point_decision(Block& b, lala::AVar var, const Itv& child) {
  auto& d = b.decisions[b.depth];
  d.var = var;
  d.current_idx = -1;
  d.children[0] = child;
  mark(b);
  push_ropes(b, true);
}

/** A decision that underapproximates with the children `var in first`, then `var in second`.
 * Thread 0 only. */
template <class Block, class Itv>
CUDA INLINE void push_points_decision(Block& b, lala::AVar var, const Itv& first, const Itv& second) {
  auto& d = b.decisions[b.depth];
  d.var = var;
  d.current_idx = -1;
  d.children[0] = first;
  d.children[1] = second;
  mark(b);
  push_ropes(b, false);
}

} // namespace uass

#endif
