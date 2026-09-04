// Copyright 2025 Pierre Talbot

#ifndef TURBO_LIGHT_BRANCH_HPP
#define TURBO_LIGHT_BRANCH_HPP

#include "lala/logic/ast.hpp"

/** A binary branching decision over a universe (e.g. interval).
 * This is the branching representation used by the barebones architecture: a decision is a variable
 * and the two universes to be embedded in the store, without any intermediate logical formula.
 */
template <class U>
struct LightBranch {
  template <class U2>
  friend class LightBranch;

  lala::AVar var;
  U children[2];
  /** Ropes are used for fast backtracking: `ropes[1]` is the depth we need to backtrack to if the right node is a leaf. */
  int ropes[2];
  int current_idx;

  CUDA INLINE LightBranch(): current_idx(-1) {}
  LightBranch(const LightBranch&) = default;
  LightBranch(LightBranch&&) = default;
  CUDA INLINE LightBranch(lala::AVar var, const U& left, const U& right)
   : var(var), current_idx(-1)
  {
    children[0] = left;
    children[1] = right;
  }

  CUDA INLINE const U& next() {
    assert(has_next());
    return children[++current_idx];
  }

  CUDA INLINE const U& operator[](int idx) {
    return children[idx];
  }

  CUDA INLINE bool has_next() const {
    return current_idx < 1;
  }

  CUDA INLINE void prune() {
    current_idx = 2;
  }

  CUDA INLINE bool is_pruned() const {
    return current_idx >= 2;
  }

  CUDA INLINE const U& current() const {
    assert(current_idx != -1 && current_idx < 2);
    return children[current_idx];
  }
};

#endif
