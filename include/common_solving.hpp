// Copyright 2023 Pierre Talbot

#ifndef COMMON_SOLVING_HPP
#define COMMON_SOLVING_HPP

#include <algorithm>
#include <chrono>
#include <thread>

#include "config.hpp"
#include "statistics.hpp"

#include "allocator.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"

#include "vstore.hpp"
#include "cartesian_product.hpp"
#include "interval.hpp"
#include "pc.hpp"
#include "terms.hpp"
#include "fixpoint.hpp"
#include "search_tree.hpp"
#include "bab.hpp"

#include "value_order.hpp"
#include "variable_order.hpp"
#include "split.hpp"

#include "flatzinc_parser.hpp"

using namespace lala;
using namespace battery;

template <class Universe, class Allocator, class ManagedAlloc = Allocator>
struct AbstractDomains {
  const int sty = 0;
  const int pty = 1;
  const int tty = 2;
  const int split_ty = 3;
  const int bab_ty = 4;

  using IStore = VStore<Universe, Allocator>;
  using IPC = PC<IStore>; // Interval Propagators Completion
  using ISplitInputLB = Split<IPC, InputOrder<IPC>, LowerBound<IPC>>;
  using IST = SearchTree<IPC, ISplitInputLB>;
  using IBAB = BAB<IST>;

  template <class Alloc>
  AbstractDomains(const Configuration<Alloc>& config):
    config(config) {}

  AbstractDomains(AbstractDomains&& other) = default;

  shared_ptr<IStore, Allocator> store;
  shared_ptr<IPC, Allocator> ipc;
  shared_ptr<ISplitInputLB, Allocator> split;
  shared_ptr<IST, Allocator> search_tree;
  shared_ptr<IBAB, Allocator> bab;

  // The environment of variables, storing the mapping between variable's name and their representation in the abstract domains.
  VarEnv<Allocator> env;

  // Information about the output of the solutions expected by MiniZinc.
  FlatZincOutput<ManagedAlloc> fzn_output;

  Configuration<ManagedAlloc> config;
  Statistics stats;
};

#endif
