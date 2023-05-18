// Copyright 2023 Pierre Talbot

#ifndef COMMON_SOLVING_HPP
#define COMMON_SOLVING_HPP

#include <algorithm>
#include <chrono>
#include <thread>

#include "config.hpp"
#include "statistics.hpp"

#include "battery/allocator.hpp"
#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"

#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "lala/pc.hpp"
#include "lala/terms.hpp"
#include "lala/fixpoint.hpp"
#include "lala/search_tree.hpp"
#include "lala/bab.hpp"
#include "lala/split_strategy.hpp"

#include "lala/flatzinc_parser.hpp"

using namespace lala;
using namespace battery;

template <class A, class Timepoint>
bool check_timeout(A& a, const Timepoint& start) {
  auto now = std::chrono::high_resolution_clock::now();
  a.stats.duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
  if(a.config.timeout_ms == 0) {
    return true;
  }
  if(a.stats.duration >= a.config.timeout_ms) {
    a.stats.exhaustive = false;
    return false;
  }
  return true;
}

template <class Universe, class Allocator, class FastAllocator = Allocator>
struct AbstractDomains {
  /** Version of the abstract domains with a simple allocator, to represent the best solutions. */
  using LIStore = VStore<typename Universe::local_type, Allocator>;
  using LIPC = PC<LIStore, Allocator>; // Interval Propagators Completion
  using LIST = SearchTree<LIPC>;

  using IStore = VStore<Universe, FastAllocator>;
  using IPC = PC<IStore, Allocator>; // Interval Propagators Completion
  using Split = SplitStrategy<IPC>;
  using IST = SearchTree<IPC>;
  using IBAB = BAB<IST, LIST>;

  template <class U2, class Alloc2, class FastAllocator2>
  CUDA AbstractDomains(const AbstractDomains<U2, Alloc2, FastAllocator2>& other,
    const Allocator& allocator = Allocator(),
    const FastAllocator fast_allocator = FastAllocator())
   : fzn_output(other.fzn_output)
   , config(other.config)
   , stats(other.stats)
   , env(other.env)
  {
    AbstractDeps<Allocator, FastAllocator> deps(allocator, fast_allocator);
    store = deps.template clone<IStore>(other.store);
    ipc = deps.template clone<IPC>(other.ipc);
    split = deps.template clone<Split>(other.split);
    search_tree = deps.template clone<IST>(other.search_tree);
    bab = deps.template clone<IBAB>(other.bab);
  }

  template <class Alloc>
  CUDA AbstractDomains(const Configuration<Alloc>& config):
    config(config) {}

  AbstractDomains(AbstractDomains&& other) = default;

  shared_ptr<IStore, Allocator> store;
  shared_ptr<IPC, Allocator> ipc;
  shared_ptr<Split, Allocator> split;
  shared_ptr<IST, Allocator> search_tree;
  shared_ptr<IBAB, Allocator> bab;

  // The environment of variables, storing the mapping between variable's name and their representation in the abstract domains.
  VarEnv<Allocator> env;

  // Information about the output of the solutions expected by MiniZinc.
  FlatZincOutput<Allocator> fzn_output;

  Configuration<Allocator> config;
  Statistics stats;

  CUDA void allocate(int num_vars) {
    store = make_shared<IStore, Allocator>(env.extends_abstract_dom(), num_vars);
    ipc = make_shared<IPC, Allocator>(env.extends_abstract_dom(), store);
    split = make_shared<Split, Allocator>(env.extends_abstract_dom(), ipc);
    search_tree = make_shared<IST, Allocator>(env.extends_abstract_dom(), ipc, split);
    bab = make_shared<IBAB, Allocator>(env.extends_abstract_dom(), search_tree);
    if(config.verbose_solving) {
      printf("%%Abstract domain allocated.\n");
    }
  }

  CUDA void deallocate() {
    store = nullptr;
    ipc = nullptr;
    split = nullptr;
    search_tree = nullptr;
    bab = nullptr;
    env = VarEnv<standard_allocator>{}; // this is to release the memory used by `VarEnv`.
  }

  template <class F>
  CUDA bool interpret(const F& f) {
    auto r = bab->interpret_tell_in(f, env);
    if(!r.has_value()) {
      r.print_diagnostics();
      return false;
    }
    local::BInc has_changed;
    bab->tell(std::move(r.value()), has_changed);
    stats.variables = store->vars();
    stats.constraints = ipc->num_refinements();
    if(split->num_strategies() == 0) {
      return interpret_default_strategy<F>();
    }
    return true;
  }

private:
  template <class F>
  CUDA bool interpret_default_strategy() {
    if(config.verbose_solving) {
      printf("%%No split strategy provided, using the default one (first_fail, indomain_split).\n");
    }
    typename F::Sequence seq;
    seq.push_back(F::make_nary("first_fail", {}));
    seq.push_back(F::make_nary("indomain_split", {}));
    for(int i = 0; i < env.num_vars(); ++i) {
      seq.push_back(F::make_avar(env[i].avars[0]));
    }
    F search_strat = F::make_nary("search", std::move(seq));
    auto r = bab->interpret_tell_in(search_strat, env);
    if(!r.has_value()) {
      r.print_diagnostics();
      return false;
    }
    local::BInc has_changed;
    bab->tell(std::move(r.value()), has_changed);
    return true;
  }

public:

  CUDA void print_store() const {
    for(int i = 0; i < store->vars(); ++i) {
      (*store)[i].print();
      printf("%s", (i+1 == store->vars() ? "\n" : ", "));
    }
  }

  CUDA void on_node() {
    stats.nodes++;
  }

  CUDA bool on_solution_node() {
    fzn_output.print_solution(env, bab->optimum());
    stats.print_mzn_separator();
    stats.solutions++;
    stats.depth_max = battery::max(stats.depth_max, search_tree->depth());
    if(config.stop_after_n_solutions != 0 &&
       stats.solutions >= config.stop_after_n_solutions)
    {
      stats.exhaustive = false;
      return false;
    }
    return true;
  }

  CUDA void on_failed_node() {
    stats.fails += 1;
    stats.depth_max = battery::max(stats.depth_max, search_tree->depth());
  }

  CUDA void on_finish() {
    stats.print_mzn_final_separator();
    if(config.print_statistics) {
      stats.print_mzn_statistics();
      stats.print_mzn_objective(*bab);
    }
  }

  template <class U2, class Alloc2, class FastAlloc2>
  CUDA void join(AbstractDomains<U2, Alloc2, FastAlloc2>& other) {
    other.bab->extract(*bab);
    stats.join(other.stats);
  }
};

#endif
