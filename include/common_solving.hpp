// Copyright 2023 Pierre Talbot

#ifndef TURBO_COMMON_SOLVING_HPP
#define TURBO_COMMON_SOLVING_HPP

#include <algorithm>
#include <chrono>
#include <thread>
#include <csignal>

#include "config.hpp"
#include "statistics.hpp"

#include "battery/utility.hpp"
#include "battery/allocator.hpp"
#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"

#include "lala/simplifier.hpp"
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
#include "lala/XCSP3_parser.hpp"

using namespace lala;

void block_signal_ctrlc() {
  sigset_t ctrlc;
  sigemptyset(&ctrlc);
  sigaddset(&ctrlc, SIGINT);
  sigaddset(&ctrlc, SIGTERM);
  if(sigprocmask(SIG_BLOCK, &ctrlc, NULL) != 0) {
    printf("%% ERROR: Unable to deal with CTRL-C. Therefore, we will not be able to print the latest solution before quitting.\n");
    perror(NULL);
    return;
  }
}

bool must_quit() {
  sigset_t pending_sigs;
  sigemptyset(&pending_sigs);
  sigpending(&pending_sigs);
  return sigismember(&pending_sigs, SIGINT) == 1 || sigismember(&pending_sigs, SIGTERM) == 1;
}

/** Check if the timeout of the current execution is exceeded and returns `false` otherwise.
 * It also update the statistics relevant to the solving duration and the exhaustive flag if we reach the timeout.
 */
template <class A, class Timepoint>
bool check_timeout(A& a, const Timepoint& start) {
  auto now = std::chrono::high_resolution_clock::now();
  a.stats.duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
  if(a.config.timeout_ms == 0) {
    return true;
  }
  if(a.stats.duration >= a.config.timeout_ms) {
    if(a.config.verbose_solving) {
      printf("%% Timeout reached.\n");
    }
    a.stats.exhaustive = false;
    return false;
  }
  return true;
}

/** This is a simple wrapper aimed at giving a unique type to the allocator, to use them in AbstractDeps. */
template <class Alloc, size_t n>
struct UniqueAlloc {
  Alloc allocator;
  UniqueAlloc() = default;
  CUDA UniqueAlloc(const Alloc& alloc): allocator(alloc) {}
  UniqueAlloc(const UniqueAlloc& alloc) = default;
  UniqueAlloc& operator=(const UniqueAlloc& alloc) = default;
  CUDA void* allocate(size_t bytes) {
    return allocator.allocate(bytes);
  }
  CUDA void deallocate(void* data) {
    allocator.deallocate(data);
  }
};

template <class Alloc, size_t n>
struct UniqueLightAlloc {
  CUDA void* allocate(size_t bytes) {
    return Alloc{}.allocate(bytes);
  }
  CUDA void deallocate(void* data) {
    Alloc{}.deallocate(data);
  }
};

/** This class is parametrized by a universe of discourse, which is the domain of the variables in the store and various allocators:
 * - BasicAllocator: default allocator, used to allocate abstract domains, the environment, storing intermediate results, etc.
 * - PropAllocator: allocator used for the PC abstract domain, to allocate the propagators.
 * - StoreAllocator: allocator used for the store, to allocate the variables.
 *
 * Normally, you should use the fastest memory for the store, then for the propagators and then for the rest.
 */
template <class Universe,
  class BasicAllocator,
  class PropAllocator,
  class StoreAllocator>
struct AbstractDomains {
  using universe_type = typename Universe::local_type;

  /** Version of the abstract domains with a simple allocator, to represent the best solutions. */
  using LIStore = VStore<universe_type, BasicAllocator>;

  using IStore = VStore<Universe, StoreAllocator>;
  using IPC = PC<IStore, PropAllocator>; // Interval Propagators Completion
  using ISimplifier = Simplifier<IPC, BasicAllocator>;
  using Split = SplitStrategy<IPC, BasicAllocator>;
  using IST = SearchTree<IPC, Split, BasicAllocator>;
  using IBAB = BAB<IST, LIStore>;

  using basic_allocator_type = BasicAllocator;
  using prop_allocator_type = PropAllocator;
  using store_allocator_type = StoreAllocator;

  using this_type = AbstractDomains<Universe, BasicAllocator, PropAllocator, StoreAllocator>;

  struct tag_copy_cons{};

  template <class U2, class BasicAlloc2, class PropAllocator2, class StoreAllocator2>
  CUDA AbstractDomains(const AbstractDomains<U2, BasicAlloc2, PropAllocator2, StoreAllocator2>& other,
    const BasicAllocator& basic_allocator = BasicAllocator(),
    const PropAllocator& prop_allocator = PropAllocator(),
    const StoreAllocator& store_allocator = StoreAllocator(),
    const tag_copy_cons& tag = tag_copy_cons{})
   : basic_allocator(basic_allocator)
   , prop_allocator(prop_allocator)
   , store_allocator(store_allocator)
   , fzn_output(other.fzn_output, basic_allocator)
   , config(other.config, basic_allocator)
   , stats(other.stats)
   , env(other.env, basic_allocator)
   , store(store_allocator)
   , ipc(prop_allocator)
   , simplifier(basic_allocator)
   , split(basic_allocator)
   , eps_split(basic_allocator)
   , search_tree(basic_allocator)
   , best(basic_allocator)
   , bab(basic_allocator)
  {
    AbstractDeps<BasicAllocator, PropAllocator, StoreAllocator> deps{basic_allocator, prop_allocator, store_allocator};
    store = deps.template clone<IStore>(other.store);
    ipc = deps.template clone<IPC>(other.ipc);
    simplifier = battery::allocate_shared<ISimplifier, BasicAllocator>(basic_allocator, *other.simplifier, typename ISimplifier::light_copy_tag{}, ipc, basic_allocator);
    split = deps.template clone<Split>(other.split);
    eps_split = deps.template clone<Split>(other.eps_split);
    search_tree = deps.template clone<IST>(other.search_tree);
    bab = deps.template clone<IBAB>(other.bab);
    best = bab->optimum_ptr();
  }

  CUDA AbstractDomains(const this_type& other,
    const BasicAllocator& basic_allocator = BasicAllocator(),
    const PropAllocator& prop_allocator = PropAllocator(),
    const StoreAllocator& store_allocator = StoreAllocator())
   : this_type(other, basic_allocator, prop_allocator, store_allocator, tag_copy_cons{})
  {}

  template <class Alloc>
  CUDA AbstractDomains(const Configuration<Alloc>& config,
   const BasicAllocator& basic_allocator = BasicAllocator(),
   const PropAllocator& prop_allocator = PropAllocator(),
   const StoreAllocator& store_allocator = StoreAllocator())
  : basic_allocator(basic_allocator)
  , prop_allocator(prop_allocator)
  , store_allocator(store_allocator)
  , config(config, basic_allocator)
  , env(basic_allocator)
  , fzn_output(basic_allocator)
  , store(store_allocator)
  , ipc(prop_allocator)
  , simplifier(basic_allocator)
  , split(basic_allocator)
  , eps_split(basic_allocator)
  , search_tree(basic_allocator)
  , best(basic_allocator)
  , bab(basic_allocator)
  {}

  AbstractDomains(AbstractDomains&& other) = default;

  BasicAllocator basic_allocator;
  PropAllocator prop_allocator;
  StoreAllocator store_allocator;

  abstract_ptr<IStore> store;
  abstract_ptr<IPC> ipc;
  abstract_ptr<ISimplifier> simplifier;
  abstract_ptr<Split> split;
  abstract_ptr<Split> eps_split;
  abstract_ptr<IST> search_tree;
  abstract_ptr<LIStore> best;
  abstract_ptr<IBAB> bab;

  // The environment of variables, storing the mapping between variable's name and their representation in the abstract domains.
  VarEnv<BasicAllocator> env;

  // Information about the output of the solutions expected by MiniZinc.
  FlatZincOutput<BasicAllocator> fzn_output;

  Configuration<BasicAllocator> config;
  Statistics stats;

  CUDA void allocate(int num_vars) {
    env = VarEnv<basic_allocator_type>{basic_allocator};
    store = battery::allocate_shared<IStore, StoreAllocator>(store_allocator, env.extends_abstract_dom(), num_vars, store_allocator);
    ipc = battery::allocate_shared<IPC, PropAllocator>(prop_allocator, env.extends_abstract_dom(), store, prop_allocator);
    // If the simplifier is already allocated, it means we are currently reallocating the abstract domains after preprocessing.
    if(!simplifier) {
      simplifier = battery::allocate_shared<ISimplifier, BasicAllocator>(basic_allocator, env.extends_abstract_dom(), ipc, basic_allocator);
    }
    split = battery::allocate_shared<Split, BasicAllocator>(basic_allocator, env.extends_abstract_dom(), ipc, basic_allocator);
    eps_split = battery::allocate_shared<Split, BasicAllocator>(basic_allocator, env.extends_abstract_dom(), ipc, basic_allocator);
    search_tree = battery::allocate_shared<IST, BasicAllocator>(basic_allocator, env.extends_abstract_dom(), ipc, split, basic_allocator);
    // Note that `best` must have the same abstract type then store (otherwise projection of the variables will fail).
    best = battery::allocate_shared<LIStore, BasicAllocator>(basic_allocator, store->aty(), num_vars, basic_allocator);
    bab = battery::allocate_shared<IBAB, BasicAllocator>(basic_allocator, env.extends_abstract_dom(), search_tree, best);
    if(config.verbose_solving) {
      printf("%% Abstract domain allocated.\n");
    }
  }

  // This force the deallocation of shared memory inside a kernel.
  CUDA void deallocate() {
    store = nullptr;
    ipc = nullptr;
    simplifier = nullptr;
    split = nullptr;
    eps_split = nullptr;
    search_tree = nullptr;
    bab = nullptr;
    env = VarEnv<BasicAllocator>{basic_allocator}; // this is to release the memory used by `VarEnv`.
  }

  // Mainly to interpret the IN constraint in IPC instead of only over-approximating in intervals.
  template <class F>
  CUDA void typing(F& f) const {
    switch(f.index()) {
      case F::Seq:
      if(f.sig() == ::lala::IN && f.seq(1).is(F::S) && f.seq(1).s().size() > 1) {
          f.type_as(ipc->aty());
          return;
        }
        for(int i = 0; i < f.seq().size(); ++i) {
          typing(f.seq(i));
        }
        break;
      case F::ESeq:
        for(int i = 0; i < f.eseq().size(); ++i) {
          typing(f.eseq(i));
        }
        break;
    }
  }

  template <class F>
  CUDA bool interpret(const F& f) {
    if(config.verbose_solving) {
      printf("%% Interpreting the formula...\n");
    }
    auto r = bab->interpret_tell_in(f, env);
    if(!r.has_value()) {
      r.print_diagnostics();
      return false;
    }
    local::BInc has_changed;
    bab->tell(std::move(r.value()), has_changed);
    stats.variables = store->vars();
    stats.constraints = ipc->num_refinements();
    bool can_interpret = true;
    if(split->num_strategies() == 0) {
      can_interpret &= interpret_default_strategy<F>();
    }
    if(eps_split->num_strategies() == 0) {
      can_interpret &= interpret_default_eps_strategy<F>();
    }
    return can_interpret;
  }

  template <class F>
  CUDA void prepare_simplifier(F& f) {
    if(config.verbose_solving) {
      printf("%% Simplifying the formula...\n");
    }
    auto r = simplifier->interpret_tell_in(f, env);
    if(!r.has_value()) {
      if(config.verbose_solving) {
        printf("WARNING: Could not simplify the formula because:\n");
        r.print_diagnostics();
        return;
      }
    }
    simplifier->tell(std::move(r.value()));
  }

  template <class F>
  void type_and_interpret(F& f) {
    typing(f);
    if(config.print_ast) {
      printf("%% Typed AST:\n");
      f.print(true);
      printf("\n");
    }
    if(!interpret(f)) {
      exit(EXIT_FAILURE);
    }

    if(config.print_ast) {
      printf("%% Interpreted AST:\n");
      ipc->deinterpret(env).print(true);
      printf("\n");
    }
    if(config.verbose_solving) {
      printf("%% Formula has been intepreted.\n");
    }
  }

  using FormulaPtr = battery::shared_ptr<TFormula<basic_allocator_type>, basic_allocator_type>;

  FormulaPtr prepare_solver() {
    auto start = std::chrono::high_resolution_clock::now();

    // I. Parse the FlatZinc model.
    FormulaPtr f;
    if(config.input_format() == InputFormat::FLATZINC) {
      f = parse_flatzinc(config.problem_path.data(), fzn_output);
    }
    else if(config.input_format() == InputFormat::XCSP3) {
      f = parse_xcsp3(config.problem_path.data(), fzn_output);
    }
    if(!f) {
      std::cerr << "Could not parse input file." << std::endl;
      exit(EXIT_FAILURE);
    }

    if(config.verbose_solving) {
      printf("%% Input file parsed\n");
    }

    if(config.print_ast) {
      printf("%% Parsed AST:\n");
      f->print(true);
      printf("\n");
    }

    allocate(num_quantified_vars(*f));
    type_and_interpret(*f);

    auto interpretation_time = std::chrono::high_resolution_clock::now();
    stats.interpretation_duration += std::chrono::duration_cast<std::chrono::milliseconds>(interpretation_time - start).count();
    return f;
  }

  void preprocess() {
    auto raw_formula = prepare_solver();
    auto start = std::chrono::high_resolution_clock::now();
    prepare_simplifier(*raw_formula);
    GaussSeidelIteration fp_engine;
    fp_engine.fixpoint(*ipc);
    fp_engine.fixpoint(*simplifier);
    auto f = simplifier->deinterpret();
    stats.eliminated_variables = simplifier->num_eliminated_variables();
    stats.eliminated_formulas = simplifier->num_eliminated_formulas();
    allocate(num_quantified_vars(f));
    type_and_interpret(f);
    auto interpretation_time = std::chrono::high_resolution_clock::now();
    stats.interpretation_duration += std::chrono::duration_cast<std::chrono::milliseconds>(interpretation_time - start).count();
  }

private:
  template <class F>
  CUDA bool interpret_default_strategy() {
    if(config.verbose_solving) {
      printf("%% No split strategy provided, using the default one (first_fail, indomain_split).\n");
    }
    config.free_search = true;
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

  template <class F>
  CUDA bool interpret_default_eps_strategy() {
    typename F::Sequence seq;
    seq.push_back(F::make_nary("first_fail", {}));
    seq.push_back(F::make_nary("indomain_split", {}));
    for(int i = 0; i < env.num_vars(); ++i) {
      seq.push_back(F::make_avar(env[i].avars[0]));
    }
    F search_strat = F::make_nary("search", std::move(seq));
    auto r = eps_split->interpret_tell_in(search_strat, env);
    if(!r.has_value()) {
      r.print_diagnostics();
      return false;
    }
    eps_split->tell(r.value());
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
    stats.depth_max = battery::max(stats.depth_max, search_tree->depth());
  }

  CUDA bool is_printing_intermediate_sol() {
    return bab->is_satisfaction() || config.print_intermediate_solutions;
  }

  CUDA void print_solution() {
    fzn_output.print_solution(env, *best, *simplifier);
    stats.print_mzn_separator();
  }

  CUDA bool update_solution_stats() {
    stats.solutions++;
    if(bab->is_satisfaction() && config.stop_after_n_solutions != 0 &&
       stats.solutions >= config.stop_after_n_solutions)
    {
      stats.exhaustive = false;
      return false;
    }
    return true;
  }

  CUDA bool on_solution_node() {
    if(is_printing_intermediate_sol()) {
      print_solution();
    }
    return update_solution_stats();
  }

  CUDA void on_failed_node() {
    stats.fails += 1;
  }

  CUDA void print_final_solution() {
    if(!is_printing_intermediate_sol() && stats.solutions > 0) {
      print_solution();
    }
    stats.print_mzn_final_separator();
  }

  CUDA void print_mzn_statistics() {
    if(config.print_statistics) {
      config.print_mzn_statistics();
      stats.print_mzn_statistics();
      if(!bab->objective_var().is_untyped() && !best->is_bot()) {
        stats.print_mzn_objective(best->project(bab->objective_var()), bab->is_minimization());
      }
      stats.print_mzn_end_stats();
    }
  }

  /** Extract in `this` the content of `other`. */
  template <class U2, class BasicAlloc2, class PropAlloc2, class StoreAlloc2>
  CUDA void join(AbstractDomains<U2, BasicAlloc2, PropAlloc2, StoreAlloc2>& other) {
    if(bab->is_optimization() && !other.best->is_bot() && bab->compare_bound(*other.best, *best)) {
      other.best->extract(*best);
    }
    stats.join(other.stats);
  }
};

using Itv = Interval<ZInc<int, battery::local_memory>>;
using CP = AbstractDomains<Itv,
  battery::statistics_allocator<battery::standard_allocator>,
  battery::statistics_allocator<UniqueLightAlloc<battery::standard_allocator, 0>>,
  battery::statistics_allocator<UniqueLightAlloc<battery::standard_allocator, 1>>>;

#endif
