// Copyright 2023 Pierre Talbot

#ifndef TURBO_COMMON_SOLVING_HPP
#define TURBO_COMMON_SOLVING_HPP

#ifdef _WINDOWS
#include <atomic>
#endif
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
#include "lala/pir.hpp"
#include "lala/fixpoint.hpp"
#include "lala/search_tree.hpp"
#include "lala/bab.hpp"
#include "lala/split_strategy.hpp"
#include "lala/interpretation.hpp"

#include "lala/flatzinc_parser.hpp"

#ifdef WITH_XCSP3PARSER
  #include "lala/XCSP3_parser.hpp"
#endif

using namespace lala;

#ifdef _WINDOWS
//
// The Microsoft Windows API does not support sigprocmask() or sigpending(),
// so we have to fall back to traditional signal handling.
//
static std::atomic<bool> got_signal;
static void (*prev_sigint)(int);
static void (*prev_sigterm)(int);

void signal_handler(int signum)
{
    signal(SIGINT, signal_handler); // re-arm
    signal(SIGTERM, signal_handler); // re-arm

    got_signal = true; // volatile

    if (signum == SIGINT && prev_sigint != SIG_DFL && prev_sigint != SIG_IGN) {
        (*prev_sigint)(signum);
    }
    if (signum == SIGTERM && prev_sigterm != SIG_DFL && prev_sigterm != SIG_IGN) {
        (*prev_sigterm)(signum);
    }
}

void block_signal_ctrlc() {
    prev_sigint = signal(SIGINT, signal_handler);
    prev_sigterm = signal(SIGTERM, signal_handler);
}

bool must_quit() {
    return static_cast<bool>(got_signal);
}

#else

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
#endif

/** Check if the timeout of the current execution is exceeded and returns `false` otherwise.
 * It also update the statistics relevant to the solving duration and the exhaustive flag if we reach the timeout.
 */
template <class A, class Timepoint>
bool check_timeout(A& a, const Timepoint& start) {
  a.stats.update_timer(Timer::OVERALL, start);
  if(a.config.timeout_ms == 0) {
    return true;
  }
  if(a.stats.time_ms_of(Timer::OVERALL) >= static_cast<int64_t>(a.config.timeout_ms)) {
    if(a.config.verbose_solving) {
      printf("%% CPU: Timeout reached.\n");
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
  UniqueAlloc(UniqueAlloc&& alloc) = default;
  UniqueAlloc& operator=(const UniqueAlloc& alloc) = default;
  UniqueAlloc& operator=(UniqueAlloc&& alloc) = default;
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
  using IPC = PIR<IStore, PropAllocator>; // Interval Propagators Completion
  using ISimplifier = Simplifier<IPC, BasicAllocator>;
  using Split = SplitStrategy<IPC, BasicAllocator>;
  using IST = SearchTree<IPC, Split, BasicAllocator>;
  using IBAB = BAB<IST, LIStore>;

  using basic_allocator_type = BasicAllocator;
  using prop_allocator_type = PropAllocator;
  using store_allocator_type = StoreAllocator;

  using this_type = AbstractDomains<Universe, BasicAllocator, PropAllocator, StoreAllocator>;

  struct tag_copy_cons{};

  struct tag_gpu_block_copy{};

  /** We copy `other` in a new element, and ignore every variable not used in a GPU block.
   * This is because copying everything in each block is very slow.
   *
   * NOTE: It is not the allocation itself that is slow, I think it calling many copy constructors for atomic variables (note that in simplifier we have an atomic memory if the underlying domain has one).
  */
  template <class U2, class BasicAlloc2, class PropAllocator2, class StoreAllocator2>
  CUDA AbstractDomains(const tag_gpu_block_copy&,
    bool enable_sharing, // `true` if the propagators are not in the shared memory.
    const AbstractDomains<U2, BasicAlloc2, PropAllocator2, StoreAllocator2>& other,
    const BasicAllocator& basic_allocator = BasicAllocator(),
    const PropAllocator& prop_allocator = PropAllocator(),
    const StoreAllocator& store_allocator = StoreAllocator())
   : basic_allocator(basic_allocator)
   , prop_allocator(prop_allocator)
   , store_allocator(store_allocator)
   , solver_output(basic_allocator)
   , config(other.config, basic_allocator)
   , stats(other.stats)
   , env(basic_allocator)
   , store(store_allocator)
   , ipc(prop_allocator)
   , simplifier(basic_allocator)
   , split(basic_allocator)
   , eps_split(basic_allocator)
   , search_tree(basic_allocator)
   , best(basic_allocator)
   , bab(basic_allocator)
  {
    AbstractDeps<BasicAllocator, PropAllocator, StoreAllocator> deps{enable_sharing, basic_allocator, prop_allocator, store_allocator};
    store = deps.template clone<IStore>(other.store);
    ipc = deps.template clone<IPC>(other.ipc);
    split = deps.template clone<Split>(other.split);
    eps_split = deps.template clone<Split>(other.eps_split);
    search_tree = deps.template clone<IST>(other.search_tree);
    bab = deps.template clone<IBAB>(other.bab);
    best = bab->optimum_ptr();
  }

  template <class U2, class BasicAlloc2, class PropAllocator2, class StoreAllocator2>
  CUDA AbstractDomains(const AbstractDomains<U2, BasicAlloc2, PropAllocator2, StoreAllocator2>& other,
    const BasicAllocator& basic_allocator = BasicAllocator(),
    const PropAllocator& prop_allocator = PropAllocator(),
    const StoreAllocator& store_allocator = StoreAllocator(),
    const tag_copy_cons& tag = tag_copy_cons{})
   : AbstractDomains(tag_gpu_block_copy{}, false, other, basic_allocator, prop_allocator, store_allocator)
  {
    solver_output = other.solver_output;
    env = other.env;
    simplifier = battery::allocate_shared<ISimplifier, BasicAllocator>(basic_allocator, *other.simplifier, typename ISimplifier::light_copy_tag{}, ipc, basic_allocator);
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
  , solver_output(basic_allocator)
  , store(store_allocator)
  , ipc(prop_allocator)
  , simplifier(basic_allocator)
  , split(basic_allocator)
  , eps_split(basic_allocator)
  , search_tree(basic_allocator)
  , best(basic_allocator)
  , bab(basic_allocator)
  {
    size_t num_subproblems = 1;
    num_subproblems <<= config.subproblems_power;
    stats.eps_num_subproblems = num_subproblems;
  }

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
  SolverOutput<BasicAllocator> solver_output;

  Configuration<BasicAllocator> config;
  Statistics<BasicAllocator> stats;

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

private:
  // We first try to interpret, and if it does not work, we interpret again with the diagnostics mode turned on.
  template <class F, class Env, class A>
  CUDA bool interpret_and_diagnose_and_tell(const F& f, Env& env, A& a) {
    IDiagnostics diagnostics;
    if(!interpret_and_tell(f, env, a, diagnostics)) {
      IDiagnostics diagnostics2;
      interpret_and_tell<true>(f, env, a, diagnostics2);
      diagnostics2.print();
      return false;
    }
    return true;
  }

public:
  template <class F>
  CUDA bool interpret(const F& f) {
    if(config.verbose_solving) {
      printf("%% Interpreting the formula...\n");
    }
    if(!interpret_and_diagnose_and_tell(f, env, *bab)) {
      return false;
    }
    /** If some variables were added during the interpretation, we must resize `best` as well.
     * If we don't do it now, it will be done during the solving (when calling bab.extract) which will lead to a resize of the underlying store.
     * The problem is that the resize will be done on the device! If it was allocated in managed memory, it will be now reallocated in device memory, leading to a segfault later on.
    */
    if(store->vars() != best->vars()) {
      store->extract(*best);
      best->join_top();
    }
    stats.variables = store->vars();
    stats.constraints = ipc->num_deductions();
    bool can_interpret = true;
    /** We add a search strategy by default for the variables that potentially do not occur in the previous strategies. */
    can_interpret &= interpret_default_strategy<F>();
    can_interpret &= interpret_default_eps_strategy<F>();
    return can_interpret;
  }

  template <class F>
  CUDA bool prepare_simplifier(F& f) {
    if(config.verbose_solving) {
      printf("%% Simplifying the formula...\n");
    }
    IDiagnostics diagnostics;
    typename ISimplifier::template tell_type<basic_allocator_type> tell{basic_allocator};
    if(top_level_ginterpret_in<IKind::TELL>(*simplifier, f, env, tell, diagnostics)) {
      simplifier->deduce(std::move(tell));
      return true;
    }
    else if(config.verbose_solving) {
      printf("WARNING: Could not simplify the formula because:\n");
      IDiagnostics diagnostics2;
      top_level_ginterpret_in<IKind::TELL, true>(*simplifier, f, env, tell, diagnostics2);
      diagnostics2.print();
    }
    return false;
  }

  template <class F>
  void type_and_interpret(F& f) {
    if(config.verbose_solving) {
      printf("%% Typing the formula...\n");
    }
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
      printf("%% Formula has been interpreted.\n");
    }
  }

  using FormulaPtr = battery::shared_ptr<TFormula<basic_allocator_type>, basic_allocator_type>;

  FormulaPtr prepare_solver() {
    // I. Parse the FlatZinc model.
    FormulaPtr f;
    if(config.input_format() == InputFormat::FLATZINC) {
      f = parse_flatzinc(config.problem_path.data(), solver_output);
    }
#ifdef WITH_XCSP3PARSER
    else if(config.input_format() == InputFormat::XCSP3) {
      f = parse_xcsp3(config.problem_path.data(), solver_output);
    }
#endif
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
    *f = eval(*f);
    if(config.verbose_solving) {
      printf("%% Formula syntactically simplified.\n");
    }
    stats.print_stat("parsed_variables", num_quantified_vars(*f));
    stats.print_stat("parsed_constraints", num_constraints(*f));
    *f = normalize(ternarize(*f));
    size_t num_vars = num_quantified_vars(*f);
    /** TNF = ternary normal form (apply normalize and ternarize functions). */
    stats.print_stat("tnf_variables", num_vars);
    stats.print_stat("tnf_constraints", num_constraints(*f));
    allocate(num_vars);
    type_and_interpret(*f);
    return f;
  }

  struct vstat {
    size_t num_occurrences;
    bool infinite_domain;
    size_t domain_size;
    vstat() = default;
  };

  /** The constraint network must be normalized. */
  void analyze_pir() const {
    if(ipc->is_bot()) {
      return;
    }
    if(config.verbose_solving) {
      printf("%% Analyzing the constraint network...\n");
    }
    std::vector<vstat> vstats(store->vars(), vstat{});
    for(int i = 0; i < store->vars(); ++i) {
      auto width = (*store)[i].width().lb();
      vstats[i].infinite_domain = width.is_top();
      if(!vstats[i].infinite_domain) {
        vstats[i].domain_size = width.value() + 1;
      }
    }

    for(int i = 0; i < ipc->num_deductions(); ++i) {
      bytecode_type bytecode = ipc->load_deduce(i);
      vstats[bytecode.x.vid()].num_occurrences++;
      vstats[bytecode.y.vid()].num_occurrences++;
      vstats[bytecode.z.vid()].num_occurrences++;
    }

    size_t num_constants = 0;
    size_t num_2bits_vars = 0;
    size_t num_64bits_vars = 0;
    size_t num_128bits_vars = 0;
    size_t num_256bits_vars = 0;
    size_t num_512bits_vars = 0;
    size_t num_65536bits_vars = 0;
    size_t num_infinites = 0;
    double average_occ_vars = 0;
    size_t sum_props_of_vars = 0;
    size_t max_occ_vars = 0;
    size_t num_vars_in_2_constraints = 0;
    size_t num_vars_in_3_constraints = 0;
    size_t num_vars_in_4_to_10_constraints = 0;
    size_t num_vars_in_more_than_10_constraints = 0;
    size_t largest_dom = 0;
    size_t sum_domain_size = 0;
    for(int i = 0; i < vstats.size(); ++i) {
      if(vstats[i].infinite_domain) {
        ++num_infinites;
      }
      else {
        largest_dom = battery::max(largest_dom, vstats[i].domain_size);
        sum_domain_size += vstats[i].domain_size;
      }
      if(vstats[i].domain_size == 1) {
        ++num_constants;
      }
      else {
        num_2bits_vars += vstats[i].domain_size == 2;
        num_64bits_vars += vstats[i].domain_size <= 64;
        num_128bits_vars += vstats[i].domain_size <= 128;
        num_256bits_vars += vstats[i].domain_size <= 256;
        num_512bits_vars += vstats[i].domain_size <= 512;
        num_65536bits_vars += vstats[i].domain_size <= 65536;
        sum_props_of_vars += vstats[i].num_occurrences;
        max_occ_vars = battery::max(max_occ_vars, vstats[i].num_occurrences);
      }
      num_vars_in_2_constraints += vstats[i].num_occurrences == 2;
      num_vars_in_3_constraints += vstats[i].num_occurrences == 3;
      num_vars_in_4_to_10_constraints += vstats[i].num_occurrences > 3 && vstats[i].num_occurrences <= 10;
      num_vars_in_more_than_10_constraints += vstats[i].num_occurrences > 10;
    }
    average_occ_vars =  static_cast<double>(sum_props_of_vars) / static_cast<double>(vstats.size() - num_constants);

    stats.print_stat("num_constants", num_constants);
    stats.print_stat("num_infinite_domains", num_infinites);
    // Print the number of average occurrence of variables in the constraints.
    stats.print_stat("sum_props_of_vars", sum_props_of_vars);
    print_memory_statistics("sum_props_of_vars", sum_props_of_vars*4); // 1 integer per propagator indexes.
    stats.print_stat("avg_constraints_per_unassigned_var", average_occ_vars);
    // Print the maximum number of occurrence of a single variable in the constraints.
    stats.print_stat("max_constraints_per_unassigned_var", max_occ_vars);
    // Print the number of variables occurring only twice in the constraints.
    stats.print_stat("num_vars_in_2_constraints", num_vars_in_2_constraints);
    stats.print_stat("num_vars_in_3_constraints", num_vars_in_3_constraints);
    stats.print_stat("num_vars_in_4_to_10_constraints", num_vars_in_4_to_10_constraints);
    stats.print_stat("num_vars_in_more_than_10_constraints", num_vars_in_more_than_10_constraints);
    // Print the number of bits required to represent the bounded variables in a bitset.
    stats.print_stat("sum_domain_size", sum_domain_size);
    print_memory_statistics("sum_domain_size", sum_domain_size/8);
    // Print the largest variable in the constraint network.
    stats.print_stat("largest_domain", largest_dom);
    // Print the number of variables with a width of 2, 64, 128, ... bits or less in the constraint network.
    stats.print_stat("num_2bits_vars", num_2bits_vars);
    stats.print_stat("num_64bits_vars", num_64bits_vars);
    stats.print_stat("num_128bits_vars", num_128bits_vars);
    stats.print_stat("num_256bits_vars", num_256bits_vars);
    stats.print_stat("num_512bits_vars", num_512bits_vars);
    stats.print_stat("num_65536bits_vars", num_65536bits_vars);
    // For each operator, print how many times they each occur in the constraint network.
  }

  void preprocess() {
    auto start = stats.start_timer_host();
    auto raw_formula = prepare_solver();
    if(prepare_simplifier(*raw_formula)) {
      GaussSeidelIteration fp_engine;
      fp_engine.fixpoint(ipc->num_deductions(), [&](size_t i) { return ipc->deduce(i); });
      if(config.simplify) {
        fp_engine.fixpoint(simplifier->num_deductions(), [&](size_t i) { return simplifier->deduce(i); });
        auto f = simplifier->deinterpret();
        stats.eliminated_variables = simplifier->num_eliminated_variables();
        stats.eliminated_formulas = simplifier->num_eliminated_formulas();
        if(config.verbose_solving) {
          printf("%% Formula simplified.\n");
          printf("%% Ternarizing the formula...\n");
        }
        using F = TFormula<basic_allocator_type>;
        f = ternarize(f);
        if(config.verbose_solving) {
          printf("%% Formula ternarized.\n");
          if(config.print_ast) {
            f.print(false);
          }
          printf("%% Normalizing the formula...\n");
        }
        f = normalize(f);
        if(config.verbose_solving) {
          printf("%% Formula normalized.\n");
          if(config.print_ast) {
            f.print(false);
          }
        }
        size_t num_vars = num_quantified_vars(f);
        stats.print_stat("simplified_tnf_vars", num_vars);
        stats.print_stat("simplified_tnf_constraints", num_constraints(f));
        allocate(num_vars);
        type_and_interpret(f);
      }
    }
    if(config.network_analysis) {
      analyze_pir();
    }
    stats.stop_timer(Timer::PREPROCESSING, start);
    stats.print_timing_stat("preprocessing_time", Timer::PREPROCESSING);
  }

private:
  template <class F>
  CUDA bool interpret_default_strategy() {
    config.free_search = true;
    typename F::Sequence seq;
    seq.push_back(F::make_nary("first_fail", {}));
    seq.push_back(F::make_nary("indomain_min", {}));
    for(int i = 0; i < env.num_vars(); ++i) {
      seq.push_back(F::make_avar(env[i].avars[0]));
    }
    F search_strat = F::make_nary("search", std::move(seq));
    if(!interpret_and_diagnose_and_tell(search_strat, env, *bab)) {
      return false;
    }
    return true;
  }

  template <class F>
  CUDA bool interpret_default_eps_strategy() {
    typename F::Sequence seq;
    seq.push_back(F::make_nary("first_fail", {}));
    seq.push_back(F::make_nary("indomain_min", {}));
    for(int i = 0; i < env.num_vars(); ++i) {
      seq.push_back(F::make_avar(env[i].avars[0]));
    }
    F search_strat = F::make_nary("search", std::move(seq));
    if(!interpret_and_diagnose_and_tell(search_strat, env, *eps_split)) {
      return false;
    }
    return true;
  }

public:
  CUDA bool on_node() {
    stats.nodes++;
    stats.depth_max = battery::max(stats.depth_max, search_tree->depth());
    if(stats.nodes >= config.stop_after_n_nodes) {
      prune();
      return true;
    }
    return false;
  }

  CUDA bool is_printing_intermediate_sol() {
    return bab->is_satisfaction() || config.print_intermediate_solutions;
  }

  CUDA void print_solution() {
    solver_output.print_solution(env, *best, *simplifier);
    stats.print_mzn_separator();
  }

  CUDA void prune() {
    stats.exhaustive = false;
  }

  /** Return `true` if the search state must be pruned. */
  CUDA bool update_solution_stats() {
    stats.solutions++;
    if(bab->is_satisfaction() && config.stop_after_n_solutions != 0 &&
       stats.solutions >= config.stop_after_n_solutions)
    {
      prune();
      return true;
    }
    return false;
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
      stats.print_mzn_statistics(config.or_nodes);
      if(!bab->objective_var().is_untyped() && !best->is_top()) {
        stats.print_mzn_objective(best->project(bab->objective_var()), bab->is_minimization());
      }
      stats.print_mzn_end_stats();
    }
  }

  /** Extract in `this` the content of `other`. */
  template <class U2, class BasicAlloc2, class PropAlloc2, class StoreAlloc2>
  CUDA void meet(AbstractDomains<U2, BasicAlloc2, PropAlloc2, StoreAlloc2>& other) {
    if(bab->is_optimization() && !other.best->is_top() && bab->compare_bound(*other.best, *best)) {
      other.best->extract(*best);
    }
    stats.meet(other.stats);
  }
};

using Itv = Interval<local::ZLB>;

template <class Universe, class Allocator = battery::standard_allocator>
using CP = AbstractDomains<Universe,
  battery::statistics_allocator<Allocator>,
  battery::statistics_allocator<UniqueLightAlloc<Allocator, 0>>,
  battery::statistics_allocator<UniqueLightAlloc<Allocator, 1>>>;

#endif
