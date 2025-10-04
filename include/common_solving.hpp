// Copyright 2023 Pierre Talbot

#ifndef TURBO_COMMON_SOLVING_HPP
#define TURBO_COMMON_SOLVING_HPP

#include <atomic>
#include <algorithm>
#include <chrono>
#include <thread>
#include <csignal>
#include <random>

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

#ifndef TURBO_ITV_BITS
  #define TURBO_ITV_BITS 32
#endif

#if (TURBO_ITV_BITS == 64)
  using bound_value_type = long long int;
#elif (TURBO_ITV_BITS == 16)
  using bound_value_type = short int;
#elif (TURBO_ITV_BITS == 32)
  using bound_value_type = int;
#else
  #error "Invalid value for TURBO_ITV_BITS: must be 16, 32 or 64."
#endif
using Itv = Interval<ZLB<bound_value_type, battery::local_memory>>;

static std::atomic<bool> got_signal;
static void (*prev_sigint)(int);
static void (*prev_sigterm)(int);

void signal_handler(int signum)
{
  std::signal(SIGINT, signal_handler); // re-arm
  std::signal(SIGTERM, signal_handler); // re-arm
  got_signal = true; // volatile
  if (signum == SIGINT && prev_sigint != SIG_DFL && prev_sigint != SIG_IGN) {
    (*prev_sigint)(signum);
  }
  if (signum == SIGTERM && prev_sigterm != SIG_DFL && prev_sigterm != SIG_IGN) {
    (*prev_sigterm)(signum);
  }
}

void block_signal_ctrlc() {
  prev_sigint = std::signal(SIGINT, signal_handler);
  prev_sigterm = std::signal(SIGTERM, signal_handler);
}

template <class A>
bool must_quit(A& a) {
  if(static_cast<bool>(got_signal)) {
    a.prune();
    return true;
  }
  return false;
}

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
    a.prune();
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
#ifdef TURBO_IPC_ABSTRACT_DOMAIN
  using IProp = PC<IStore, PropAllocator>; // Interval Propagators using general propagator completion.
#else
  using IProp = PIR<IStore, PropAllocator>; // Interval Propagators using the TNF representation of propagators.
#endif
  using ISimplifier = Simplifier<IProp, BasicAllocator>;
  using Split = SplitStrategy<IProp, BasicAllocator>;
  using IST = SearchTree<IProp, Split, BasicAllocator>;
  using IBAB = BAB<IST, LIStore>;

  using basic_allocator_type = BasicAllocator;
  using prop_allocator_type = PropAllocator;
  using store_allocator_type = StoreAllocator;

  using this_type = AbstractDomains<Universe, BasicAllocator, PropAllocator, StoreAllocator>;

  using F = TFormula<basic_allocator_type>;

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
   , minimize_obj_var(other.minimize_obj_var)
   , store(store_allocator)
   , iprop(prop_allocator)
   , simplifier(basic_allocator)
   , split(basic_allocator)
   , search_tree(basic_allocator)
   , best(basic_allocator)
   , bab(basic_allocator)
  {
    AbstractDeps<BasicAllocator, PropAllocator, StoreAllocator> deps{enable_sharing, basic_allocator, prop_allocator, store_allocator};
    store = deps.template clone<IStore>(other.store);
    iprop = deps.template clone<IProp>(other.iprop);
    split = deps.template clone<Split>(other.split);
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
    simplifier = battery::allocate_shared<ISimplifier, BasicAllocator>(basic_allocator, *other.simplifier, typename ISimplifier::light_copy_tag{}, iprop, basic_allocator);
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
  , stats(0,0,false,config.print_statistics)
  , env(basic_allocator)
  , solver_output(basic_allocator)
  , store(store_allocator)
  , iprop(prop_allocator)
  , simplifier(basic_allocator)
  , split(basic_allocator)
  , search_tree(basic_allocator)
  , best(basic_allocator)
  , bab(basic_allocator)
  {
    if(config.subproblems_power != -1) {
      size_t num_subproblems = 1;
      num_subproblems <<= config.subproblems_power;
      stats.eps_num_subproblems = num_subproblems;
    }
  }

  AbstractDomains(AbstractDomains&& other) = default;

  BasicAllocator basic_allocator;
  PropAllocator prop_allocator;
  StoreAllocator store_allocator;

  abstract_ptr<IStore> store;
  abstract_ptr<IProp> iprop;
  abstract_ptr<ISimplifier> simplifier;
  abstract_ptr<Split> split;
  abstract_ptr<IST> search_tree;
  abstract_ptr<LIStore> best;
  abstract_ptr<IBAB> bab;

  // The environment of variables, storing the mapping between variable's name and their representation in the abstract domains.
  VarEnv<BasicAllocator> env;

  // Information about the output of the solutions expected by MiniZinc.
  SolverOutput<BasicAllocator> solver_output;

  // The barebones architecture only supports minimization.
  // In case of maximization, we create a new objective variable that is the negation of the original one.
  AVar minimize_obj_var;

  Configuration<BasicAllocator> config;
  Statistics<BasicAllocator> stats;

  CUDA void allocate(int num_vars, bool with_simplifier) {
    env = VarEnv<basic_allocator_type>{basic_allocator};
    store = battery::allocate_shared<IStore, StoreAllocator>(store_allocator, env.extends_abstract_dom(), num_vars, store_allocator);
    iprop = battery::allocate_shared<IProp, PropAllocator>(prop_allocator, env.extends_abstract_dom(), store, prop_allocator);
    if(with_simplifier) {
      simplifier = battery::allocate_shared<ISimplifier, BasicAllocator>(basic_allocator, env.extends_abstract_dom(), store->aty(), iprop, basic_allocator);
    }
    split = battery::allocate_shared<Split, BasicAllocator>(basic_allocator, env.extends_abstract_dom(), store->aty(), iprop, basic_allocator);
    search_tree = battery::allocate_shared<IST, BasicAllocator>(basic_allocator, env.extends_abstract_dom(), iprop, split, basic_allocator);
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
    iprop = nullptr;
    simplifier = nullptr;
    split = nullptr;
    search_tree = nullptr;
    bab = nullptr;
    env = VarEnv<BasicAllocator>{basic_allocator}; // this is to release the memory used by `VarEnv`.
  }

private:
  // Mainly to interpret the IN constraint in IProp instead of only over-approximating in intervals.
  template <class F>
  CUDA void typing(F& f, bool toplevel = true) const {
    if(toplevel && config.verbose_solving) {
      printf("%% Typing the formula...\n");
    }
    switch(f.index()) {
      case F::Seq:
        if(f.sig() == ::lala::IN && f.seq(1).is(F::S) && f.seq(1).s().size() > 1) {
          f.type_as(iprop->aty());
          return;
        }
        for(int i = 0; i < f.seq().size(); ++i) {
          typing(f.seq(i), false);
        }
        break;
      case F::ESeq:
        for(int i = 0; i < f.eseq().size(); ++i) {
          typing(f.eseq(i), false);
        }
        break;
    }
    if(toplevel && config.print_ast) {
      printf("%% Typed AST:\n");
      f.print(true);
      printf("\n");
    }
  }

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
    if(config.print_ast) {
      printf("%% Interpreted AST:\n");
      iprop->deinterpret(env).print();
      printf("\n");
    }
    if(config.verbose_solving) {
      printf("%% Formula has been interpreted.\n");
    }
    /** If some variables were added during the interpretation, we must resize `best` as well.
     * If we don't do it now, it will be done during the solving (when calling bab.extract) which will lead to a resize of the underlying store.
     * The problem is that the resize will be done on the device! If it was allocated in managed memory, it will be now reallocated in device memory, leading to a segfault later on.
    */
    if(store->vars() != best->vars()) {
      store->extract(*best);
      best->join_top();
    }
    if(config.arch == Arch::BAREBONES) {
      if(bab->is_minimization()) {
        minimize_obj_var = bab->objective_var();
      }
      else if(bab->is_maximization()) {
        auto minobj = env.variable_of("__MINIMIZE_OBJ");
        assert(minobj.has_value());
        assert(minobj->get().avar_of(store->aty()).has_value());
        minimize_obj_var = minobj->get().avar_of(store->aty()).value();
      }
    }
    stats.variables = store->vars();
    stats.constraints = iprop->num_deductions();
    bool can_interpret = true;
    /** We add a search strategy by default for the variables that potentially do not occur in the previous strategies.
     * Not necessary with barebones architecture: it is taken into account by the algorithm.
     */
    can_interpret &= interpret_default_strategy<F>();
    return can_interpret;
  }

  using FormulaPtr = battery::shared_ptr<TFormula<basic_allocator_type>, basic_allocator_type>;

  /** Parse a constraint network in the FlatZinc or XCSP3 format.
   * The parsed formula is then syntactically simplified (`eval` function).
  */
  FormulaPtr parse_cn() {
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
      f->print();
      printf("\n");
    }
    stats.print_stat("parsed_variables", num_quantified_vars(*f));
    stats.print_stat("parsed_constraints", num_constraints(*f));
    *f = eval(*f);
    if(config.verbose_solving) {
      printf("%% Formula syntactically simplified.\n");
    }
    return f;
  }

  template <class F>
  void initialize_simplifier(const F& f) {
    IDiagnostics diagnostics;
    typename ISimplifier::template tell_type<basic_allocator_type> tell{basic_allocator};
    if(!top_level_ginterpret_in<IKind::TELL>(*simplifier, f, env, tell, diagnostics)) {
      printf("%% ERROR: Could not simplify the formula because:\n");
      IDiagnostics diagnostics2;
      top_level_ginterpret_in<IKind::TELL, true>(*simplifier, f, env, tell, diagnostics2);
      diagnostics2.print();
      exit(EXIT_FAILURE);
    }
    simplifier->deduce(std::move(tell));
  }

  void preprocess_ipc(F& f) {
    size_t num_vars = num_quantified_vars(f);
    allocate(num_vars, true);
    typing(f);
    if(!interpret(f)) {
      exit(EXIT_FAILURE);
    }
    GaussSeidelIteration fp_engine;
    fp_engine.fixpoint(iprop->num_deductions(), [&](size_t i) { return iprop->deduce(i); });
    /* We need to initialize the simplifier even if we don't simplify.
       This is because the simplifier equivalence classes is used in SolverOutput. */
    initialize_simplifier(f);
    if(config.disable_simplify) {
      return;
    }
    if(config.verbose_solving) {
      printf("%% Simplifying the formula...\n");
    }
    fp_engine.fixpoint(simplifier->num_deductions(), [&](size_t i) { return simplifier->deduce(i); });
    f = simplifier->deinterpret();
    if(config.verbose_solving) {
      printf("%% Formula simplified.\n");
    }
    f = normalize(f);
    num_vars = num_quantified_vars(f);
    stats.print_stat("variables_after_simplification", num_vars);
    stats.print_stat("constraints_after_simplification", num_constraints(f));
    allocate(num_vars, false);
    typing(f);
    if(!interpret(f)) {
      exit(EXIT_FAILURE);
    }
  }

  // Given maximize(x), add the variable __MINIMIZE_OBJ with constraint __MINIMIZE_OBJ = -x.
  void add_minimize_objective_var(F& f, const F::Existential& max_var) {
    if(f.is(F::Seq)) {
      if(f.sig() == Sig::MAXIMIZE && f.seq(0).is_variable()) {
        LVar<basic_allocator_type> minimize_obj("__MINIMIZE_OBJ");
        f = F::make_binary(f,
          Sig::AND,
          F::make_binary(
            F::make_exists(f.seq(0).type(), minimize_obj, battery::get<1>(max_var)),
            Sig::AND,
            F::make_binary(
              F::make_lvar(f.seq(0).type(), minimize_obj),
              Sig::EQ,
              F::make_unary(Sig::NEG, f.seq(0)))));
      }
      else if(f.sig() == Sig::AND) {
        for(int i = 0; i < f.seq().size(); ++i) {
          add_minimize_objective_var(f.seq(i), max_var);
        }
      }
    }
  }

  void preprocess_tcn(F& f) {
    f = ternarize(f, VarEnv<BasicAllocator>(), {0,1,2});
    battery::vector<F> extra;
    f = normalize(f, extra);
    size_t num_vars = num_quantified_vars(f);
    stats.print_stat("tnf_variables", num_vars);
    stats.print_stat("tnf_constraints", num_tnf_constraints(f));
    allocate(num_vars, true);
    if(!interpret(f)) {
      exit(EXIT_FAILURE);
    }
    simplifier->init_env(env);
    if(config.disable_simplify) {
      /** Even when we don't simplify, we still need to initialize the equivalence classes.
       * This is necessary to call `print_variable` on `simplifier` when finding a solution. */
      simplifier->initialize(num_vars, 0);
      return;
    }
    auto& tnf = f.seq();
    simplifier->initialize_tnf(num_vars, tnf);
    size_t preprocessing_fixpoint_iterations = 0;
    SimplifierStats preprocessing_stats;
    size_t eliminated_variables = 0;
    local::B has_changed = true;
    GaussSeidelIteration fp_engine;
    /** We apply several preprocessing steps until we reach a fixpoint. */
    while(!iprop->is_bot() && has_changed) {
      has_changed = false;
      preprocessing_fixpoint_iterations++;
      SimplifierStats local_preprocessing_stats;
      fp_engine.fixpoint(iprop->num_deductions(),
        [&](size_t i) { return iprop->deduce(i); },
        [&](){ return iprop->is_bot(); },
        has_changed);
      if(has_changed) {
        simplifier->meet_equivalence_classes();
      }
      has_changed |= simplifier->algebraic_simplify(tnf, local_preprocessing_stats);
      simplifier->eliminate_entailed_constraints(*iprop, tnf, local_preprocessing_stats);
      if(num_vars < 1000000) { // otherwise ICSE is too slow, needs to be improved.
        has_changed |= simplifier->i_cse(tnf, local_preprocessing_stats);
      }
      if(has_changed) {
        simplifier->meet_equivalence_classes();
        local_preprocessing_stats.print(stats, preprocessing_fixpoint_iterations);
      }
      preprocessing_stats.merge(local_preprocessing_stats);
    }
    // simplifier->eliminate_entailed_constraints(*iprop, tnf, preprocessing_stats);
    simplifier->eliminate_useless_variables(tnf, eliminated_variables);
    f = simplifier->deinterpret(tnf, true);
    F extra_f = F::make_nary(AND, std::move(extra));
    simplifier->substitute(extra_f);
    if(config.verbose_solving) {
      printf("%% Formula simplified.\n");
    }
    F f2 = F::make_binary(std::move(f), AND, std::move(extra_f));
    num_vars = num_quantified_vars(f2);
    preprocessing_stats.print(stats);
    stats.print_stat("eliminated_variables", eliminated_variables);
    stats.print_stat("preprocessing_fixpoint_iterations", preprocessing_fixpoint_iterations);
    stats.print_stat("variables_after_simplification", num_vars);
    stats.print_stat("constraints_after_simplification", num_tnf_constraints(f2));
    if(iprop->is_bot()) {
      return;
    }
    allocate(num_vars, false);
    if(!interpret(f2)) {
      exit(EXIT_FAILURE);
    }
  }

  const char* name_of_abstract_domain() const {
    #define STR_(x) #x
    #define STR(x) STR_(x)
    #ifdef TURBO_IPC_ABSTRACT_DOMAIN
      return "ipc_itv" STR(TURBO_ITV_BITS) "_z";
    #else
      return "pir_itv" STR(TURBO_ITV_BITS) "_z";
    #endif
  }

  const char* name_of_entailed_removal() const {
    #ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
      return "deactivated";
    #else
      return "by_indexes_scan";
    #endif
  }

  void preprocess() {
    auto start = stats.start_timer_host();
    FormulaPtr f_ptr = parse_cn();
    stats.print_stat("abstract_domain", name_of_abstract_domain());
    stats.print_stat("entailed_prop_removal", name_of_entailed_removal());
    if(config.arch == Arch::BAREBONES) {
      auto max_var = find_maximize_var(*f_ptr);
      if(max_var.has_value()) {
        auto max_var_decl = find_existential_of(*f_ptr, max_var.value());
        if(max_var_decl.has_value()) {
          add_minimize_objective_var(*f_ptr, max_var_decl.value());
        }
      }
    }
  #ifdef TURBO_IPC_ABSTRACT_DOMAIN
    constexpr bool use_ipc = true;
  #else
    constexpr bool use_ipc = false;
  #endif
    if(use_ipc && !config.force_ternarize) {
      preprocess_ipc(*f_ptr);
    }
    else {
      preprocess_tcn(*f_ptr);
    }
    push_eps_strategy();
    std::mt19937 random_generator(config.seed);
    split->shuffle_random_strategies(random_generator);
    if(config.network_analysis) {
      if constexpr(use_ipc) {
        printf("%% WARNING: -network_analysis option is only valid with the PIR abstract domain.\n");
      }
      else {
        analyze_pir();
      }
    }
    stats.stop_timer(Timer::PREPROCESSING, start);
    stats.print_timing_stat("preprocessing_time", Timer::PREPROCESSING);
    stats.print_mzn_end_stats();
  }

private:
  template <class F>
  CUDA bool interpret_default_strategy() {
    typename F::Sequence seq;
    seq.push_back(F::make_nary("first_fail", {}));
    seq.push_back(F::make_nary("indomain_min", {}));
    F search_strat = F::make_nary("search", std::move(seq));
    if(!interpret_and_diagnose_and_tell(search_strat, env, *bab)) {
      return false;
    }
    return true;
  }

  void push_eps_strategy() {
    if(config.eps_var_order == "default") {
      return;
    }
    auto var_strat = variable_order_of_string(config.eps_var_order);
    if(!var_strat.has_value()) {
      printf("Unrecognized option `-eps_var_order %s`\n", config.eps_var_order.data());
      exit(EXIT_FAILURE);
    }
    auto value_strat = value_order_of_string(config.eps_value_order);
    if(!value_strat.has_value()) {
      printf("Unrecognized option `-eps_value_order %s`\n", config.eps_value_order.data());
      exit(EXIT_FAILURE);
    }
    split->push_eps_strategy(var_strat.value(), value_strat.value());
  }

  struct vstat {
    size_t num_occurrences;
    bool infinite_domain;
    size_t domain_size;
    vstat() = default;
  };

  /** Only for PIR abstract domain. */
  void analyze_pir() const {
    if(config.verbose_solving) {
      printf("%% Analyzing the constraint network...\n");
    }
    if(iprop->is_bot()) {
      printf("%% Constraint network UNSAT at root, analysis cancelled...\n");
      return;
    }
    std::vector<vstat> vstats(store->vars(), vstat{});
    for(int i = 0; i < store->vars(); ++i) {
      auto width = (*store)[i].width().lb();
      vstats[i].infinite_domain = width.is_top();
      if(!vstats[i].infinite_domain) {
        vstats[i].domain_size = width.value() + 1;
      }
    }

    std::map<Sig, size_t> op_stats{{{Sig::EQ,0}, {Sig::LEQ,0}, {Sig::NEQ,0}, {Sig::GT,0}, {ADD,0}, {MUL,0}, {EMOD,0}, {EDIV,0}, {MIN,0}, {MAX,0}}};
    std::map<Sig, size_t> reified_op_stats{{{Sig::EQ,0}, {Sig::LEQ,0}}};

    for(int i = 0; i < iprop->num_deductions(); ++i) {
      bytecode_type bytecode = iprop->load_deduce(i);
      vstats[bytecode.x.vid()].num_occurrences++;
      vstats[bytecode.y.vid()].num_occurrences++;
      vstats[bytecode.z.vid()].num_occurrences++;
      if(!op_stats.contains(bytecode.op)) {
        printf("%% WARNING: operator not explicitly managed in PIR: %d\n", bytecode.op);
        op_stats[bytecode.op] = 0;
      }
      auto dom = iprop->project(bytecode.x);
      if((is_arithmetic_comparison(bytecode.op)) &&
        (dom.lb().value() != dom.ub().value() || dom.lb().value() == 0))
      {
        if(dom.lb().value() != dom.ub().value()) {
          reified_op_stats[bytecode.op]++;
        }
        else {
          op_stats[negate_arithmetic_comparison(bytecode.op)]++;
        }
      }
      else {
        op_stats[bytecode.op]++;
      }
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
    stats.print_memory_statistics(config.verbose_solving, "sum_props_of_vars", sum_props_of_vars*4); // 1 integer per propagator indexes.
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
    stats.print_memory_statistics(config.verbose_solving, "sum_domain_size", sum_domain_size/8);
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
    for(auto& [sig,count] : op_stats) {
      std::string stat_name("num_op_");
      stat_name += string_of_sig_txt(sig);
      stats.print_stat(stat_name.c_str(), count);
    }
    for(auto& [sig,count] : reified_op_stats) {
      std::string stat_name("num_op_reified_");
      stat_name += string_of_sig_txt(sig);
      stats.print_stat(stat_name.c_str(), count);
    }
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
    print_solution(*best);
  }

  template <class BestStore>
  CUDA void print_solution(const BestStore& best_store) {
    solver_output.print_solution(env, best_store, *simplifier);
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

template <class Universe, class Allocator = battery::standard_allocator>
using CP = AbstractDomains<Universe,
  battery::statistics_allocator<Allocator>,
  battery::statistics_allocator<UniqueLightAlloc<Allocator, 0>>,
  battery::statistics_allocator<UniqueLightAlloc<Allocator, 1>>>;

#endif
