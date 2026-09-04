// Copyright 2025 Pierre Talbot

#ifndef TURBO_INTERPRETATION_HPP
#define TURBO_INTERPRETATION_HPP

#include <optional>

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/string.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "battery/allocator.hpp"

#include "lala/logic/logic.hpp"
#include "lala/universes/arith_bound.hpp"
#include "lala/universes/nbitset.hpp"
#include "lala/interval.hpp"
#include "lala/vstore.hpp"

#include "lala/pir.hpp"

#include "search_strategy.hpp"

/**
 * This file gathers the interpretation of logical formulas into the abstract universes and
 * abstract domains used by Turbo, and the deinterpretation going the other way.
 *
 * Interpretation is deliberately kept outside of the abstract domains: a domain is a lattice and
 * has no reason to know about logical formulas -- the same domains are useful for program
 * analysis or neural network verification, where the concrete object is not a formula. Moreover,
 * a product of domains admits more than one interpretation of the same formula, so there is no
 * single interpretation function that belongs "inside" a domain.
 *
 * The file is layered bottom-up:
 *   1. Diagnostics: `IDiagnostics` and the macros used to report why a formula is uninterpretable.
 *   2. Environment: interpretation of quantifiers and variable occurrences into `AVar`.
 *   3. Pre-universes: interpretation of *constants* (`pre_interpreter`).
 *   4. Abstract universes: `ArithBound`, `NBitset`, `Interval`.
 *   5. Generic interpretation: `ginterpret_in` and friends, shared by all lattices.
 *   6. Abstract domains: `VStore`, `PIR`, `Simplifier`.
 *   7. Constraint network: the top-level dispatch specific to Turbo's input language.
 */

namespace lala {

/* ------------------------------------------------------------------------------------------- *
 * 1. Diagnostics
 * ------------------------------------------------------------------------------------------- */

/** `IDiagnostics` is used to diagnose why a formula cannot be interpreted (error) or if it was
    interpreted by under- or over-approximation (warnings).
    If a formula cannot be interpreted, we must explain why.
    This is similar to compilation errors in a compiler. */
class IDiagnostics {
public:
  using allocator_type = battery::standard_allocator;
  using F = TFormula<allocator_type>;
  using this_type = IDiagnostics;

private:
  battery::string<allocator_type> ad_name;
  battery::string<allocator_type> description;
  F uninterpretable_formula;
  AType aty;
  battery::vector<IDiagnostics, allocator_type> suberrors;
  bool fatal;

  CUDA void print_indent(int indent) const {
    printf("%% ");
    for(int i = 0; i < indent; ++i) {
      printf(" ");
    }
  }

  CUDA void print_line(const char* line, int indent) const {
    print_indent(indent);
    printf("%s", line);
  }

public:
  CUDA NI IDiagnostics(): fatal(false), aty(-2) {}   // -2 is a special value indicating it is a top-level diagnostics.

  // If fatal is false, it is considered as a warning.
  template <class F2>
  CUDA NI IDiagnostics(bool fatal,
    battery::string<allocator_type> ad_name,
    battery::string<allocator_type> description,
    const F2& uninterpretable_formula,
    AType aty = UNTYPED)
   : fatal(fatal),
     ad_name(std::move(ad_name)),
     description(std::move(description)),
     uninterpretable_formula(uninterpretable_formula),
     aty(aty)
  {}

  CUDA NI this_type& add_suberror(IDiagnostics&& suberror) {
    fatal |= suberror.is_fatal();
    suberrors.push_back(std::move(suberror));
    return *this;
  }

  CUDA size_t num_suberrors() const {
    return suberrors.size();
  }

  /** Deletes all suberrors between `i` and `n-1`. */
  CUDA void cut(size_t i) {
    suberrors.resize(i);
    fatal = false;
    for(size_t j = 0; j < suberrors.size(); ++j) {
      if(suberrors[j].is_fatal()) {
        fatal = true;
        return;
      }
    }
  }

  /** This operator moves all `suberrors[i..(n-1)]` as a suberror of `suberrors[i-1]`.
   * If only warnings are present, `suberrors[i-1]` is converted into a warning.
   * If `succeeded` is true, then all suberrors are erased.
   */
  CUDA void merge(bool succeeded, size_t i) {
    assert(i > 0);
    assert(i <= suberrors.size());
    suberrors[i-1].fatal = !succeeded;
    for(size_t j = i; j < suberrors.size(); ++j) {
      // In case of success, we erase the fatal suberrors.
      if(!succeeded || !suberrors[j].is_fatal()) {
        suberrors[i-1].add_suberror(std::move(suberrors[j]));
      }
    }
    cut((suberrors[i-1].num_suberrors() == 0 && succeeded) ? i-1 : i);
  }

  CUDA NI void print(int indent = 0) const {
    // If it is not a top-level error, we print it, otherwise all errors are listed as `suberrors`.
    if(aty != -2) {
      if(fatal) {
        print_line("[error] ", indent);
      }
      else {
        print_line("[warning] ", indent);
      }
      print_line("Uninterpretable formula.", indent);
      print_indent(indent);
      printf("  Abstract domain: %s\n", ad_name.data());
      print_line("  Abstract type: ", indent);
      if(aty == UNTYPED) {
        printf("untyped\n");
      }
      else {
        printf("%d\n", aty);
      }
      print_line("  Formula: ", indent);
      uninterpretable_formula.print(true);
      printf("\n");
      print_indent(indent);
      printf("  Description: %s\n", description.data());
    }
    else {
      indent -= 2;
    }
    for(int i = 0; i < suberrors.size(); ++i) {
      suberrors[i].print(indent + 2);
      printf("\n");
    }
  }

  CUDA bool is_fatal() const { return fatal; }
  CUDA bool has_warning() const {
    for(int i = 0; i < suberrors.size(); ++i) {
      if(!suberrors[i].is_fatal()) {
        return true;
      }
    }
    return false;
  }
};

#define INTERPRETATION_ERROR(MSG) \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostics(true, name, (MSG), f)); \
  }

#define INTERPRETATION_WARNING(MSG) \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostics(false, name, (MSG), f)); \
  }

#define RETURN_INTERPRETATION_ERROR(MSG) \
  INTERPRETATION_ERROR(MSG) \
  return false;

#define RETURN_INTERPRETATION_WARNING(MSG) \
  INTERPRETATION_WARNING(MSG) \
  return true;

/** This macro creates a high-level error message that is possibly erased if `call` does not lead to any error.
 * If `call` leads to errors, these errors are moved as suberrors of the high-level error message.
 * Additionally, `merge` is executed if `call` does not lead to any error.
 */
#define CALL_WITH_ERROR_CONTEXT_WITH_MERGE(MSG, CALL, MERGE) \
  size_t error_context = 0; \
  if constexpr(diagnose) { \
    diagnostics.add_suberror(IDiagnostics(false, name, (MSG), f)); \
    error_context = diagnostics.num_suberrors(); \
  } \
  bool res = CALL; \
  if constexpr(diagnose) { \
    diagnostics.merge(res, error_context); \
  } \
  if(res) { MERGE; } \
  return res;

#define CALL_WITH_ERROR_CONTEXT(MSG, CALL) \
  CALL_WITH_ERROR_CONTEXT_WITH_MERGE(MSG, CALL, {})

/* ------------------------------------------------------------------------------------------- *
 * 2. Environment
 * ------------------------------------------------------------------------------------------- */

/** A string in the allocator of the formula `F`, used to build diagnostics messages. */
template <class F> using fstring = battery::string<typename F::allocator_type>;

/** Interpret an existential quantifier in `env`, and return the abstract variable created.
 * Variable redeclaration does not lead to an error, instead the abstract type of the variable is
 * added to the abstract variables list (`avars`) of the variable. */
template <bool diagnose = false, class F, class Alloc>
CUDA NI bool interpret_existential_in(const F& f, VarEnv<Alloc>& env, AVar& avar, IDiagnostics& diagnostics) {
  const char* name = "VarEnv";
  const auto& vname = battery::get<0>(f.exists());
  if(f.type() == UNTYPED) {
    RETURN_INTERPRETATION_ERROR("Untyped abstract type: variable `" + vname + "` has no abstract type.");
  }
  auto var = env.variable_of(vname);
  if(var.has_value()) {
    if(var->get().sort != battery::get<1>(f.exists())) {
      RETURN_INTERPRETATION_ERROR("Invalid redeclaration with different sort: variable `" + vname + "` has already been declared and the sort does not coincide.");
    }
  }
  avar = env.extends_vars(f.type(), vname, battery::get<1>(f.exists()));
  return true;
}

/** Interpret an occurrence of a logical variable in `env`. */
template <bool diagnose = false, class F, class Alloc>
CUDA NI bool interpret_lv_in(const F& f, const VarEnv<Alloc>& env, AVar& avar, IDiagnostics& diagnostics) {
  const char* name = "VarEnv";
  const auto& vname = f.lv();
  auto var = env.variable_of(vname);
  if(var.has_value()) {
    if(f.type() != UNTYPED) {
      auto avarf = var->get().avar_of(f.type());
      if(avarf.has_value()) {
        avar = AVar(*avarf);
        return true;
      }
      else {
        RETURN_INTERPRETATION_ERROR("Variable `" + vname + "` has not been declared in the abstract domain `" + fstring<F>::from_int(f.type()) + "`.");
      }
    }
    else {
      // We take the first abstract variable as a representative. Need more thought on this, but currently we need it for the simplifier, because each variable is typed in both PC and Simplifier, and this interpretation fails.
      avar = AVar(var->get().avars[0]);
      return true;
    }
  }
  else {
    RETURN_INTERPRETATION_ERROR("Undeclared variable `" + vname + "`.");
  }
}

/** A variable environment can interpret formulas of two forms:
 *    - Existential formula with a valid abstract type (`f.type() != UNTYPED`).
 *    - Variable occurrence.
 * It returns an abstract variable (`AVar`) corresponding to the variable created (existential) or
 * already present (occurrence). */
template <bool diagnose = false, class F, class Alloc>
CUDA NI bool interpret_in(const F& f, VarEnv<Alloc>& env, AVar& avar, IDiagnostics& diagnostics) {
  const char* name = "VarEnv";
  if(f.is(F::E)) {
    return interpret_existential_in<diagnose>(f, env, avar, diagnostics);
  }
  else if(f.is(F::LV)) {
    return interpret_lv_in<diagnose>(f, env, avar, diagnostics);
  }
  else if(f.is(F::V)) {
    if(env.contains(f.v())) {
      avar = f.v();
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Undeclared abstract variable `" + fstring<F>::from_int(f.v().aty()) + ", " + fstring<F>::from_int(f.v().vid()) + "`.");
    }
  }
  else {
    RETURN_INTERPRETATION_ERROR("Unsupported formula: `VarEnv` can only interpret quantifiers and occurrences of variables.");
  }
}

/* ------------------------------------------------------------------------------------------- *
 * 3. Pre-universes: interpretation of constants
 * ------------------------------------------------------------------------------------------- */

/** Interpretation of the *constants* of a pre-universe, and the deinterpretation of a value back
 * into a logical constant. Specialize this class to give a pre-universe a different
 * interpretation of constants. */
template <class PreUniverse>
struct pre_interpreter;

/** Interpret a constant in the lattice of decreasing integers according to the downset semantics.
    Overflows are not verified.
    Interpretations:
      * Formulas of kind `F::Z` are interpreted exactly: \f$ [\![ x:\mathbb{Z} \leq k:\mathbb{Z} ]\!] = k \f$.
      * Formulas of kind `F::R` are over-approximated: \f$ [\![ x:\mathbb{Z} \leq [l..u]:\mathbb{R} ]\!] = \lfloor u \rfloor \f$.
    Examples:
      * \f$ [\![x <= [3.5..3.5]:R ]\!] = 3 \f$: there is no integer greater than 3 satisfying this constraint.
      * \f$ [\![x <= [2.9..3.1]:R ]\!] = 3 \f$. */
template <class VT>
struct pre_interpreter<PreZUB<VT>> {
  using pre_universe = PreZUB<VT>;
  using value_type = typename pre_universe::value_type;

private:
  template<bool diagnose, bool is_tell, bool dualize, class F>
  CUDA NI static bool interpret_constant(const F& f, value_type& k, IDiagnostics& diagnostics) {
    const char* name = pre_universe::name;
    if(f.is(F::Z)) {
      auto z = f.z();
      if(z == pre_universe::bot() || z == pre_universe::top()) {
        RETURN_INTERPRETATION_ERROR("Constant of sort `Int` with the minimal or maximal representable value of the underlying integer type. We use those values to model negative and positive infinities. Example: Suppose we use a byte type, `x >= 256` is interpreted as `x >= INF` which is always false and thus is different from the intended constraint.");
      }
      k = z;
      return true;
    }
    else if(f.is(F::R)) {
      if constexpr(dualize) {
        if constexpr(is_tell) {
          k = battery::ru_cast<value_type>(battery::get<0>(f.r()));
        }
        else {
          k = battery::ru_cast<value_type>(battery::get<1>(f.r()));
        }
      }
      else {
        if constexpr(is_tell) {
          k = battery::rd_cast<value_type>(battery::get<1>(f.r()));
        }
        else {
          k = battery::rd_cast<value_type>(battery::get<0>(f.r()));
        }
      }
      return true;
    }
    else if(f.is(F::B)) {
      k = f.b() ? pre_universe::one() : pre_universe::zero();
      return true;
    }
    RETURN_INTERPRETATION_ERROR("Only constants of sorts `Int`, `Bool` and `Real` can be interpreted by an integer abstract universe.");
  }

public:
  template<bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_tell(const F& f, value_type& tell, IDiagnostics& diagnostics) {
    return interpret_constant<diagnose, true, dualize>(f, tell, diagnostics);
  }

  /** Similar to `interpret_tell` but the formula is under-approximated, in particular: \f$ [\![ x:\mathbb{Z} \leq [l..u]:\mathbb{R} ]\!] = \lfloor u \rfloor \f$. */
  template<bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_ask(const F& f, value_type& ask, IDiagnostics& diagnostics) {
    return interpret_constant<diagnose, false, dualize>(f, ask, diagnostics);
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with
      the current abstract universe. Variables of type `Int` are interpreted exactly
      (\f$ \mathbb{Z} = \gamma(\top) \f$). */
  template<bool diagnose, class F, bool dualize = false>
  CUDA NI static bool interpret_type(const F& f, value_type& k, IDiagnostics& diagnostics) {
    const char* name = pre_universe::name;
    assert(f.is(F::E));
    const auto& sort = battery::get<1>(f.exists());
    if(sort.is_int()) {
      k = dualize ? pre_universe::bot() : pre_universe::top();
      return true;
    }
    else {
      const auto& vname = battery::get<0>(f.exists());
      RETURN_INTERPRETATION_ERROR("The type of `" + vname + "` can only be `Int`.")
    }
  }

  /** Given an Integer value, create a logical constant representing that value.
   * Note that the lattice order has no influence here.
   * \pre `v != bot()` and `v != top()`. */
  template<class F>
  CUDA static F deinterpret(const value_type& v) {
    return F::make_z(v);
  }
};

/** `PreZLB` is the dual of `PreZUB`: tell becomes ask and vice-versa. */
template <class VT>
struct pre_interpreter<PreZLB<VT>> {
  using pre_universe = PreZLB<VT>;
  using dual_type = typename pre_universe::dual_type;
  using value_type = typename pre_universe::value_type;

  template <bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_tell(const F& f, value_type& tell, IDiagnostics& diagnostics) {
    return pre_interpreter<dual_type>::template interpret_ask<diagnose, F, true>(f, tell, diagnostics);
  }

  template <bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_ask(const F& f, value_type& ask, IDiagnostics& diagnostics) {
    return pre_interpreter<dual_type>::template interpret_tell<diagnose, F, true>(f, ask, diagnostics);
  }

  template <bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_type(const F& f, value_type& k, IDiagnostics& diagnostics) {
    return pre_interpreter<dual_type>::template interpret_type<diagnose, F, true>(f, k, diagnostics);
  }

  template<class F>
  CUDA static F deinterpret(const value_type& v) {
    return pre_interpreter<dual_type>::template deinterpret<F>(v);
  }
};

/** Interpret a constant in the lattice of decreasing floating-point numbers according to the
    downset semantics.
    Interpretations:
      * Formulas of kind `F::Z` might be over-approximated (if the integer cannot be represented in a floating-point number because it is too large).
      * Formulas of kind `F::R` might be over-approximated to the upper bound of the interval (if the real number is represented by an interval [lb..ub] where lb != ub).
      * Other kind of formulas are not supported. */
template <class VT>
struct pre_interpreter<PreFUB<VT>> {
  using pre_universe = PreFUB<VT>;
  using value_type = typename pre_universe::value_type;

private:
  template<bool diagnose, bool is_tell, class F>
  CUDA NI static bool interpret_constant(const F& f, value_type& k, IDiagnostics& diagnostics) {
    const char* name = pre_universe::name;
    if(f.is(F::Z)) {
      auto z = f.z();
      // We do not consider the min and max values of integers to be infinities when they are part of the logical formula.
      if constexpr(is_tell) {
        k = battery::ru_cast<value_type, decltype(z), false>(z);
      }
      else {
        k = battery::rd_cast<value_type, decltype(z), false>(z);
      }
      return true;
    }
    else if(f.is(F::R)) {
      if constexpr(is_tell) {
        k = battery::ru_cast<value_type>(battery::get<1>(f.r()));
      }
      else {
        k = battery::rd_cast<value_type>(battery::get<0>(f.r()));
      }
      return true;
    }
    RETURN_INTERPRETATION_ERROR("Only a constant of sort `Int` or `Real` can be interpreted by a floating-point abstract universe.")
  }

public:
  template<bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_tell(const F& f, value_type& k, IDiagnostics& diagnostics) {
    return interpret_constant<diagnose, true>(f, k, diagnostics);
  }

  /** Same as `interpret_tell` but the constant is under-approximated instead. */
  template<bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_ask(const F& f, value_type& k, IDiagnostics& diagnostics) {
    return interpret_constant<diagnose, false>(f, k, diagnostics);
  }

  /** Verify if the type of a variable, introduced by an existential quantifier, is compatible with
      the current abstract universe.
      Interpretations:
        * Variables of type `Int` are always over-approximated (\f$ \mathbb{Z} \subseteq \gamma(\top) \f$).
        * Variables of type `Real` are represented exactly (only initially because \f$ \mathbb{R} = \gamma(\top) \f$). */
  template<bool diagnose, class F, bool dualize = false>
  CUDA NI static bool interpret_type(const F& f, value_type& k, IDiagnostics& diagnostics) {
    const char* name = pre_universe::name;
    assert(f.is(F::E));
    const auto& vname = battery::get<0>(f.exists());
    const auto& cty = battery::get<1>(f.exists());
    if(cty.is_int()) {
      k = dualize ? pre_universe::bot() : pre_universe::top();
      RETURN_INTERPRETATION_WARNING("Variable `" + vname + "` of sort `Int` is over-approximated in a floating-point abstract universe.");
    }
    else if(cty.is_real()) {
      k = dualize ? pre_universe::bot() : pre_universe::top();
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Variable `" + vname + "` can only be of sort `Real`, or be over-approximated if the sort is `Bool` or `Int`.");
    }
  }

  /** Given a floating-point value, create a logical constant representing that value.
   * The constant is represented by a singleton interval of `double` [v..v].
   * Note that the lattice order has no influence here.
   * \pre `v != bot()` and `v != top()`. */
  template<class F>
  CUDA static F deinterpret(const value_type& v) {
    return F::make_real(v, v);
  }
};

/** `PreFLB` is the dual of `PreFUB`. */
template <class VT>
struct pre_interpreter<PreFLB<VT>> {
  using pre_universe = PreFLB<VT>;
  using dual_type = typename pre_universe::dual_type;
  using value_type = typename pre_universe::value_type;

  template <bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_tell(const F& f, value_type& tell, IDiagnostics& diagnostics) {
    return pre_interpreter<dual_type>::template interpret_ask<diagnose, F>(f, tell, diagnostics);
  }

  template <bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_ask(const F& f, value_type& ask, IDiagnostics& diagnostics) {
    return pre_interpreter<dual_type>::template interpret_tell<diagnose, F>(f, ask, diagnostics);
  }

  template<bool diagnose, class F, bool dualize = false>
  CUDA static bool interpret_type(const F& f, value_type& k, IDiagnostics& diagnostics) {
    return pre_interpreter<dual_type>::template interpret_type<diagnose, F, true>(f, k, diagnostics);
  }

  template<class F>
  CUDA static F deinterpret(const value_type& v) {
    return pre_interpreter<dual_type>::template deinterpret<F>(v);
  }
};

/* ------------------------------------------------------------------------------------------- *
 * 4. Abstract universes
 * ------------------------------------------------------------------------------------------- */

/* --- ArithBound --- */

namespace impl {
  /** Interpret a formula of the form `x <sig> k` in an arithmetic bound. */
  template<bool diagnose = false, class F, class U, class Mem>
  CUDA NI bool interpret_tell_x_op_k(const F& f, ArithBound<U, Mem>& tell, IDiagnostics& diagnostics) {
    using A = ArithBound<U, Mem>;
    using local_type = typename A::local_type;
    using pre_universe = typename A::pre_universe;
    using value_type = typename A::value_type;
    const char* name = A::name;
    value_type value = pre_universe::top();
    bool res = pre_interpreter<pre_universe>::template interpret_tell<diagnose>(f.seq(1), value, diagnostics);
    if(res) {
      if(f.sig() == EQ || f.sig() == U::sig_order()) {  // e.g., x <= 4 or x >= 4.24
        tell.meet(local_type(value));
      }
      else if(f.sig() == U::sig_strict_order()) {  // e.g., x < 4 or x > 4.24
        if constexpr(A::preserve_concrete_covers) {
          tell.meet(local_type(pre_universe::prev(value)));
        }
        else {
          tell.meet(local_type(value));
        }
      }
      else {
        RETURN_INTERPRETATION_ERROR("The symbol `" + LVar<typename F::allocator_type>(string_of_sig(f.sig())) + "` is not supported in the tell language of this universe.");
      }
    }
    return res;
  }

  /** Interpret a formula of the form `x <sig> k` in the ask language of an arithmetic bound. */
  template<bool diagnose = false, class F, class U, class Mem>
  CUDA NI bool interpret_ask_x_op_k(const F& f, ArithBound<U, Mem>& tell, IDiagnostics& diagnostics) {
    using A = ArithBound<U, Mem>;
    using local_type = typename A::local_type;
    using pre_universe = typename A::pre_universe;
    using value_type = typename A::value_type;
    const char* name = A::name;
    value_type value = pre_universe::top();
    bool res = pre_interpreter<pre_universe>::template interpret_ask<diagnose>(f.seq(1), value, diagnostics);
    if(res) {
      if(f.sig() == U::sig_order()) {
        tell.meet(local_type(value));
      }
      else if(f.sig() == NEQ || f.sig() == U::sig_strict_order()) {
        // We could actually do a little bit better in the case of FLB/FUB.
        // If the real number `k` is approximated by `[f, g]`, it actually means `]f, g[` so we could safely choose `r` since it already under-approximates `k`.
        tell.meet(local_type(pre_universe::prev(value)));
      }
      else {
        RETURN_INTERPRETATION_ERROR("The symbol `" + LVar<typename F::allocator_type>(string_of_sig(f.sig())) + "` is not supported in the ask language of this universe.");
      }
    }
    return res;
  }

  /** Interpret `x in S` in an arithmetic bound. */
  template<bool diagnose = false, class F, class U, class Mem>
  CUDA NI bool interpret_tell_set(const F& f, ArithBound<U, Mem>& tell, IDiagnostics& diagnostics) {
    using A = ArithBound<U, Mem>;
    using local_type = typename A::local_type;
    using pre_universe = typename A::pre_universe;
    using value_type = typename A::value_type;
    const char* name = A::name;
    if(!f.seq(1).is(F::S)) {
      RETURN_INTERPRETATION_ERROR("The constant `S` in a constraint `x in S` must be a set.");
    }
    const auto& set = f.seq(1).s();
    if(set.size() == 0) {
      tell.meet_bot();
      return true;
    }
    value_type join_s = pre_universe::bot();
    constexpr int bound_index = A::is_lower_bound ? 0 : 1;
    // We interpret each component of the set and take the meet of all the results.
    for(int i = 0; i < set.size(); ++i) {
      auto bound = battery::get<bound_index>(set[i]);
      value_type set_element = pre_universe::top();
      bool res = pre_interpreter<pre_universe>::template interpret_tell<diagnose>(bound, set_element, diagnostics);
      if(!res) {
        return false;
      }
      join_s = pre_universe::join(join_s, set_element);
    }
    tell.meet(local_type(join_s));
    return true;
  }
}

/** Expects a predicate of the form `x <op> k` where `x` is any variable's name, and `k` a constant.
 * The symbol `<op>` is expected to be `U::sig_order()`, `U::sig_strict_order()`,  `=` or `in`.
 * Existential formula \f$ \exists{x:T} \f$ can also be interpreted (only to top) depending on the
 * underlying pre-universe. */
template<bool diagnose = false, class F, class Env, class U, class Mem>
CUDA NI bool interpret_tell_in(const F& f, const Env&, ArithBound<U, Mem>& tell, IDiagnostics& diagnostics) {
  using A = ArithBound<U, Mem>;
  using local_type = typename A::local_type;
  using pre_universe = typename A::pre_universe;
  const char* name = A::name;
  if(f.is(F::E)) {
    typename U::value_type val;
    bool res = pre_interpreter<pre_universe>::template interpret_type<diagnose>(f, val, diagnostics);
    if(res) {
      tell.meet(local_type(val));
    }
    return res;
  }
  else {
    if(f.is_binary() && f.seq(0).is_variable() && f.seq(1).is_constant()) {
      // `x in k` is equivalent to `x >= meet k` where `>=` is the lattice order `U::sig_order()`.
      if(f.sig() == IN) {
        return impl::interpret_tell_set<diagnose>(f, tell, diagnostics);
      }
      else {
        return impl::interpret_tell_x_op_k<diagnose>(f, tell, diagnostics);
      }
    }
    else {
      RETURN_INTERPRETATION_ERROR("Only binary formulas of the form `x <sig> k` where if x is a variable and k is a constant are supported.");
    }
  }
}

/** Expects a predicate of the form `x <op> k` where `x` is any variable's name, and `k` a constant.
 * The symbol `<op>` is expected to be `U::sig_order()`, `U::sig_strict_order()` or `!=`. */
template<bool diagnose = false, class F, class Env, class U, class Mem>
CUDA NI bool interpret_ask_in(const F& f, const Env&, ArithBound<U, Mem>& ask, IDiagnostics& diagnostics) {
  const char* name = ArithBound<U, Mem>::name;
  if(f.is_binary() && f.seq(0).is_variable() && f.seq(1).is_constant()) {
    return impl::interpret_ask_x_op_k<diagnose>(f, ask, diagnostics);
  }
  else {
    RETURN_INTERPRETATION_ERROR("Only binary formulas of the form `x <sig> k` where if x is a variable and k is a constant are supported.");
  }
}

template<IKind kind, bool diagnose = false, class F, class Env, class U, class Mem>
CUDA NI bool interpret_in(const F& f, const Env& env, ArithBound<U, Mem>& value, IDiagnostics& diagnostics) {
  if constexpr(kind == IKind::TELL) {
    return interpret_tell_in<diagnose>(f, env, value, diagnostics);
  }
  else {
    return interpret_ask_in<diagnose>(f, env, value, diagnostics);
  }
}

/** Deinterpret the current value to a logical constant. */
template<class F, class U, class Mem>
CUDA NI F deinterpret_constant(const ArithBound<U, Mem>& a) {
  return pre_interpreter<typename ArithBound<U, Mem>::pre_universe>::template deinterpret<F>(a.value());
}

/** \return \f$ x <op> i \f$ where `x` is a variable's name, `i` the current value and `<op>` depends
 * on the underlying universe. If `U` preserves top, `true` is returned whenever \f$ a = \top \f$,
 * if it preserves bottom `false` is returned whenever \f$ a = \bot \f$.
 * We always return an exact approximation. */
template<class Env, class U, class Mem, class Allocator = typename Env::allocator_type>
CUDA NI TFormula<Allocator> deinterpret_in(const ArithBound<U, Mem>& a, AVar avar, const Env& env, const Allocator& allocator = Allocator()) {
  using A = ArithBound<U, Mem>;
  using F = TFormula<Allocator>;
  if(A::preserve_top && a.is_top()) {
    return F::make_true();
  }
  else if(A::preserve_bot && a.is_bot()) {
    return F::make_false();
  }
  return F::make_binary(
    F::make_avar(avar),
    U::sig_order(),
    deinterpret_constant<F>(a),
    UNTYPED, allocator);
}

/* --- NBitset --- */

namespace impl {
  template<bool diagnose, class F, size_t N, class Mem, class T>
  CUDA NI bool interpret_existential_bitset(const F& f, NBitset<N, Mem, T>& k, IDiagnostics& diagnostics) {
    using local_type = typename NBitset<N, Mem, T>::local_type;
    const char* name = NBitset<N, Mem, T>::name;
    const auto& sort = battery::get<1>(f.exists());
    if(sort.is_int()) {
      return true;
    }
    else if(sort.is_bool()) {
      k.meet(local_type(0,1));
      return true;
    }
    else {
      const auto& vname = battery::get<0>(f.exists());
      RETURN_INTERPRETATION_ERROR(("NBitset only supports variables of type `Int` or `Bool`, but `" + vname + "` has another sort."));
    }
  }

  template<bool diagnose, bool negated, class F, size_t N, class Mem, class T>
  CUDA NI bool interpret_tell_set_bitset(const F& f, const F& k, NBitset<N, Mem, T>& tell, IDiagnostics& diagnostics) {
    using A = NBitset<N, Mem, T>;
    using local_type = typename A::local_type;
    const char* name = A::name;
    using sort_type = Sort<typename F::allocator_type>;
    std::optional<sort_type> sort = f.seq(1).sort();
    if(sort.has_value() &&
       (sort.value() == sort_type(sort_type::Set, sort_type(sort_type::Int))
     || sort.value() == sort_type(sort_type::Set, sort_type(sort_type::Bool))))
    {
      const auto& set = f.seq(1).s();
      local_type join_s = local_type::bot();
      bool over_appx = false;
      for(int i = 0; i < set.size(); ++i) {
        int l = battery::get<0>(set[i]).to_z();
        int u = battery::get<1>(set[i]).to_z();
        join_s.join(local_type(l, u));
        if(l < 0 || u >= A::capacity() - 2) {
          over_appx = true;
        }
      }
      if constexpr(negated) {
        join_s = join_s.complement();
        // In any case both out-of-range flags must be set: if no element is below zero, then some
        // elements in the negation are; and if some elements are below zero it's not all of them.
        join_s.join_out_of_range();
      }
      tell.meet(join_s);
      if(over_appx) {
        RETURN_INTERPRETATION_WARNING("Constraint `x in S` is over-approximated because some elements of `S` fall outside the bitset.");
      }
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("NBitset only supports membership (`x in S`) where `S` is a set of integers.");
    }
  }

  template<bool diagnose, class F, size_t N, class Mem, class T>
  CUDA NI bool interpret_tell_x_op_k_bitset(const F& f, logic_int k, Sig sig, NBitset<N, Mem, T>& tell, IDiagnostics& diagnostics) {
    using A = NBitset<N, Mem, T>;
    using local_type = typename A::local_type;
    const char* name = A::name;
    if(sig == LT) {
      return interpret_tell_x_op_k_bitset<diagnose>(f, k-1, LEQ, tell, diagnostics);
    }
    else if(sig == GT) {
      return interpret_tell_x_op_k_bitset<diagnose>(f, k+1, GEQ, tell, diagnostics);
    }
    else if(k < 0 || k >= A::capacity() - 2) {
      if((k == -1 && sig == LEQ) || (k == A::capacity() - 2 && sig == GEQ)) {
        // this is fine because x <= -1 and x >= n-2 can be represented exactly.
      }
      else {
        INTERPRETATION_WARNING("Constraint `x <op> k` is over-approximated because `k` is not representable in the bitset. Note that for a bitset of size `n`, the only values representable exactly are in the interval `[0, n-3]` because two bits are used to represent all negative values and all values exceeding the size of the bitset.");
        // If it is NEQ, we can't give a better approximation than top.
        if(sig == NEQ) {
          return true;
        }
      }
    }
    switch(sig) {
      case EQ: tell.meet(local_type(k, k)); break;
      case NEQ: tell.meet(local_type(k, k).complement()); break;
      case LEQ: tell.meet(local_type(-1, k)); break;
      case GEQ: tell.meet(local_type(k, A::capacity())); break;
      default: RETURN_INTERPRETATION_ERROR("This symbol is not supported.");
    }
    return true;
  }

  template<bool diagnose, bool negated, class F, size_t N, class Mem, class T>
  CUDA NI bool interpret_binary_bitset(const F& f, NBitset<N, Mem, T>& tell, IDiagnostics& diagnostics) {
    const char* name = NBitset<N, Mem, T>::name;
    if(f.sig() == IN) {
      return interpret_tell_set_bitset<diagnose, negated>(f, f.seq(1), tell, diagnostics);
    }
    else if(f.seq(1).is(F::Z) || f.seq(1).is(F::B)) {
      return interpret_tell_x_op_k_bitset<diagnose>(f, f.seq(1).to_z(), f.sig(), tell, diagnostics);
    }
    else {
      RETURN_INTERPRETATION_ERROR("Only integer and Boolean constants are supported in NBitset.");
    }
  }
}

/** Support the following language where all constants `k` are integer or Boolean values:
 *   * `var x:Z`
 *   * `var x:B`
 *   * `x <op> k` where `k` is an integer constant and `<op>` in {==, !=, <, <=, >, >=}.
 *   * `x in S` where `S` is a set of integers.
 * It can be over-approximated if the element `k` falls out of the bitset. */
template<bool diagnose = false, class F, class Env, size_t N, class Mem, class T>
CUDA NI bool interpret_tell_in(const F& f, const Env& env, NBitset<N, Mem, T>& tell, IDiagnostics& diagnostics) {
  const char* name = NBitset<N, Mem, T>::name;
  if(f.is(F::E)) {
    return impl::interpret_existential_bitset<diagnose>(f, tell, diagnostics);
  }
  else if(f.is_unary() && f.sig() == NOT && f.seq(0).is_binary()) {
    return impl::interpret_binary_bitset<diagnose, true>(f.seq(0), tell, diagnostics);
  }
  else if(f.is_binary() && f.seq(0).is_variable() && f.seq(1).is_constant()) {
    return impl::interpret_binary_bitset<diagnose, false>(f, tell, diagnostics);
  }
  else {
    RETURN_INTERPRETATION_ERROR("Only binary formulas of the form `x <sig> k` where if x is a variable and k is a constant are supported. We also supports existential quantifier and membership in a set of integers (x in S).");
  }
}

/** Support the same language than the "tell language" without existential. */
template<bool diagnose = false, class F, class Env, size_t N, class Mem, class T>
CUDA NI bool interpret_ask_in(const F& f, const Env& env, NBitset<N, Mem, T>& k, IDiagnostics& diagnostics) {
  using local_type = typename NBitset<N, Mem, T>::local_type;
  const char* name = NBitset<N, Mem, T>::name;
  local_type b = local_type::top();
  auto nf = negate(f);
  if(!nf.has_value()) {
    RETURN_INTERPRETATION_ERROR("Could not negate the formula in order to ask-interpret it.");
  }
  if(f.is(F::E)) {
    RETURN_INTERPRETATION_ERROR("Existential quantification is not supported in ask interpretation.");
  }
  if(interpret_tell_in<diagnose>(nf.value(), env, b, diagnostics)) {
    k.meet(b.complement());
    return true;
  }
  else {
    return false;
  }
}

template<IKind kind, bool diagnose = false, class F, class Env, size_t N, class Mem, class T>
CUDA NI bool interpret_in(const F& f, const Env& env, NBitset<N, Mem, T>& k, IDiagnostics& diagnostics) {
  if constexpr(kind == IKind::ASK) {
    return interpret_ask_in<diagnose>(f, env, k, diagnostics);
  }
  else {
    return interpret_tell_in<diagnose>(f, env, k, diagnostics);
  }
}

/* --- Interval --- */

/** Support the same language than the Cartesian product, and more:
 *    * `var x:B` when the underlying universe is arithmetic and preserve concrete covers.
 * Therefore, the element `k` is always in \f$ \gamma(lb) \cap \gamma(ub) \f$. */
template<bool diagnose = false, class F, class Env, class U2>
CUDA NI bool interpret_tell_in(const F& f, const Env& env, Interval<U2>& k, IDiagnostics& diagnostics) {
  using A = Interval<U2>;
  using LB = typename A::LB;
  using UB = typename A::UB;
  using local_type = typename A::local_type;
  const char* name = A::name;
  if constexpr(LB::preserve_concrete_covers && LB::is_arithmetic) {
    if(f.is(F::E)) {
      auto sort = f.sort();
      if(sort.has_value() && sort->is_bool()) {
        k.meet(local_type(LB::geq_k(LB::pre_universe::zero()), UB::leq_k(UB::pre_universe::one())));
        return true;
      }
    }
  }
  bool r;
  CALL_WITH_ERROR_CONTEXT(
    "No component of this interval can interpret this formula.",
      (r = interpret_tell_in<diagnose>(f, env, k.lb(), diagnostics),
       r |= interpret_tell_in<diagnose>(f, env, k.ub(), diagnostics),
       r));
}

/** Support the same language than the Cartesian product, and more:
 *    * `x != k` is under-approximated by interpreting `x != k` in the lower bound.
 *    * `x == k` is interpreted by over-approximating `x == k` in both bounds and then verifying both bounds are the same.
 *    * `x in {[l..u]} is interpreted by under-approximating `x >= l` and `x <= u`. */
template<bool diagnose = false, class F, class Env, class U2>
CUDA NI bool interpret_ask_in(const F& f, const Env& env, Interval<U2>& k, IDiagnostics& diagnostics) {
  using A = Interval<U2>;
  using local_type = typename A::local_type;
  const char* name = A::name;
  local_type itv = local_type::top();
  if(f.is_binary() && f.sig() == NEQ) {
    return interpret_ask_in<diagnose>(f, env, k.lb(), diagnostics);
  }
  else if(f.is_binary() && f.sig() == EQ) {
    CALL_WITH_ERROR_CONTEXT_WITH_MERGE(
      "When interpreting equality, the underlying bounds LB and UB failed to agree on the same value.",
      (interpret_tell_in<diagnose>(f, env, itv.lb(), diagnostics) &&
       interpret_tell_in<diagnose>(f, env, itv.ub(), diagnostics) &&
       itv.lb() == itv.ub()),
      (k.meet(itv)));
  }
  else if(f.is_binary() && f.sig() == IN && f.seq(0).is_variable()
   && f.seq(1).is(F::S) && f.seq(1).s().size() == 1)
  {
    const auto& lb = battery::get<0>(f.seq(1).s()[0]);
    const auto& ub = battery::get<1>(f.seq(1).s()[0]);
    if(lb == ub) {
      CALL_WITH_ERROR_CONTEXT(
        "Failed to interpret the decomposition of set membership `x in {[v..v]}` into equality `x == v`.",
        (interpret_ask_in<diagnose>(F::make_binary(f.seq(0), EQ, lb), env, k, diagnostics)));
    }
    CALL_WITH_ERROR_CONTEXT_WITH_MERGE(
      "Failed to interpret the decomposition of set membership `x in {[l..u]}` into `x >= l /\\ x <= u`.",
      (interpret_ask_in<diagnose>(F::make_binary(f.seq(0), geq_of_constant(lb), lb), env, itv.lb(), diagnostics) &&
       interpret_ask_in<diagnose>(F::make_binary(f.seq(0), leq_of_constant(ub), ub), env, itv.ub(), diagnostics)),
      (k.meet(itv))
    );
  }
  bool r;
  CALL_WITH_ERROR_CONTEXT(
    "No component of this interval can interpret this formula.",
      (r = interpret_ask_in<diagnose>(f, env, k.lb(), diagnostics),
       r |= interpret_ask_in<diagnose>(f, env, k.ub(), diagnostics),
       r));
}

template<IKind kind, bool diagnose = false, class F, class Env, class U2>
CUDA NI bool interpret_in(const F& f, const Env& env, Interval<U2>& k, IDiagnostics& diagnostics) {
  if constexpr(kind == IKind::ASK) {
    return interpret_ask_in<diagnose>(f, env, k, diagnostics);
  }
  else {
    return interpret_tell_in<diagnose>(f, env, k, diagnostics);
  }
}

template<class Env, class U, class Allocator = typename Env::allocator_type>
CUDA NI TFormula<Allocator> deinterpret_in(const Interval<U>& a, AVar x, const Env& env, const Allocator& allocator = Allocator()) {
  using F = TFormula<Allocator>;
  if(a.is_bot()) {
    return F::make_false();
  }
  if(a.is_top()) {
    return F::make_true();
  }
  if(a.lb().is_top()) {
    return deinterpret_in(a.ub(), x, env, allocator);
  }
  else if(a.ub().is_top()) {
    return deinterpret_in(a.lb(), x, env, allocator);
  }
  F logical_lb = deinterpret_constant<F>(a.lb());
  F logical_ub = deinterpret_constant<F>(a.ub());
  logic_set<F> logical_set(1, allocator);
  logical_set[0] = battery::make_tuple(std::move(logical_lb), std::move(logical_ub));
  F set = F::make_set(std::move(logical_set));
  F var = F::make_avar(x);
  return F::make_binary(var, IN, std::move(set), UNTYPED, allocator);
}

/** Deinterpret the current value to a logical constant.
 * The lower bound is deinterpreted, and it is up to the user to check that interval is a singleton.
 * A special case is made for real numbers where both bounds are used, since the logical
 * interpretation uses an interval. */
template<class F, class U>
CUDA NI F deinterpret_constant(const Interval<U>& a) {
  F logical_lb = deinterpret_constant<F>(a.lb());
  if(logical_lb.is(F::R)) {
    F logical_ub = deinterpret_constant<F>(a.ub());
    battery::get<1>(logical_lb.r()) = battery::get<0>(logical_ub.r());
  }
  return logical_lb;
}

/* ------------------------------------------------------------------------------------------- *
 * 5. Generic interpretation
 * ------------------------------------------------------------------------------------------- */

/** Interpret `true` in the lattice `L`.
 * \return `true` if `L` preserves the top element w.r.t. the concrete domain or if `true` is
 * interpreted by under-approximation (kind == ASK). */
template <class L, IKind kind, bool diagnose = false, class F>
CUDA bool ginterpret_true(const F& f, IDiagnostics& diagnostics) {
  assert(f.is_true());
  if constexpr(kind == IKind::ASK || L::preserve_top) {
    return true;
  }
  else {
    const char* name = L::name;
    RETURN_INTERPRETATION_ERROR("Bottom is not preserved, hence we cannot over-approximate `true` in this abstract domain.");
  }
}

/** Extended and unified interface to ask and tell interpretation of a formula in an abstract
 * domain. It provides a default interpretation for common formulas such as `true`, `false` and
 * conjunction whenever `A` satisfies some lattice-theoretic conditions. */
template <IKind kind, bool diagnose = false, class A, class F, class Env, class I>
CUDA bool ginterpret_in(const A& a, const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) {
  const char* name = A::name;
  if(f.is_true()) {
    return ginterpret_true<A, kind, diagnose>(f, diagnostics);
  }
  else if(f.is_false()) {
    if constexpr(kind == IKind::TELL || A::preserve_bot) {
      // We don't know how `bot` is represented by this abstract domain, so we just forward the interpretation call.
      return interpret_in<kind, diagnose>(a, f, env, intermediate, diagnostics);
    }
    else {
      RETURN_INTERPRETATION_ERROR("Bot is not preserved, hence we cannot under-approximate `true` in this abstract domain.");
    }
  }
  else if(f.is(F::Seq) && f.sig() == AND) {
    if constexpr(kind == IKind::ASK || A::preserve_meet) {
      for(int i = 0; i < f.seq().size(); ++i) {
        if(!ginterpret_in<kind, diagnose>(a, f.seq(i), env, intermediate, diagnostics)) {
          return false;
        }
      }
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Meet is not preserved, hence we cannot over-approximate conjunctions in this abstract domain.");
    }
  }
  // In the other cases, we cannot provide a default interpretation, so we forward the call to the abstract domain.
  return interpret_in<kind, diagnose>(a, f, env, intermediate, diagnostics);
}

/** Extended and unified interface to ask and tell interpretation of a formula in an abstract
 * universe. It provides a default interpretation for common formulas such as `true`, `false`,
 * conjunction and disjunction whenever `U` satisfies some lattice-theoretic conditions. */
template <IKind kind, bool diagnose = false, class F, class Env, class U>
CUDA bool ginterpret_in(const F& f, const Env& env, U& value, IDiagnostics& diagnostics) {
  const char* name = U::name;
  if(f.is_true()) {
    return ginterpret_true<U, kind, diagnose>(f, diagnostics);
  }
  else if(f.is_false()) {
    if constexpr(kind == IKind::TELL || U::preserve_bot) {
      value.meet_bot();
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Bot is not preserved, hence we cannot under-approximate `true` in this abstract universe.");
    }
  }
  else if(f.is(F::Seq)) {
    if(f.sig() == AND) {
      if constexpr(kind == IKind::ASK || U::preserve_meet) {
        for(int i = 0; i < f.seq().size(); ++i) {
          if(!ginterpret_in<kind, diagnose>(f.seq(i), env, value, diagnostics)) {
            return false;
          }
        }
        return true;
      }
      else {
        RETURN_INTERPRETATION_ERROR("Meet is not preserved, hence we cannot over-approximate conjunctions in this abstract universe.");
      }
    }
    else if(f.sig() == OR) {
      if constexpr(kind == IKind::TELL || U::preserve_join) {
        using U2 = typename U::local_type;
        U2 join_value = U2::bot();
        for(int i = 0; i < f.seq().size(); ++i) {
          U2 x = U2::top();
          if(!ginterpret_in<kind, diagnose>(f.seq(i), env, x, diagnostics)) {
            return false;
          }
          join_value.join(x);
        }
        value.meet(join_value);
        return true;
      }
      else {
        RETURN_INTERPRETATION_ERROR("Join is not preserved, hence we cannot under-approximate disjunctions in this abstract universe.");
      }
    }
  }
  // In the other cases, we cannot provide a default interpretation, so we forward the call to the abstract element.
  return interpret_in<kind, diagnose>(f, env, value, diagnostics);
}

/** Top-level version of `ginterpret_in`, we restore `env` and `intermediate` in case of failure. */
template <IKind kind, bool diagnose = false, class A, class F, class Env, class I>
CUDA bool top_level_ginterpret_in(const A& a, const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) {
  auto snap = env.snapshot();
  I copy = intermediate;
  if(ginterpret_in<kind, diagnose>(a, f, env, intermediate, diagnostics)) {
    return true;
  }
  else {
    env.restore(snap);
    intermediate = std::move(copy);
    return false;
  }
}

template <class A, class Alloc = battery::standard_allocator, class Env>
CUDA A make_top(Env& env, Alloc alloc = Alloc{}) {
  if constexpr(A::is_abstract_universe) {
    return A::top();
  }
  else {
    return A::top(env, alloc);
  }
}

template <bool diagnose = false, class TellAlloc = battery::standard_allocator, class F, class Env, class L>
CUDA bool interpret_and_tell(const F& f, Env& env, L& value, IDiagnostics& diagnostics, TellAlloc tell_alloc = TellAlloc{}) {
  if constexpr(L::is_abstract_universe) {
    return ginterpret_in<IKind::TELL, diagnose>(f, env, value, diagnostics);
  }
  else {
    typename L::template tell_type<TellAlloc> tell(tell_alloc);
    if(top_level_ginterpret_in<IKind::TELL, diagnose>(value, f, env, tell, diagnostics)) {
      value.deduce(tell);
      return true;
    }
    else {
      return false;
    }
  }
}

template <class A, bool diagnose = false, class F, class Env, class TellAlloc = typename A::allocator_type>
CUDA std::optional<A> create_and_interpret_and_tell(const F& f,
 Env& env, IDiagnostics& diagnostics,
 typename A::allocator_type alloc = typename A::allocator_type{},
 TellAlloc tell_alloc = TellAlloc{})
{
  auto snap = env.snapshot();
  A a{make_top<A>(env, alloc)};
  if(interpret_and_tell<diagnose>(f, env, a, diagnostics, tell_alloc)) {
    return {std::move(a)};
  }
  else {
    env.restore(snap);
    return {};
  }
}

/* ------------------------------------------------------------------------------------------- *
 * 6. Abstract domains
 * ------------------------------------------------------------------------------------------- */

/* --- VStore --- */

namespace impl {
  template <bool diagnose, class F, class Env, class I, class U, class Alloc>
  CUDA NI bool interpret_existential_store(const VStore<U, Alloc>& store, const F& f, Env& env,
    I& tell, IDiagnostics& diagnostics)
  {
    using Alloc2 = typename I::allocator_type;
    using A = VStore<U, Alloc>;
    using local_universe = typename A::local_universe;
    assert(f.is(F::E));
    typename A::template var_dom<Alloc2> k;
    if(interpret_tell_in<diagnose>(f, env, k.dom, diagnostics)) {
      if(interpret_in<diagnose>(f.map_atype(store.aty()), env, k.avar, diagnostics)) {
        assert(k.avar.aty() == store.aty());
        tell.push_back(k);
        return true;
      }
    }
    return false;
  }

  /** Interpret a predicate without variables. */
  template <bool diagnose, class F, class Env, class I, class U, class Alloc>
  CUDA NI bool interpret_zero_predicate_store(const VStore<U, Alloc>& store, const F& f, const Env& env,
    I& tell, IDiagnostics& diagnostics)
  {
    using Alloc2 = typename I::allocator_type;
    using A = VStore<U, Alloc>;
    const char* name = A::name;
    if(f.is_true()) {
      return true;
    }
    else if(f.is_false()) {
      tell.push_back(typename A::template var_dom<Alloc2>(AVar{}, U::bot()));
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Only `true` and `false` can be interpreted in the store without being named.");
    }
  }

  /** Interpret a predicate with a single variable occurrence. */
  template <IKind kind, bool diagnose, class F, class Env, class I, class U, class Alloc>
  CUDA NI bool interpret_unary_predicate_store(const VStore<U, Alloc>& store, const F& f, const Env& env,
    I& tell, IDiagnostics& diagnostics)
  {
    using Alloc2 = typename I::allocator_type;
    using A = VStore<U, Alloc>;
    using local_universe = typename A::local_universe;
    const char* name = A::name;
    local_universe u;
    bool res = ginterpret_in<kind, diagnose>(f, env, u, diagnostics);
    if(res) {
      const auto& varf = var_in(f);
      // When it is not necessary, we try to avoid using the environment.
      // This is for instance useful when deduction operators add new constraints but do not have
      // access to the environment, and to avoid passing the environment around everywhere.
      if(varf.is(F::V)) {
        if(varf.v().aty() == store.aty() || varf.v().aty() == UNTYPED) {
          tell.push_back(typename A::template var_dom<Alloc2>(varf.v(), u));
        }
        else {
          RETURN_INTERPRETATION_ERROR("The variable was not declared in the current abstract element (but exists in other abstract elements).");
        }
      }
      else {
        auto var = var_in(f, env);
        if(!var.has_value()) {
          RETURN_INTERPRETATION_ERROR("Undeclared variable.");
        }
        auto avar = var->get().avar_of(store.aty());
        if(!avar.has_value()) {
          RETURN_INTERPRETATION_ERROR("The variable was not declared in the current abstract element (but exists in other abstract elements).");
        }
        assert(avar->aty() == store.aty());
        tell.push_back(typename A::template var_dom<Alloc2>(*avar, u));
      }
      return true;
    }
    else {
      RETURN_INTERPRETATION_ERROR("Could not interpret a unary predicate in the underlying abstract universe.");
    }
  }

  template <IKind kind, bool diagnose, class F, class Env, class I, class U, class Alloc>
  CUDA NI bool interpret_predicate_store(const VStore<U, Alloc>& store, const F& f, Env& env,
    I& tell, IDiagnostics& diagnostics)
  {
    const char* name = VStore<U, Alloc>::name;
    if(f.type() != UNTYPED && f.type() != store.aty()) {
      RETURN_INTERPRETATION_ERROR("The abstract type of this predicate does not match the one of the current abstract element.");
    }
    if constexpr(kind == IKind::TELL) {
      if(f.is(F::E)) {
        return interpret_existential_store<diagnose>(store, f, env, tell, diagnostics);
      }
    }
    switch(num_vars(f)) {
      case 0: return interpret_zero_predicate_store<diagnose>(store, f, env, tell, diagnostics);
      case 1: return interpret_unary_predicate_store<kind, diagnose>(store, f, env, tell, diagnostics);
      default: RETURN_INTERPRETATION_ERROR("Interpretation of n-ary predicate is not supported in VStore.");
    }
  }
}

/** The store of variables expects a formula with a single variable (including existential
 * quantifiers) that can be handled by the abstract universe `U`.
 *
 * Variables must be existentially quantified before a formula containing variables can be
 * interpreted. Variables are immediately assigned to an index of `VStore` and initialized to
 * \f$ \top_U \f$. Shadowing/redeclaration of variables with existential quantifier is not
 * supported.  The variable mapping is added to the environment only if the interpretation
 * succeeds.
 *
 * There is a small quirk: different stores might be produced if quantifiers do not appear in the
 * same order, because we attribute the first available index to variables when interpreting the
 * quantifier. In that case, the store will only be equivalent modulo the `env` structure. */
template <IKind kind, bool diagnose = false, class F, class Env, class I, class U, class Alloc>
CUDA NI bool interpret_in(const VStore<U, Alloc>& store, const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) {
  const char* name = VStore<U, Alloc>::name;
  if(f.is_untyped() || f.type() == store.aty()) {
    return impl::interpret_predicate_store<kind, diagnose>(store, f, env, intermediate, diagnostics);
  }
  else {
    RETURN_INTERPRETATION_ERROR("This abstract element cannot interpret a formula with a different type.");
  }
}

template <bool diagnose = false, class F, class Env, class I, class U, class Alloc>
CUDA NI bool interpret_tell_in(const VStore<U, Alloc>& store, const F& f, Env& env, I& tell, IDiagnostics& diagnostics) {
  return interpret_in<IKind::TELL, diagnose>(store, f, env, tell, diagnostics);
}

/** Similar to `interpret_tell_in` but does not support existential quantifiers and therefore
 * leaves `env` unchanged. */
template <bool diagnose = false, class F, class Env, class I, class U, class Alloc>
CUDA NI bool interpret_ask_in(const VStore<U, Alloc>& store, const F& f, const Env& env, I& ask, IDiagnostics& diagnostics) {
  return interpret_in<IKind::ASK, diagnose>(store, f, const_cast<Env&>(env), ask, diagnostics);
}

namespace impl {
  template<class U2, class Env, class Allocator2, class U, class Alloc>
  CUDA TFormula<typename Env::allocator_type> deinterpret_var_store(const VStore<U, Alloc>& store,
    AVar avar, const U2& dom, const Env& env, const Allocator2& allocator)
  {
    auto f = deinterpret_in(dom, avar, env, allocator);
    f.type_as(store.aty());
    map_avar_to_lvar(f, env);
    return std::move(f);
  }
}

template<class Env, class U, class Alloc, class Allocator2 = typename Env::allocator_type>
CUDA NI TFormula<Allocator2> deinterpret_in(const VStore<U, Alloc>& store, const Env& env, const Allocator2& allocator = Allocator2()) {
  using F = TFormula<Allocator2>;
  if(store.vars() == 0) {
    return store.is_bot() ? F::make_false() : F::make_true();
  }
  typename F::Sequence seq{allocator};
  for(int i = 0; i < store.vars(); ++i) {
    AVar v(store.aty(), i);
    seq.push_back(F::make_exists(store.aty(), env.name_of(v), env.sort_of(v)));
    seq.push_back(impl::deinterpret_var_store(store, v, store[i], env, allocator));
  }
  return F::make_nary(AND, std::move(seq), store.aty());
}

template<class I, class Env, class U, class Alloc, class Allocator2 = typename Env::allocator_type>
CUDA NI TFormula<Allocator2> deinterpret_in(const VStore<U, Alloc>& store, const I& intermediate, const Env& env, const Allocator2& allocator = Allocator2()) {
  using F = TFormula<Allocator2>;
  if(intermediate.size() == 0) {
    return F::make_true();
  }
  else if(intermediate.size() == 1) {
    return impl::deinterpret_var_store(store, intermediate[0].avar, intermediate[0].dom, env, allocator);
  }
  else {
    typename F::Sequence seq{allocator};
    for(int i = 0; i < intermediate.size(); ++i) {
      seq.push_back(impl::deinterpret_var_store(store, intermediate[i].avar, intermediate[i].dom, env, allocator));
    }
    return F::make_nary(AND, std::move(seq), store.aty());
  }
}

/* --- PIR --- */

namespace impl {
  /** We interpret the formula `f` in `intermediate`; only one constraint is added to
   * `intermediate` if the interpretation succeeds. */
  template <IKind kind, bool diagnose, class F, class Env, class Intermediate, class A, class Alloc>
  CUDA bool interpret_formula_pir(const PIR<A, Alloc>& pir, const F& f, Env& env, Intermediate& intermediate, IDiagnostics& diagnostics) {
    const char* name = PIR<A, Alloc>::name;
    if(f.type() != pir.aty() && !f.is_untyped()) {
      RETURN_INTERPRETATION_ERROR("The type of the formula does not match the type of this abstract domain.");
    }
    if(f.is_binary()) {
      Sig sig = f.sig();
      // Expect constraint of the form X = Y <OP> Z, or Y <OP> Z = X.
      int left = f.seq(0).is_binary() ? 1 : 0;
      int right = f.seq(1).is_binary() ? 1 : 0;
      if((sig == EQ || sig == EQUIV)  && (left + right == 1)) {
        auto& X = f.seq(left);
        auto& Y = f.seq(right).seq(0);
        auto& Z = f.seq(right).seq(1);
        bytecode_type bytecode;
        bytecode.op = f.seq(right).sig();
        if(X.is_variable() && Y.is_variable() && Z.is_variable() &&
          (bytecode.op == ADD || bytecode.op == MUL || ::lala::is_z_division(bytecode.op)
          || bytecode.op == MIN || bytecode.op == MAX
          || bytecode.op == EQ || bytecode.op == LEQ))
        {
          if( interpret_in<diagnose>(X, env, bytecode.x, diagnostics)
           && interpret_in<diagnose>(Y, env, bytecode.y, diagnostics)
           && interpret_in<diagnose>(Z, env, bytecode.z, diagnostics))
          {
            intermediate.bytecodes.push_back(bytecode);
            return true;
          }
          RETURN_INTERPRETATION_ERROR("Could not interpret the variables in the environment.");
        }
      }
    }
    RETURN_INTERPRETATION_ERROR("The shape of this formula is not supported.");
  }
}

/** PIR expects a non-conjunctive formula \f$ c \f$ which can either be interpreted in the
 * sub-domain `A` or in the current domain. */
template <IKind kind, bool diagnose = false, class F, class Env, class I, class A, class Alloc>
CUDA NI bool interpret_in(const PIR<A, Alloc>& pir, const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) {
  const char* name = PIR<A, Alloc>::name;
  size_t error_context = 0;
  if constexpr(diagnose) {
    diagnostics.add_suberror(IDiagnostics(false, name, "Uninterpretable formula in both PIR and its sub-domain.", f));
    error_context = diagnostics.num_suberrors();
  }
  bool res = false;
  AType current = f.type();
  const_cast<F&>(f).type_as(pir.subdomain()->aty()); // We will restore the type after the call to the sub-domain.
  if(interpret_in<kind, diagnose>(*pir.subdomain(), f, env, intermediate.sub_value, diagnostics)) {
    res = true;
  }
  const_cast<F&>(f).type_as(current);
  if(!res) {
    res = impl::interpret_formula_pir<kind, diagnose>(pir, f, env, intermediate, diagnostics);
  }
  if constexpr(diagnose) {
    diagnostics.merge(res, error_context);
  }
  return res;
}

template <bool diagnose = false, class F, class Env, class I, class A, class Alloc>
CUDA NI bool interpret_tell_in(const PIR<A, Alloc>& pir, const F& f, Env& env, I& tell, IDiagnostics& diagnostics) {
  return interpret_in<IKind::TELL, diagnose>(pir, f, env, tell, diagnostics);
}

template <bool diagnose = false, class F, class Env, class I, class A, class Alloc>
CUDA NI bool interpret_ask_in(const PIR<A, Alloc>& pir, const F& f, const Env& env, I& ask, IDiagnostics& diagnostics) {
  return interpret_in<IKind::ASK, diagnose>(pir, f, const_cast<Env&>(env), ask, diagnostics);
}

namespace impl {
  template<class Env, class Allocator2, class A, class Alloc>
  CUDA NI TFormula<Allocator2> deinterpret_bytecode(const PIR<A, Alloc>& pir, bytecode_type bytecode, const Env& env, Allocator2 allocator) {
    using F = TFormula<Allocator2>;
    auto X = F::make_lvar(bytecode.x.aty(), LVar<Allocator2>(env.name_of(bytecode.x), allocator));
    auto Y = F::make_lvar(bytecode.y.aty(), LVar<Allocator2>(env.name_of(bytecode.y), allocator));
    auto Z = F::make_lvar(bytecode.z.aty(), LVar<Allocator2>(env.name_of(bytecode.z), allocator));
    return F::make_binary(X, EQ, F::make_binary(Y, bytecode.op, Z, pir.aty(), allocator), pir.aty(), allocator);
  }
}

template<class Env, class A, class Alloc, class Allocator2 = typename Env::allocator_type>
CUDA NI TFormula<Allocator2> deinterpret_in(const PIR<A, Alloc>& pir, const Env& env, bool remove_entailed, size_t& num_entailed, Allocator2 allocator = Allocator2()) {
  using F = TFormula<Allocator2>;
  typename F::Sequence seq{allocator};
  seq.push_back(deinterpret_in(*pir.subdomain(), env, allocator));
  for(int i = 0; i < pir.num_deductions(); ++i) {
    if(remove_entailed && pir.ask(i)) {
      num_entailed++;
      continue;
    }
    seq.push_back(impl::deinterpret_bytecode(pir, pir.load_deduce(i), env, allocator));
  }
  return F::make_nary(AND, std::move(seq), pir.aty());
}

template<class Env, class A, class Alloc, class Allocator2 = typename Env::allocator_type>
CUDA NI TFormula<Allocator2> deinterpret_in(const PIR<A, Alloc>& pir, const Env& env, Allocator2 allocator = Allocator2()) {
  size_t num_entailed = 0;
  return deinterpret_in(pir, env, false, num_entailed, allocator);
}

template<class I, class Env, class A, class Alloc, class Allocator2 = typename Env::allocator_type>
CUDA NI TFormula<Allocator2> deinterpret_in(const PIR<A, Alloc>& pir, const I& intermediate, const Env& env, Allocator2 allocator = Allocator2()) {
  using F = TFormula<Allocator2>;
  typename F::Sequence seq{allocator};
  seq.push_back(deinterpret_in(*pir.subdomain(), intermediate.sub_value, env, allocator));
  for(int i = 0; i < intermediate.bytecodes.size(); ++i) {
    seq.push_back(impl::deinterpret_bytecode(pir, intermediate.bytecodes[i], env, allocator));
  }
  return F::make_nary(AND, std::move(seq), pir.aty());
}

} // namespace lala


/* ------------------------------------------------------------------------------------------- *
 * 7. Constraint network: the top-level dispatch, specific to Turbo's input language
 * ------------------------------------------------------------------------------------------- */

/** The diagnostics macros (`RETURN_INTERPRETATION_ERROR`, ...) name `IDiagnostics` unqualified.
 * We only import that one name: a `using namespace lala` at this scope would make the `Sig`
 * enumerators (`LT`, `GT`, `EQ`, `IN`, ...) ambiguous with the XCSP3 parser's own constants. */
using lala::IDiagnostics;

/** The result of interpreting a constraint network: the constraints to be told to the
 * propagators domain `IProp`, the objective and the search strategies. */
template <class IProp, class Alloc>
struct interpreted_cn {
  using allocator_type = Alloc;

  typename IProp::template tell_type<Alloc> constraints;
  Objective objective;
  SearchStrategies<Alloc> strategies;

  CUDA interpreted_cn(const Alloc& alloc = Alloc{})
   : constraints(alloc), strategies(alloc)
  {}

  interpreted_cn(const interpreted_cn&) = default;
  interpreted_cn(interpreted_cn&&) = default;
  interpreted_cn& operator=(const interpreted_cn&) = default;
  interpreted_cn& operator=(interpreted_cn&&) = default;

  CUDA allocator_type get_allocator() const {
    return strategies.get_allocator();
  }
};

/** Interpret `minimize(x)` or `maximize(x)` in `objective`.
 * An objective already fixed to a constant is ignored (warning), as it makes the problem a
 * satisfaction problem. */
template <bool diagnose = false, class F, class Env>
CUDA NI bool interpret_objective(const F& f, Env& env, Objective& objective, IDiagnostics& diagnostics) {
  const char* name = "Objective";
  assert(f.is(F::Seq) && (f.sig() == lala::MINIMIZE || f.sig() == lala::MAXIMIZE));
  if(f.seq(0).is_variable()) {
    if(!objective.is_satisfaction()) {
      RETURN_INTERPRETATION_ERROR("Multi-objective optimization is not supported.");
    }
    lala::AVar x;
    if(!interpret_in<diagnose>(f.seq(0), env, x, diagnostics)) {
      return false;
    }
    objective = Objective(x, f.sig() == lala::MINIMIZE);
    return true;
  }
  // If the objective variable is already fixed to a constant, we ignore this predicate.
  // If there is only one objective, it becomes a satisfaction problem.
  else if(lala::num_vars(f.seq(0)) == 0) {
    RETURN_INTERPRETATION_WARNING("This objective is already fixed to a constant, thus it is ignored.");
  }
  else {
    RETURN_INTERPRETATION_ERROR("Optimization predicates expect a variable to optimize (not an expression). Instead, you can create a new variable with the expression to optimize.");
  }
}

/** Interpret a predicate of the form `search(VariableOrder, ValueOrder, x_1, x_2, ..., x_n)` in `strat`. */
template <bool diagnose = false, class F, class Env, class Alloc>
CUDA NI bool interpret_strategy(const F& f, Env& env, StrategyType<Alloc>& strat, IDiagnostics& diagnostics) {
  const char* name = "SearchStrategy";
  if(!(f.is(F::ESeq)
    && f.eseq().size() >= 2
    && f.esig() == "search"
    && f.eseq()[0].is(F::ESeq) && f.eseq()[0].eseq().size() == 0
    && f.eseq()[1].is(F::ESeq) && f.eseq()[1].eseq().size() == 0))
  {
    RETURN_INTERPRETATION_ERROR("A search strategy must be a predicate of the form `search(input_order, indomain_min, x1, ..., xN)`.");
  }
  const auto& var_order_str = f.eseq()[0].esig();
  const auto& val_order_str = f.eseq()[1].esig();
  if(var_order_str == "input_order") { strat.var_order = VariableOrder::INPUT_ORDER; }
  else if(var_order_str == "first_fail") { strat.var_order = VariableOrder::FIRST_FAIL; }
  else if(var_order_str == "anti_first_fail") { strat.var_order = VariableOrder::ANTI_FIRST_FAIL; }
  else if(var_order_str == "smallest") { strat.var_order = VariableOrder::SMALLEST; }
  else if(var_order_str == "largest") { strat.var_order = VariableOrder::LARGEST; }
  else if(var_order_str == "random") { strat.var_order = VariableOrder::RANDOM; }
  else {
    RETURN_INTERPRETATION_ERROR("This variable order strategy is unsupported.");
  }
  if(val_order_str == "indomain_min") { strat.val_order = ValueOrder::MIN; }
  else if(val_order_str == "indomain_max") { strat.val_order = ValueOrder::MAX; }
  else if(val_order_str == "indomain_median") {
    printf("WARNING: indomain_median is not supported since we use interval domain. It is replaced by SPLIT");
    strat.val_order = ValueOrder::SPLIT;
  }
  else if(val_order_str == "indomain_split") { strat.val_order = ValueOrder::SPLIT; }
  else if(val_order_str == "indomain_reverse_split") { strat.val_order = ValueOrder::REVERSE_SPLIT; }
  else {
    RETURN_INTERPRETATION_ERROR("This value order strategy is unsupported.");
  }
  for(int i = 2; i < f.eseq().size(); ++i) {
    if(f.eseq(i).is(F::LV)) {
      strat.vars.push_back(lala::AVar{});
      if(!interpret_in<diagnose>(f.eseq(i), env, strat.vars.back(), diagnostics)) {
        return false;
      }
    }
    else if(f.eseq(i).is(F::V)) {
      strat.vars.push_back(f.eseq(i).v());
    }
    else if(lala::num_vars(f.eseq(i)) > 0) {
      RETURN_INTERPRETATION_ERROR("The predicate `search` only supports variables or constants, but an expression containing a variable was passed to it.");
    }
    // Ignore constant expressions.
    else {}
  }
  return true;
}

/** Route each conjunct of the constraint network to the element interpreting it.
 * `true` is interpreted exactly since all the elements involved preserve the top element. */
template <bool diagnose = false, class IProp, class F, class Env, class Alloc>
CUDA NI bool interpret_cn_in(const IProp& iprop, const F& f, Env& env,
  interpreted_cn<IProp, Alloc>& intermediate, IDiagnostics& diagnostics)
{
  if(f.is_true()) {
    return true;
  }
  else if(f.is(F::Seq) && f.sig() == lala::AND) {
    for(int i = 0; i < f.seq().size(); ++i) {
      if(!interpret_cn_in<diagnose>(iprop, f.seq(i), env, intermediate, diagnostics)) {
        return false;
      }
    }
    return true;
  }
  else if(f.is(F::Seq) && (f.sig() == lala::MINIMIZE || f.sig() == lala::MAXIMIZE)) {
    return interpret_objective<diagnose>(f, env, intermediate.objective, diagnostics);
  }
  else if(f.is(F::ESeq) && f.esig() == "search") {
    StrategyType<Alloc> strat(intermediate.get_allocator());
    if(!interpret_strategy<diagnose>(f, env, strat, diagnostics)) {
      return false;
    }
    intermediate.strategies.push_back(std::move(strat));
    return true;
  }
  // Any other formula is a constraint, interpreted in the propagators domain (and, for the
  // formulas it cannot represent, in its underlying store of variables).
  return lala::ginterpret_in<lala::IKind::TELL, diagnose>(iprop, f, env, intermediate.constraints, diagnostics);
}

/** Interpret the constraint network `f` and, on success, deduce the constraints in `iprop` and
 * store the objective and the search strategies in `objective` and `strategies`.
 * On failure, `env`, `iprop`, `objective` and `strategies` are left unchanged.
 * `TellAlloc` is the allocator of the intermediate representation, which only lives for the
 * duration of the interpretation. */
template <bool diagnose = false, class TellAlloc = battery::standard_allocator,
  class IProp, class F, class Env, class Alloc>
CUDA NI bool interpret_and_tell_cn(IProp& iprop, const F& f, Env& env,
  Objective& objective, SearchStrategies<Alloc>& strategies, IDiagnostics& diagnostics,
  TellAlloc tell_alloc = TellAlloc{})
{
  auto snap = env.snapshot();
  interpreted_cn<IProp, TellAlloc> intermediate(tell_alloc);
  intermediate.objective = objective;
  if(!interpret_cn_in<diagnose>(iprop, f, env, intermediate, diagnostics)) {
    env.restore(snap);
    return false;
  }
  iprop.deduce(intermediate.constraints);
  objective = intermediate.objective;
  for(int i = 0; i < intermediate.strategies.size(); ++i) {
    strategies.push_back(StrategyType<Alloc>(intermediate.strategies[i], strategies.get_allocator()));
  }
  return true;
}

#endif
