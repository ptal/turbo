// Copyright 2023 Pierre Talbot

#ifndef TURBO_SIMPLIFIER_HPP
#define TURBO_SIMPLIFIER_HPP

#include "lala/logic/logic.hpp"
#include "lala/universes/arith_bound.hpp"
#include "lala/abstract_deps.hpp"
#include "battery/dynamic_bitset.hpp"

#include "env.hpp"
#include "interpretation.hpp"

namespace lala {

struct SimplifierStats {
  int current_iteration;
  std::vector<size_t> eliminated_equality_constraints_;
  std::vector<size_t> eliminated_constraints_by_as_;
  std::vector<size_t> eliminated_entailed_constraints_;
  std::vector<std::vector<size_t>> eliminated_constraints_by_icse_;
  std::vector<size_t> eliminated_useless_variables_;

  SimplifierStats()
    : current_iteration(-1)
  {}

  void prepare_next_iteration() {
    current_iteration++;
    eliminated_equality_constraints_.push_back(0);
    eliminated_constraints_by_as_.push_back(0);
    eliminated_entailed_constraints_.push_back(0);
    eliminated_constraints_by_icse_.emplace_back();
    eliminated_useless_variables_.push_back(0);
  }

  auto& eliminated_equality_constraints() { return eliminated_equality_constraints_[current_iteration]; }
  auto& eliminated_constraints_by_as() { return eliminated_constraints_by_as_[current_iteration]; }
  auto& eliminated_entailed_constraints() { return eliminated_entailed_constraints_[current_iteration]; }
  auto& eliminated_constraints_by_icse() { return eliminated_constraints_by_icse_[current_iteration]; }
  auto& eliminated_useless_variables() { return eliminated_useless_variables_[current_iteration]; }
};

/** This abstract domain works at the level of logical formulas.
 * It deduces the formula by performing a number of simplifications w.r.t. an underlying abstract domain including:
 *  1. Removing assigned variables.
 *  2. Removing unused variables.
 *  3. Removing entailed formulas.
 *  4. Removing variable equality by tracking equivalence classes.
 *
 * The simplified formula can be obtained by calling `deinterpret()`.
 * Given a solution to the simplified formula, the value of the variables deleted can be obtained by calling `print_variable()`.
 */
template<class A, class Allocator>
class Simplifier {
public:
  using allocator_type = Allocator;
  using sub_type = A;
  using sub_allocator_type = typename sub_type::allocator_type;
  using universe_type = typename sub_type::universe_type;
  using memory_type = typename universe_type::memory_type;
  using this_type = Simplifier<sub_type, allocator_type>;

  constexpr static const bool is_abstract_universe = false;
  constexpr static const bool sequential = universe_type::sequential;
  constexpr static const bool is_totally_ordered = false;
  // Note that I did not define the concretization function formally yet... This is not yet an fully fledged abstract domain, we need to work more on it!
  constexpr static const bool preserve_bot = true;
  constexpr static const bool preserve_top = true;
  constexpr static const bool preserve_join = true;
  constexpr static const bool preserve_meet = true;
  constexpr static const bool injective_concretization = true;
  constexpr static const bool preserve_concrete_covers = true;
  constexpr static const char* name = "Simplifier";

  template<class A2, class Alloc2>
  friend class Simplifier;

  using formula_sequence = battery::vector<TFormula<allocator_type>, allocator_type>;

private:
  AType atype;
  AType store_aty;
  abstract_ptr<sub_type> sub;
  // We keep a copy of the variable environment in which the formula has been initially interpreted.
  // This is necessary to project the variables and ask constraints in the subdomain during deduction.
  VarEnv<allocator_type> env;
  // Read-only conjunctive formula, where each is treated independently.
  formula_sequence formulas;
  // Write-only (accessed in only 1 thread because this is not a parallel lattice entity) conjunctive formula, the main operation is a map between formulas and simplified_formulas.
  formula_sequence simplified_formulas;
  // eliminated_variables[i] is `true` when the variable `i` can be removed because it does not occur in any constraint.
  battery::dynamic_bitset<memory_type, allocator_type> eliminated_variables;
  // eliminated_formulas[i] is `true` when the formula `i` is entailed.
  battery::dynamic_bitset<memory_type, allocator_type> eliminated_formulas;
  // `equivalence_classes[i]` contains the index of the representative variable in the equivalence class of the variable `i`.
  battery::vector<ZUB<int, memory_type>, allocator_type> equivalence_classes;
  // `constants[i]` contains the universe value of the representative variables `i`, aggregated by meet on the values of all variables in the equivalence class.
  battery::vector<universe_type, allocator_type> constants;

public:
  CUDA Simplifier(AType atype
    , AType store_aty
    , abstract_ptr<sub_type> sub
    , const allocator_type& alloc = allocator_type())
   : atype(atype), store_aty(store_aty), sub(sub), env(alloc)
   , formulas(alloc), simplified_formulas(alloc)
   , eliminated_variables(alloc), eliminated_formulas(alloc)
   , equivalence_classes(alloc), constants(alloc)
  {}

  CUDA Simplifier(this_type&& other)
    : atype(other.atype), store_aty(other.store_aty), sub(std::move(other.sub)), env(other.env)
    , formulas(std::move(other.formulas)), simplified_formulas(std::move(other.simplified_formulas))
    , eliminated_variables(std::move(other.eliminated_variables)), eliminated_formulas(std::move(other.eliminated_formulas))
    , equivalence_classes(std::move(other.equivalence_classes)), constants(std::move(other.constants))
  {}

  struct light_copy_tag {};

  // This return a light copy of `other`, basically just keeping the equivalence classes and the environment to be able to print solutions and call `representative`.
  template<class A2, class Alloc2>
  CUDA Simplifier(const Simplifier<A2, Alloc2>& other, light_copy_tag tag, abstract_ptr<sub_type> sub, const allocator_type& alloc = allocator_type())
   : atype(other.atype)
   , store_aty(other.store_aty)
   , sub(sub)
   , env(other.env, alloc)
   , equivalence_classes(other.equivalence_classes, alloc)
   , constants(other.constants, alloc)
  {}

  CUDA allocator_type get_allocator() const {
    return formulas.get_allocator();
  }

  CUDA AType aty() const {
    return atype;
  }

  /** @parallel @order-preserving @increasing  */
  CUDA local::B is_bot() const {
    return sub->is_bot();
  }

  /** Returns the number of variables currently represented by this abstract element. */
  CUDA size_t vars() const {
    return equivalence_classes.size();
  }

  template <class Alloc>
  struct tell_type {
    int num_vars;
    formula_sequence formulas;
    VarEnv<Alloc>* env;
    tell_type(const Alloc& alloc = Alloc())
      : num_vars(0), formulas(alloc), env(nullptr)
    {}
  };

public:

  CUDA void initialize(int num_vars, int num_cons) {
    eliminated_variables.resize(num_vars);
    eliminated_variables.reset();
    eliminated_formulas.resize(num_cons);
    eliminated_formulas.reset();
    constants.resize(num_vars);
    for(int i = 0; i < constants.size(); ++i) {
      constants[i].join_top();
    }
    equivalence_classes.resize(num_vars);
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      equivalence_classes[i] = i;
    }
  }

  /** We initialize the equivalence classes and var/cons elimination masks.
   * Further, we eliminate all constraints in `tnf` that are not in TNF.
   * (It is the existential quantifiers and unary constraints that are re-generated from the underlying store later.)
   */
  template <class Seq>
  CUDA void initialize_tnf(int num_vars, const Seq& tnf) {
    initialize(num_vars, tnf.size());
    int z = 0;
    for(int i = 0; i < tnf.size(); ++i) {
      if(!is_tnf(tnf[i])) {
        ++z;
        eliminate(eliminated_formulas, i);
      }
    }
  }

public:
  /** @sequential */
  template <class Alloc2>
  CUDA bool deduce(tell_type<Alloc2>&& t) {
    if(t.env != nullptr) { // could be nullptr if the interpreted formula is true.
      env = *(t.env);
      initialize(t.num_vars, t.formulas.size());
      formulas = std::move(t.formulas);
      simplified_formulas.resize(formulas.size());
      return true;
    }
    return false;
  }

  // `f` must be a formula from `formulas`.
  CUDA AVar var_of(const TFormula<allocator_type>& f) const {
    using F = TFormula<allocator_type>;
    if(f.is(F::LV)) {
      assert(env.variable_of(f.lv()).has_value());
      assert(env.variable_of(f.lv())->get().avar_of(store_aty).has_value());
      return env.variable_of(f.lv())->get().avar_of(store_aty).value();
    }
    else {
      assert(f.is(F::V));
      assert(env[f.v()].avar_of(store_aty).has_value());
      return env[f.v()].avar_of(store_aty).value();
    }
  }

public:
  /** Print the abstract universe of `vname` taking into account simplifications (representative variable and constant).
  */
  template <class Alloc, class Abs, class Env>
  CUDA void print_variable(const LVar<Alloc>& vname, const Env& benv, const Abs& b) const {
    assert(env.variable_of(vname).has_value());
    const auto& local_var = env.variable_of(vname)->get();
    assert(local_var.avar_of(store_aty).has_value());
    int rep = equivalence_classes[local_var.avar_of(store_aty)->vid()];
    const auto& rep_name = env.name_of(AVar{store_aty, rep});
    auto benv_variable = benv.variable_of(rep_name);
    if(benv_variable.has_value()) {
      benv_variable->get().sort.print_value(b.project(benv_variable->get().avars[0]));
    }
    else {
      local_var.sort.print_value(constants[rep]);
    }
  }

private:
  /** \return `true` if mask[i] was changed. */
  CUDA local::B eliminate(battery::dynamic_bitset<memory_type, allocator_type>& mask, size_t i) {
    if(!mask.test(i)) {
      mask.set(i, true);
      return true;
    }
    return false;
  }

  CUDA local::B eliminate(battery::dynamic_bitset<memory_type, allocator_type>& mask, size_t i, size_t& eliminated_constraints) {
    if(eliminate(mask, i)) {
      ++eliminated_constraints;
      return true;
    }
    return false;
  }

  // We merge the domains of the variables in the same equivalence class.
  CUDA local::B vdeduce(int i) {
    const auto& u = sub->project(AVar{store_aty, i});
    size_t j = find(i);
    local::B has_changed = constants[j].meet(u);
    // Note that this code is only useful for simplifying formula not in TCN (such as tests in simplifier_test.cpp).
    // On TCN, the method `eliminate_useless_variables` does the job.
    if(!constants[j].is_bot() && constants[j].lb().value() == constants[j].ub().value()) {
      eliminate(eliminated_variables, j);
    }
    return has_changed;
  }

public:
  CUDA local::B cons_deduce(int i) {
    using F = TFormula<allocator_type>;
    local::B has_changed = false;
    // Eliminate constraint of the form x = y, and add x,y in the same equivalence class.
    if(is_var_equality(formulas[i])) {
      size_t s = 0;
      return replace_by_equivalence(var_of(formulas[i].seq(0)), var_of(formulas[i].seq(1)), i, s);
    }
    else {
      // Eliminate entailed formulas.
      IDiagnostics diagnostics;
      typename sub_type::template ask_type<allocator_type> ask;
      if(interpret_ask_in(*sub, formulas[i], env, ask, diagnostics))
      {
        if(sub->ask(ask)) {
          return eliminate(eliminated_formulas, i);
        }
      }
      // Replace assigned variables by constants.
      // Note that since everything is in a fixed point loop, both the constant and the equivalence class might be updated later on.
      // This is one of the reasons we cannot update `formulas` in-place: we would not be able to update the constant a second time (since the variable would be eliminated).
      auto f = formulas[i].map([&](const F& f, const F& parent) {
        if(f.is_variable()) {
          AVar x = var_of(f);
          size_t j = find(x.vid());
          if(!constants[j].is_bot() && constants[j].lb().value() == constants[j].ub().value()) {
            auto k = deinterpret_constant<F>(constants[j]);
            if(env[x].sort.is_bool() && k.is(F::Z) && parent.is_logical()) {
              return k.z() == 0 ? F::make_false() : F::make_true();
            }
            return std::move(k);
          }
          else if(equivalence_classes[x.vid()] != x.vid()) {
            return F::make_lvar(UNTYPED, env.name_of(AVar{store_aty, equivalence_classes[x.vid()]}));
          }
          return f.map_atype(UNTYPED);
        }
        return f;
      });
      f = eval(f);
      if(f.is_true()) {
        return eliminate(eliminated_formulas, i);
      }
      if(f != simplified_formulas[i]) {
        simplified_formulas[i] = f;
        return true;
      }
      return false;
    }
  }

  /** We have one deduction operator per variable and one per constraint in the interpreted formula. */
  CUDA size_t num_deductions() const {
    return constants.size() + formulas.size();
  }

  CUDA local::B deduce(size_t i) {
    assert(i < num_deductions());
    if(i < constants.size()) {
      return vdeduce(i);
    }
    else {
      return cons_deduce(i - constants.size());
    }
  }

  template <class Env>
  CUDA void init_env(const Env& env) {
    this->env = env;
  }

private:
  CUDA local::B replace_by_equivalence(AVar x, AVar y, int i, size_t& eliminated_constraints) {
    return replace_by_equivalence(x.vid(), y.vid(), i, eliminated_constraints);
  }

  CUDA local::B replace_by_equivalence(int x, int y, int i, size_t& eliminated_constraints) {
    merge(x, y);
    return eliminate(eliminated_formulas, i, eliminated_constraints);
  }

public:
  /** I-CSE algorithm.
   * For each pair of TNF constraints `x = y op z` and `x' = y' op' z'`, whenever `[y'] = [y]`, `op = op'` and `[z] = [z']`, we add the equivalence `x = x'` and eliminate the second constraint.
   * Note that [x] represents the equivalence class of `x`.
   * To avoid an algorithm running in O(n^2), we use a hash map to detect syntactical equivalence between `y op z` and `y' op z'`.
   * Further, for commutative operators, we redefine the equality function.
   *
   * This algorithm is applied until a fixpoint is reached.
   * \return If any change has been made.
   */
  template <class Seq>
  CUDA bool i_cse(const Seq& tnf, SimplifierStats& stats) {
    auto hash = [](const std::tuple<int,Sig,int> &right_tnf) {
      return static_cast<size_t>(std::get<0>(right_tnf))
           * static_cast<size_t>(std::get<1>(right_tnf))
           * static_cast<size_t>(std::get<2>(right_tnf));
    };
    // This equality function also checks for commutative operators (in which case the hash will also be the same).
    auto equal = [](const std::tuple<int,Sig,int> &l, const std::tuple<int,Sig,int> &r){
      return std::get<1>(l) == std::get<1>(r)
        && ((std::get<0>(l) == std::get<0>(r) && std::get<2>(l) == std::get<2>(r))
         || (is_commutative(std::get<1>(l)) && std::get<0>(l) == std::get<2>(r) && std::get<2>(l) == std::get<0>(r)));
    };
    std::unordered_map<std::tuple<int,Sig,int>, int, decltype(hash), decltype(equal)> cs(tnf.size(), hash, equal);
    bool has_changed = false;
    bool local_has_changed = true;
    while(local_has_changed) {
      size_t eliminated_constraints_by_icse = 0;
      local_has_changed = false;
      cs.clear();
      for(int i = 0; i < tnf.size(); ++i) {
        if(!eliminated_formulas.test(i)) {
          int x = find(var_of(tnf[i].seq(0)).vid());
          int y = find(var_of(tnf[i].seq(1).seq(0)).vid());
          int z = find(var_of(tnf[i].seq(1).seq(1)).vid());
          Sig op = tnf[i].seq(1).sig();
          auto p = cs.insert(std::make_pair(std::make_tuple(y, op, z), x));
          if(!p.second) { // `p.second` is false if we detect a collision.
            local_has_changed |= replace_by_equivalence(x, p.first->second, i, eliminated_constraints_by_icse);
            if(local_has_changed) {
              has_changed = true;
            }
          }
        }
      }
      stats.eliminated_constraints_by_icse().push_back(eliminated_constraints_by_icse);
    }
    return has_changed;
  }

  /** Perform algebraic simplification on the TNF.
   * The non-eliminated constraints are assumed to be in TNF.
   */
  template <class Seq>
  CUDA bool algebraic_simplify(Seq& tnf, SimplifierStats& stats) {
    using F = typename Seq::value_type;
    constexpr universe_type ZERO(0,0);
    constexpr universe_type ONE(1,1);
    auto& vstore = *sub;
    size_t elim_cons = stats.eliminated_constraints_by_as();
    size_t elim_eq = stats.eliminated_equality_constraints();
    bool has_changed = false;
    for(int i = 0; i < tnf.size(); ++i) {
      if(!eliminated_formulas.test(i)) {
        int x = find(var_of(tnf[i].seq(0)).vid());
        int y = find(var_of(tnf[i].seq(1).seq(0)).vid());
        int z = find(var_of(tnf[i].seq(1).seq(1)).vid());
        Sig sig = tnf[i].seq(1).sig();
        bool x_is_c = vstore[x].lb() == vstore[x].ub();
        bool y_is_c = vstore[y].lb() == vstore[y].ub();
        bool z_is_c = vstore[z].lb() == vstore[z].ub();
        /** Put constants on the right side of the operator. */
        if(is_commutative(sig) && y_is_c) {
          std::swap(y, z);
        }
        switch(sig) {
          case ADD: {
            /** x = x + z -> z = 0 and  x = y + x -> y = 0 */
            if(x == y || x == z) {
              int y2 = x == y ? z : y;
              vstore[y2].meet(ZERO);
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            /** x = y + 0 -> x = y */
            else if(vstore[z] == ZERO) {
              replace_by_equivalence(x, y, i, stats.eliminated_constraints_by_as());
            }
            /** x = y + y -> x = y * 2 */
            else if(y == z) {
              tnf[i].seq(1) = F::make_binary(
                F::make_lvar(UNTYPED, env.name_of(AVar{store_aty, y})),
                MUL,
                F::make_lvar(UNTYPED, LVar<allocator_type>("__CONSTANT_2")));
            }
            break;
          }
          case MUL: {
            /** x = x * k -> x = 0 (if k != 1), true otherwise. */
            if(x == y && z_is_c) {
              if(vstore[z] != ONE) {
                has_changed |= vstore[x].meet(ZERO);
              }
              else { /* true */ }
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            /** k = y * y -> y \in [-n,n] (if n * n = k), false (otherwise).
             * This is an over-approximation, thus we cannot eliminate the constraint. */
            else if(x_is_c && y == z) {
              auto n = battery::iroots_up(vstore[x].lb().value(), 2);
              if(n * n == vstore[x].lb()) {
                has_changed |= vstore[y].meet(universe_type(-n, n));
              }
              else {
                vstore[y].meet_bot(); // false because k is not a perfect square.
              }
            }
            /** x = y * 1 */
            else if(vstore[z] == ONE) {
              replace_by_equivalence(x, y, i, stats.eliminated_constraints_by_as());
            }
            /** x = x * x */
            else if(x == y && y == z) {
              vstore[x].meet(universe_type(0,1));
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            break;
          }
          case EDIV: {
            /** x = 1/x -> x \in {-1,1} (over-approximated so the constraint cannot not eliminated) */
            if(vstore[y] == ONE && x == z) {
              has_changed |= vstore[x].meet(universe_type(-1,1));
            }
            else if(vstore[y] == ZERO && x == z) {
              vstore[x].meet_bot();
            }
            else if(x_is_c && y == z) {
              if(vstore[x] != ONE) {
                vstore[y].meet_bot();
              }
              // Cannot eliminate the constraint as we must take into account that y != 0.
            }
            else if(vstore[z] == ONE) {
              replace_by_equivalence(x, y, i, stats.eliminated_constraints_by_as());
            }
            else if(x == y && y == z) {
              vstore[x].meet(ONE);
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            break;
          }
          case EMOD: {
            /** x = x mod x -> x = 0 */
            if(x == y && y == z) {
              vstore[x].meet(ZERO);
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            /** x = x mod k -> x in [0, abs(k) - 1] */
            else if(x == y && z_is_c) {
              vstore[x].meet(universe_type(0, std::abs(vstore[z].lb()) - 1));
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            /** x = k mod x is always false. */
            else if(x == z && y_is_c) {
              vstore[x].meet_bot();
            }
            /** 0 = x mod x is always true. */
            else if(y == z && vstore[x] == ZERO) {
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            break;
          }
          case MIN:
          case MAX: {
            /** x = min/max(y, y) -> x = y */
            if(y == z) {
              replace_by_equivalence(x, y, i, stats.eliminated_constraints_by_as());
            }
            /** x = min(x, y) -> 1 = (x <= y)  */
            /** x = max(x, y) -> 1 = (y <= x)  */
            else if(x == y || x == z) {
              int y2 = x == y ? z : y;
              int x2 = x;
              if(sig == MAX) {
                std::swap(x2, y2);
              }
              tnf[i].seq(0) = F::make_lvar(UNTYPED, LVar<allocator_type>("__CONSTANT_1"));
              tnf[i].seq(1) = F::make_binary(
                F::make_lvar(UNTYPED, env.name_of(AVar{store_aty, x2})),
                LEQ,
                F::make_lvar(UNTYPED, env.name_of(AVar{store_aty, y2})));
            }
            break;
          }
          case EQUIV:
          case EQ: {
            if(vstore[x] == ONE) {
              replace_by_equivalence(y, z, i, stats.eliminated_equality_constraints());
            }
            /** x = (x = k) -> false (k = 0), x = 1 (k = 1) or x = 0 */
            else if(x == y && z_is_c) {
              if(vstore[z] == ZERO) {
                vstore[x].meet_bot();
              }
              else if(vstore[z] == ONE) {
                vstore[x].meet(ONE);
              }
              else {
                vstore[x].meet(ZERO);
              }
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            else if(y == z) {
              vstore[x].meet(ONE);
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            break;
          }
          case LEQ: {
            /** x = (x <= k) -> x = 0 (k < 0), x = 1 (k > 0), false (k = 0) */
            if(x == y && z_is_c) {
              int k = vstore[z].lb();
              if(k < 0) {
                vstore[x].meet(ZERO);
              }
              else if(k > 0) {
                vstore[x].meet(ONE);
              }
              else { /** no solution with k == 0 */
                vstore[x].meet_bot();
              }
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            /** x = (k <= x) -> x = 0 (k > 1), x = 1 (k <= 1), true (k = 1). */
            else if(x == z && y_is_c) {
              int k = vstore[y].lb();
              if(k > 1) {
                vstore[x].meet(ZERO);
              }
              else if(k < 1) {
                vstore[x].meet(ONE);
              }
              else { /** true whenever k = 1 */ }
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            else if(y == z) {
              vstore[x].meet(ONE);
              eliminate(eliminated_formulas, i, stats.eliminated_constraints_by_as());
            }
            break;
          }
          default:
            printf("Unsupported operator %s in TNF algebraic simplification.\n", string_of_sig(sig));
        }
      }
    }
    return has_changed || elim_cons != stats.eliminated_constraints_by_as() || elim_eq != stats.eliminated_equality_constraints();
  }

private:
  /** Find operation in union-find algorithm.
   * An additional invariant is that:
   *   forall x. store[x] >= store[find(x)]
   * That is, the root node contains the meet of all domains in the equivalence class.
  */
  CUDA int find(int x) {
    int root = x;
    while(equivalence_classes[root] != root) {
      root = equivalence_classes[root];
    }
    while(equivalence_classes[x] != root) {
      int parent = equivalence_classes[x];
      sub->embed(AVar{store_aty, parent}, (*sub)[x]);
      equivalence_classes[x] = root;
      x = parent;
    }
    return root;
  }

  /** A simple merge operation in union-find algorithm. */
  CUDA void merge(int x, int y) {
    int rx = find(x);
    int ry = find(y);
    if(rx != ry) {
      // Easier to debug and more robust for testing: use the root with the smallest index.
      if(rx < ry) battery::swap(rx, ry);
      equivalence_classes[rx] = ry;
      sub->embed(AVar{store_aty, ry}, (*sub)[rx]);
    }
  }

public:
  CUDA void meet_equivalence_classes() {
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      int root = find(i);
      sub->embed(AVar{store_aty, root}, (*sub)[i]);
    }
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      int root = find(i);
      sub->embed(AVar{store_aty, i}, (*sub)[root]);
    }
  }

  template <class B, class Seq>
  CUDA void eliminate_entailed_constraints(const B& b, const Seq& tnf, SimplifierStats& stats) {
    for(int i = 0; i < tnf.size(); ++i) {
      if(!is_tnf(tnf[i]) || eliminated_formulas.test(i)) {
        continue;
      }
      IDiagnostics diagnostics;
      typename sub_type::template ask_type<allocator_type> ask_value;
      bool ask_success = interpret_ask_in(b, tnf[i], env, ask_value, diagnostics);
      assert(ask_success);
      if(b.ask(ask_value)) {
        eliminate(eliminated_formulas, i, stats.eliminated_entailed_constraints());
      }
    }
  }

  template <class Seq>
  CUDA void eliminate_useless_variables(const Seq& tnf, SimplifierStats& stats) {
    /** Keep only the variables that are representative and occur in at least one TNF constraint. */
    eliminated_variables.set();
    for(int i = 0; i < tnf.size(); ++i) {
      if(!eliminated_formulas.test(i)) {
        eliminated_variables.set(find(var_of(tnf[i].seq(0)).vid()), false);
        eliminated_variables.set(find(var_of(tnf[i].seq(1).seq(0)).vid()), false);
        eliminated_variables.set(find(var_of(tnf[i].seq(1).seq(1)).vid()), false);
      }
    }
    // The bitset can contain more 1s than the number of variables, hence we adjust the value using `vars()`.
    stats.eliminated_useless_variables() = eliminated_variables.count() - (eliminated_variables.size() - vars());
    // To follow the other statistics, we only count the newly eliminated variables.
    for(int i = 0; i < stats.eliminated_useless_variables_.size() - 1; ++i) {
      stats.eliminated_useless_variables() -= stats.eliminated_useless_variables_[i];
    }
    /** Eliminated variables might still occur in the variables we need to print.
     * Therefore, we save them in `constants`. */
    for(int i = 0; i < sub->vars(); ++i) {
      int root = find(i);
      constants[i] = sub->project(AVar{store_aty, root});
    }
  }

private:
  template <class F>
  void substitute_var(F& f) const {
    if(f.is_variable()) {
      AVar x = var_of(f);
      // Note: `find` is non-const, and anyways, `eliminate_useless_variables` is called before, hence find(x) = equivalence_classes[x].
      int root = equivalence_classes[x.vid()];
      if(x.vid() != root) {
        x = AVar{store_aty, root};
        f = F::make_lvar(store_aty, env.name_of(x));
      }
      /** If the variable is eliminated, but still appear in a constraint at this stage, it means it's an "extra" constraint not in TNF, and therefore substitute the variable by its constant. */
      if(eliminated_variables.test(x.vid())) {
        auto k = deinterpret_constant<F>((*sub)[x.vid()]);
        if(env[x].sort.is_bool() && k.is(F::Z)) {
          f = k.z() == 0 ? F::make_false() : F::make_true();
        }
        else {
          f = std::move(k);
        }
      }
    }
  }

public:
  /** Given any formula (not necessarily in TNF), we substitute each variable with its representative variable or a constant if it got eliminated. */
  template <class F>
  CUDA void substitute(F& f) const {
    f.inplace_map([this](F& leaf, const F&) { substitute_var(leaf); });
  }

private:
  // Deinterpret the existential quantifiers (only one per equivalence classes), and the domain of each variable.
  template <class Seq>
  CUDA void deinterpret_vars(Seq& seq) {
    using F = TFormula<allocator_type>;
    for(int i = 0; i < equivalence_classes.size(); ++i) {
      if(equivalence_classes[i] == i && !eliminated_variables.test(i)) {
        const auto& x = env[AVar{store_aty, i}];
        seq.push_back(F::make_exists(UNTYPED, x.name, x.sort));
        auto domain_constraint = deinterpret_in(constants[i], AVar{store_aty, i}, env, get_allocator());
        map_avar_to_lvar(domain_constraint, env, true);
        seq.push_back(domain_constraint);
      }
    }
  }

  template <class Seq1, class Seq2>
  CUDA void deinterpret_constraints(Seq1& seq, const Seq2& formulas, bool enable_substitute = false) {
    // Deinterpret the simplified formulas.
    for(int i = 0; i < formulas.size(); ++i) {
      if(!eliminated_formulas.test(i)) {
        seq.push_back(formulas[i]);
        if(enable_substitute) {
          substitute(seq.back());
        }
      }
    }
  }

public:
  template <class Seq>
  CUDA NI TFormula<allocator_type> deinterpret(const Seq& source, bool substitute) {
    using F = TFormula<allocator_type>;
    typename F::Sequence seq(get_allocator());
    if(is_bot()) {
      return F::make_false();
    }
    if(eliminated_variables.all() || eliminated_formulas.all()) {
      return F::make_true();
    }
    deinterpret_vars(seq);
    deinterpret_constraints(seq, source, substitute);
    return seq.size() == 0 ? F::make_true() : F::make_nary(AND, std::move(seq));
  }

  CUDA NI TFormula<allocator_type> deinterpret() {
    return deinterpret(simplified_formulas, false);
  }
};

} // namespace lala

#endif
