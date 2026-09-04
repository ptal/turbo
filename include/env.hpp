// Copyright 2021 Pierre Talbot

/** The environment mapping logical variable names to abstract variables (`AVar`).
 * It lives in Turbo rather than in lala-core because it only exists to interpret logical
 * formulas: an abstract domain manipulates `AVar` and never needs the logical names.
 * The interpretation functions themselves are in `interpretation.hpp`. */

#ifndef TURBO_ENV_HPP
#define TURBO_ENV_HPP

#include "battery/utility.hpp"
#include "battery/vector.hpp"
#include "battery/string.hpp"
#include "battery/tuple.hpp"
#include "battery/variant.hpp"
#include "lala/logic/ast.hpp"
#include "lala/logic/algorithm.hpp"

#include <string>
#include <unordered_map>
#include <functional>

namespace lala {

template<class Allocator>
struct Variable {
  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  bstring name;
  Sort<Allocator> sort;
  bvector<AVar> avars;

  Variable(Variable<Allocator>&&) = default;
  Variable(const Variable<Allocator>&) = default;

  CUDA NI Variable(const bstring& name, const Sort<Allocator>& sort, AVar av, const Allocator& allocator = Allocator{})
    : name(name, allocator), sort(sort, allocator), avars(allocator)
  {
    avars.push_back(av);
  }

  template <class Alloc2>
  CUDA NI Variable(const Variable<Alloc2>& other, const Allocator& allocator = Allocator{})
    : name(other.name, allocator)
    , sort(other.sort, allocator)
    , avars(other.avars, allocator)
  {}

  CUDA NI std::optional<AVar> avar_of(AType aty) const {
    for(int i = 0; i < avars.size(); ++i) {
      if(avars[i].aty() == aty) {
        return avars[i];
      }
    }
    return {};
  }
};

template <class Allocator>
struct ListVarIndex {
  using allocator_type = Allocator;
  using this_type = ListVarIndex<Allocator>;
  using variable_type = Variable<Allocator>;

  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  bvector<variable_type>* lvars;

  CUDA ListVarIndex(bvector<variable_type>* lvars): lvars(lvars) {}
  CUDA ListVarIndex(this_type&&, bvector<variable_type>* lvars): lvars(lvars) {}
  CUDA ListVarIndex(const this_type&, bvector<variable_type>* lvars): lvars(lvars) {}

  template <class Alloc2>
  CUDA ListVarIndex(const ListVarIndex<Alloc2>&, bvector<variable_type>* lvars)
    : lvars(lvars)
  {}

  // For this operator=, we suppose `lvars` is updated before.
  CUDA this_type& operator=(this_type&& other) {
    return *this;
  }

  CUDA this_type& operator=(const this_type& other) {
    return *this;
  }

  CUDA std::optional<size_t> lvar_index_of(const char* lv) const {
    for(size_t i = 0; i < lvars->size(); ++i) {
      if((*lvars)[i].name == lv) {
        return i;
      }
    }
    return {};
  }

  CUDA void push_back(variable_type&& var) {
    lvars->push_back(std::move(var));
  }

  CUDA void erase(const char* lv) {}

  CUDA void set_lvars(bvector<variable_type>* lvars) {
    this->lvars = lvars;
  }
};

template <class Allocator>
struct HashMapVarIndex {
  using allocator_type = Allocator;
  using this_type = ListVarIndex<Allocator>;
  using variable_type = Variable<Allocator>;

  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  bvector<variable_type>* lvars;
  std::unordered_map<std::string, size_t> lvar_index;

  HashMapVarIndex(bvector<variable_type>* lvars): lvars(lvars) {
    for(size_t i = 0; i < lvars->size(); ++i) {
      lvar_index[std::string((*lvars)[i].name.data())] = i;
    }
  }

  HashMapVarIndex(this_type&& other, bvector<variable_type>* lvars)
   : lvars(lvars), lvar_index(std::move(other.lvar_index)) {}

  HashMapVarIndex(const this_type& other, bvector<variable_type>* lvars)
   : lvars(lvars), lvar_index(other.lvar_index) {}

  template <class Alloc2>
  HashMapVarIndex(const HashMapVarIndex<Alloc2>& other, bvector<variable_type>* lvars)
    : lvars(lvars), lvar_index(other.lvar_index)
  {}

  // For this operator=, we suppose `lvars` is updated before.
  this_type& operator=(this_type&& other) {
    lvar_index = std::move(other.lvar_index);
    return *this;
  }
  this_type& operator=(const this_type& other) {
    lvar_index = other.lvar_index;
    return *this;
  }

  std::optional<size_t> lvar_index_of(const char* lv) const {
    auto it = lvar_index.find(std::string(lv));
    if(it != lvar_index.end()) {
      return {it->second};
    }
    return {};
  }

  void push_back(variable_type&& var) {
    lvar_index[std::string(var.name.data())] = lvars->size();
    lvars->push_back(std::move(var));
  }

  void erase(const char* lv) {
    lvar_index.erase(std::string(lv));
  }

  void set_lvars(bvector<variable_type>* lvars) {
    this->lvars = lvars;
  }
};

template <class Allocator>
struct DispatchIndex {
  using allocator_type = Allocator;
  using this_type = ListVarIndex<Allocator>;
  using variable_type = Variable<Allocator>;

  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  battery::unique_ptr<HashMapVarIndex<allocator_type>, allocator_type> cpu_index;
  battery::unique_ptr<ListVarIndex<allocator_type>, allocator_type> gpu_index;

  CUDA DispatchIndex(bvector<variable_type>* lvars): cpu_index(nullptr), gpu_index(nullptr) {
    gpu_index = std::move(battery::allocate_unique<ListVarIndex<allocator_type>>(lvars->get_allocator(), lvars));
    #ifndef __CUDA_ARCH__
      cpu_index = std::move(battery::allocate_unique<HashMapVarIndex<allocator_type>>(lvars->get_allocator(), lvars));
    #endif
  }

  CUDA DispatchIndex(this_type&& other, bvector<variable_type>* lvars)
   : gpu_index(std::move(other.gpu_index))
  {
    #ifndef __CUDA_ARCH__
      cpu_index = std::move(other.cpu_index);
    #endif
  }

  CUDA DispatchIndex(const this_type& other, bvector<variable_type>* lvars)
  {
    gpu_index = std::move(battery::allocate_unique<ListVarIndex<allocator_type>>(lvars->get_allocator(), *other.gpu_index, lvars));
    #ifndef __CUDA_ARCH__
      cpu_index = std::move(battery::allocate_unique<HashMapVarIndex<allocator_type>>(lvars->get_allocator(), *other.cpu_index, lvars));
    #endif
  }

  template <class Alloc2>
  CUDA DispatchIndex(const DispatchIndex<Alloc2>& other, bvector<variable_type>* lvars)
  {
    gpu_index = std::move(battery::allocate_unique<ListVarIndex<allocator_type>>(lvars->get_allocator(), *other.gpu_index, lvars));
    #ifndef __CUDA_ARCH__
      cpu_index = std::move(battery::allocate_unique<HashMapVarIndex<allocator_type>>(lvars->get_allocator(), *other.cpu_index, lvars));
    #endif
  }

  // For this operator=, we suppose `lvars` is updated before.
  CUDA this_type& operator=(this_type&& other) {
    gpu_index = std::move(other.gpu_index);
    #ifndef __CUDA_ARCH__
      cpu_index = std::move(other.cpu_index);
    #endif
    return *this;
  }

  CUDA this_type& operator=(const this_type& other) {
    *gpu_index = *other.gpu_index;
    #ifndef __CUDA_ARCH__
      *cpu_index = *other.cpu_index;
    #endif
    return *this;
  }

  CUDA std::optional<size_t> lvar_index_of(const char* lv) const {
    #ifdef __CUDA_ARCH__
      return gpu_index->lvar_index_of(lv);
    #else
      return cpu_index->lvar_index_of(lv);
    #endif
  }

  CUDA void push_back(variable_type&& var) {
    #ifdef __CUDA_ARCH__
      gpu_index->push_back(std::move(var));
    #else
      cpu_index->push_back(std::move(var));
    #endif
  }

  CUDA void erase(const char* lv) {
    #ifdef __CUDA_ARCH__
      gpu_index->erase(lv);
    #else
      cpu_index->erase(lv);
    #endif
  }

  CUDA void set_lvars(bvector<variable_type>* lvars) {
    gpu_index->set_lvars(lvars);
    #ifndef __CUDA_ARCH__
      cpu_index->set_lvars(lvars);
    #endif
  }
};

/** A `VarEnv` is a variable environment mapping between logical variables and abstract variables. */
template <class Allocator>
class VarEnv {
  template <class F> using fstring = battery::string<typename F::allocator_type>;
public:
  using allocator_type = Allocator;
  using this_type = VarEnv<Allocator>;

  constexpr static const char* name = "VarEnv";

  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  using variable_type = Variable<Allocator>;

  template <class Alloc2>
  friend class VarEnv;

private:
  bvector<variable_type> lvars;
  bvector<bvector<size_t>> avar2lvar;
  DispatchIndex<allocator_type> var_index; // Note that this must always be declared *after* `lvars` because it stores a reference to it.

public:
  CUDA NI AType extends_abstract_dom() {
    avar2lvar.push_back(bvector<int>(get_allocator()));
    return static_cast<AType>(avar2lvar.size()) - 1;
  }

  CUDA NI void extends_abstract_doms(AType aty) {
    assert(aty != UNTYPED);
    while(aty >= avar2lvar.size()) {
      extends_abstract_dom();
    }
  }

  template <class Alloc2, class Alloc3>
  CUDA NI AVar extends_vars(AType aty, const battery::string<Alloc2>& name, const Sort<Alloc3>& sort) {
    extends_abstract_doms(aty);
    AVar avar(aty, static_cast<int>(avar2lvar[aty].size()));
    // We first verify the variable doesn't already exist.
    auto lvar_idx = var_index.lvar_index_of(name.data());
    if(lvar_idx.has_value()) {
      auto avar_opt = lvars[*lvar_idx].avar_of(aty);
      if(avar_opt.has_value()) {
        return *avar_opt;
      }
      else {
        lvars[*lvar_idx].avars.push_back(avar);
      }
    }
    else {
      lvar_idx ={lvars.size()};
      var_index.push_back(Variable<allocator_type>{name, sort, avar, get_allocator()});
    }
    avar2lvar[aty].push_back(*lvar_idx);
    return avar;
  }

public:
  CUDA VarEnv(const Allocator& allocator): lvars(allocator), avar2lvar(allocator), var_index(&lvars) {}
  CUDA VarEnv(this_type&& other): lvars(std::move(other.lvars)), avar2lvar(std::move(other.avar2lvar)), var_index(std::move(other.var_index), &lvars) {}
  CUDA VarEnv(): VarEnv(Allocator{}) {}
  CUDA VarEnv(const this_type& other): lvars(other.lvars), avar2lvar(other.avar2lvar), var_index(other.var_index, &lvars) {}

  template <class Alloc2>
  CUDA VarEnv(const VarEnv<Alloc2>& other, const Allocator& allocator = Allocator{})
    : lvars(other.lvars, allocator)
    , avar2lvar(other.avar2lvar, allocator)
    , var_index(other.var_index, &lvars)
  {}

  CUDA this_type& operator=(this_type&& other) {
    lvars = std::move(other.lvars);
    avar2lvar = std::move(other.avar2lvar);
    var_index = std::move(other.var_index);
    var_index.set_lvars(&lvars);
    return *this;
  }

  CUDA this_type& operator=(const this_type& other) {
    lvars = other.lvars;
    avar2lvar = other.avar2lvar;
    var_index = DispatchIndex<allocator_type>(other.var_index, &lvars);
    var_index.set_lvars(&lvars);
    return *this;
  }

  template <class Alloc2>
  CUDA this_type& operator=(const VarEnv<Alloc2>& other) {
    lvars = other.lvars;
    avar2lvar = other.avar2lvar;
    var_index = DispatchIndex<allocator_type>(other.var_index, &lvars);
    var_index.set_lvars(&lvars);
    return *this;
  }

  CUDA allocator_type get_allocator() const {
    return lvars.get_allocator();
  }

  CUDA size_t num_abstract_doms() const {
    return avar2lvar.size();
  }

  CUDA size_t num_vars() const {
    return lvars.size();
  }

  CUDA size_t num_vars_in(AType aty) const {
    if(aty >= avar2lvar.size()) {
      return 0;
    }
    else {
      return avar2lvar[aty].size();
    }
  }

  CUDA void print() const {
    printf("Environment (%lu variables): \n", num_vars());
    printf("index\t name               sort\tavars\n");
    for(int i = 0; i < num_vars(); ++i) {
      printf("%d\t", i);
      const auto& var = lvars[i];
      printf("%s", var.name.data());
      for(int i = var.name.size(); i < 20; ++i) {
        printf(" ");
      }
      var.sort.print(); printf("\t\t");
      for(int j = 0; j < var.avars.size(); ++j) {
        printf("(%d,%d)", var.avars[j].aty(), var.avars[j].vid());
        if(j != var.avars.size() - 1) {
          printf(",");
        }
      }
      printf("\n");
    }
  }

  CUDA NI std::optional<std::reference_wrapper<const variable_type>> variable_of(const char* lv) const {
    auto r = var_index.lvar_index_of(lv);
    if(r.has_value()) {
      return std::cref(lvars[*r]);
    }
    else {
      return {};
    }
  }

  template <class Alloc2>
  CUDA std::optional<std::reference_wrapper<const variable_type>> variable_of(const battery::string<Alloc2>& lv) const {
    return variable_of(lv.data());
  }

  template <class Alloc2>
  CUDA bool contains(const battery::string<Alloc2>& lv) const {
    return contains(lv.data());
  }

  CUDA bool contains(const char* lv) const {
    return variable_of(lv).has_value();
  }

  CUDA bool contains(AVar av) const {
    if(!av.is_untyped()) {
      return avar2lvar.size() > av.aty() && avar2lvar[av.aty()].size() > av.vid();
    }
    return false;
  }

  CUDA const variable_type& operator[](int i) const {
    return lvars[i];
  }

  CUDA const variable_type& operator[](AVar av) const {
    return lvars[avar2lvar[av.aty()][av.vid()]];
  }

  CUDA const bstring& name_of(AVar av) const {
    return (*this)[av].name;
  }

  CUDA const Sort<Allocator>& sort_of(AVar av) const {
    return (*this)[av].sort;
  }

  struct snapshot_type {
    bvector<size_t> lvars_snap;
    bvector<size_t> avar2lvar_snap;
  };

  /** Save the state of the environment. */
  CUDA NI snapshot_type snapshot() const {
    snapshot_type snap;
    for(int i = 0; i < lvars.size(); ++i) {
      snap.lvars_snap.push_back(lvars[i].avars.size());
    }
    for(int i = 0; i < avar2lvar.size(); ++i) {
      snap.avar2lvar_snap.push_back(avar2lvar[i].size());
    }
    return std::move(snap);
  }

  /** Restore the environment to its previous state `snap`. */
  CUDA NI void restore(const snapshot_type& snap) {
    assert(lvars.size() >= snap.lvars_snap.size());
    assert(avar2lvar.size() >= snap.avar2lvar_snap.size());
    while(lvars.size() > snap.lvars_snap.size()) {
      var_index.erase(lvars.back().name.data());
      lvars.pop_back();
    }
    for(int i = 0; i < lvars.size(); ++i) {
      lvars[i].avars.resize(snap.lvars_snap[i]);
    }
    while(avar2lvar.size() > snap.avar2lvar_snap.size()) {
      avar2lvar.pop_back();
    }
    for(int i = 0; i < avar2lvar.size(); ++i) {
      avar2lvar[i].resize(snap.avar2lvar_snap[i]);
    }
  }
};

/** Given a formula `f` and an environment, return the first variable occurring in `f` or `{}` if `f` has no variable in `env`. */
template <class F, class Env>
CUDA NI std::optional<std::reference_wrapper<const typename Env::variable_type>> var_in(const F& f, const Env& env) {
  const auto& g = var_in(f);
  switch(g.index()) {
    case F::V:
      if(g.v().is_untyped()) { return {}; }
      else { return std::cref(env[g.v()]); }
    case F::E:
      return env.variable_of(battery::get<0>(g.exists()));
    case F::LV:
      return env.variable_of(g.lv());
  }
  return {};
}
}

#endif
