// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_FBAREBONES_FORWARD_HPP
#define TURBO_FBAREBONES_FORWARD_HPP

/** `-forward_only` for `-arch fbarebones`: the bytecodes of the network project in one direction, from
 * the variables they read to the variable they compute, as the forward propagation of `-arch jet` does.
 *
 * The bytecodes `x = y <op> z` of PIR are relations, and `PIR::fdeduce` projects each of them onto its
 * three variables. Which of the three a bytecode of the network computes depends on how the model was
 * written and ternarized:
 *   * an affine or ReLU neuron, `n = y + z`, `t = c * x`, `r = max(p, 0)`: the variable `x`;
 *   * `Sub`, `o = i - c`, which the ternarization writes `i = o + c`: the variable `y`;
 *   * `Flatten`, `Dropout`, `Pad`, ..., `o = i`, which is the bytecode `1 = (o == i)`: `y` or `z`.
 * So we find it from the network: we evaluate the bytecodes as functions from the inputs of the network
 * (`analyze`), as the network itself is evaluated, and a bytecode computes the variable that it gives
 * from the variables that are already known (the inputs and the constants of the model at first). The
 * bytecodes on which the output neurons `Y_*` depend are the network, and project onto that variable
 * only (`deduce_onto`). Every other bytecode is the property, and keeps the deduction of PIR in all
 * its directions: the comparisons `1 = (Y_0 <= Y_1)`, their combinations in a disjunction, and the terms
 * they compare, so that the property still bounds the output neurons, as with `-arch jet`.
 *
 * No version of lala-pc is needed: the projections are the ones of `PIR::fdeduce`, the projection onto
 * `x` bitwise. */

#include <cstdint>
#include <string>
#include <vector>

namespace fbarebones_forward {

/** The projections of a bytecode `x = y <op> z`: all of them (`BOTH`, the deduction of PIR), or only
 * the one onto `x`, `y` or `z`. */
enum Direction : uint8_t { BOTH = 0, ONTO_X = 1, ONTO_Y = 2, ONTO_Z = 3 };

/** The direction of each bytecode, and how many there are of each kind. */
struct Analysis {
  std::vector<uint8_t> direction;
  /** The bytecodes of the network, by direction (`onto[ONTO_X]`, ...), and among them the copies
   * `o = i` (`1 = (o == i)`). */
  int network = 0;
  int onto[4] = {0, 0, 0, 0};
  int copies = 0;
  /** The bytecodes of the property (`BOTH`). */
  int property = 0;
  /** The output neurons `Y_*` of the model, and those that the evaluation of the bytecodes reaches. */
  int outputs = 0;
  int outputs_reached = 0;
};

/** The direction of every bytecode of `cp` (a `CP<FItv>` after `preprocess`), see above. The inputs
 * are `cp.input_neurons`, the outputs the variables named `Y_*` by the ONNX parser, and the constants
 * the variables `__CONSTANT_*` of the ternarization, or any variable other than a neuron with a
 * single value at the root. */
template <class CPType>
Analysis analyze(const CPType& cp) {
  const auto& store = *cp.store;
  const auto& iprop = *cp.iprop;
  const auto aty = store.aty();
  const int nv = static_cast<int>(store.vars());
  const int nb = static_cast<int>(iprop.num_deductions());
  Analysis a;
  a.direction.assign(nb, BOTH);
  auto name = [&](int v) { return std::string(cp.env.name_of(lala::AVar(aty, v)).data()); };
  auto starts = [](const std::string& s, const char* p) { return s.rfind(p, 0) == 0; };
  /** The variables known before the evaluation: the inputs and the constants. */
  std::vector<char> known(nv, 0), is_constant(nv, 0), is_output(nv, 0);
  for(int v = 0; v < nv; ++v) {
    const std::string n = name(v);
    const auto& d = store[v];
    const bool singleton = !d.lb().is_top() && !d.ub().is_top() && d.lb().value() == d.ub().value();
    const bool neuron = starts(n, "X_") || starts(n, "Y_");
    if(starts(n, "__CONSTANT_") || (singleton && !neuron)) { is_constant[v] = 1; known[v] = 1; }
    if(starts(n, "Y_")) { is_output[v] = 1; ++a.outputs; }
  }
  for(size_t k = 0; k < cp.input_neurons.size(); ++k) {
    auto var = cp.env.variable_of(cp.input_neurons[k].data());
    if(var.has_value()) {
      auto av = var->get().avar_of(aty);
      if(av.has_value()) { known[av->vid()] = 1; }
    }
  }
  auto is_one = [&](int v) {
    const auto& d = store[v];
    return is_constant[v] && !d.lb().is_top() && d.lb().value() == 1 && d.ub().value() == 1;
  };
  /** The evaluation: `defined_by[v]` is the bytecode that computes `v`, and `target[i]` the direction of
   * the bytecode `i`. The arithmetic bytecodes first, the copies `1 = (o == i)` only when no arithmetic
   * bytecode can be evaluated, so that a comparison of the property between two outputs (`Y_0 = Y_1`)
   * is not taken as a copy before both are computed by the network. */
  std::vector<int> defined_by(nv, -1);
  std::vector<uint8_t> target(nb, BOTH);
  auto evaluate = [&](bool copies) {
    bool progress = false;
    for(int i = 0; i < nb; ++i) {
      if(target[i] != BOTH) { continue; }
      const auto bc = iprop.load_deduce(i);
      const int x = bc.x.vid(), y = bc.y.vid(), z = bc.z.vid();
      uint8_t t = BOTH;
      int v = -1;
      if(!copies) {
        switch(bc.op) {
          case lala::ADD:
            if(!known[x] && known[y] && known[z]) { t = ONTO_X; v = x; }
            else if(known[x] && !known[y] && known[z] && y != z) { t = ONTO_Y; v = y; }
            else if(known[x] && known[y] && !known[z] && y != z) { t = ONTO_Z; v = z; }
            break;
          case lala::MUL:
          case lala::MIN:
          case lala::MAX:
            if(!known[x] && known[y] && known[z]) { t = ONTO_X; v = x; }
            break;
          default: break;
        }
      }
      else if(bc.op == lala::EQ && is_one(x)) {
        if(known[y] && !known[z]) { t = ONTO_Z; v = z; }
        else if(!known[y] && known[z]) { t = ONTO_Y; v = y; }
      }
      if(t != BOTH && defined_by[v] == -1) {
        target[i] = t;
        defined_by[v] = i;
        known[v] = 1;
        progress = true;
      }
    }
    return progress;
  };
  while(true) {
    while(evaluate(false)) {}
    if(!evaluate(true)) { break; }
  }
  /** The network: the bytecodes on which the outputs depend. */
  std::vector<int> todo;
  std::vector<char> in_cone(nv, 0);
  for(int v = 0; v < nv; ++v) {
    if(is_output[v]) {
      in_cone[v] = 1;
      todo.push_back(v);
      if(defined_by[v] != -1) { ++a.outputs_reached; }
    }
  }
  while(!todo.empty()) {
    const int v = todo.back();
    todo.pop_back();
    const int i = defined_by[v];
    if(i == -1) { continue; }
    a.direction[i] = target[i];
    const auto bc = iprop.load_deduce(i);
    const int vars[3] = {bc.x.vid(), bc.y.vid(), bc.z.vid()};
    for(int u : vars) {
      if(u != v && !is_constant[u] && !in_cone[u]) { in_cone[u] = 1; todo.push_back(u); }
    }
  }
  for(int i = 0; i < nb; ++i) {
    if(a.direction[i] == BOTH) { ++a.property; continue; }
    ++a.network;
    ++a.onto[a.direction[i]];
    if(iprop.load_deduce(i).op == lala::EQ) { ++a.copies; }
  }
  return a;
}

/** The projection of the bytecode `bc` in the direction `d` (not `BOTH`), embedded in `store`; the
 * variables of `bc` are read in `store`. `Itv` is the interval type of PIR (`PIR::Itv`).
 *   * `ONTO_X`: `x = y <op> z` for `ADD`, `MUL`, `MIN` and `MAX`, bitwise the projection onto `x` of
 *     `PIR::fdeduce` (the same code, which computes it first, from `y` and `z`);
 *   * `ONTO_Y`, `ONTO_Z` of `ADD`: `y = x - z`, `z = x - y`, as `PIR::fdeduce` but from the `x` of the
 *     store (`fdeduce` uses the `x` it has just tightened, which is not the variable computed here);
 *   * `ONTO_Y`, `ONTO_Z` of `1 = (y == z)`: the copy `y = z`, `z = y`.
 * \return whether the store changed. */
template <class Itv, class Store, class Bytecode>
CUDA bool deduce_onto(Store& store, const Bytecode& bc, uint8_t d) {
  using value_t = typename Itv::LB::value_type;
  using LB = typename Itv::LB;
  using UB = typename Itv::UB;
  Itv r1(store[bc.x]);
  Itv r2(store[bc.y]);
  Itv r3(store[bc.z]);
  const value_t xl = r1.lb().value(), xu = r1.ub().value();
  const value_t yl = r2.lb().value(), yu = r2.ub().value();
  const value_t zl = r3.lb().value(), zu = r3.ub().value();
  if(bc.op == lala::EQ) {
    if(d == ONTO_Y) { return store.embed(bc.y, r3); }
    if(d == ONTO_Z) { return store.embed(bc.z, r2); }
    return false;
  }
  if(d == ONTO_Y || d == ONTO_Z) {
    /** `ADD`: `y = x - z` or `z = x - y`. */
    Itv& r = (d == ONTO_Y) ? r2 : r3;
    const Itv& other = (d == ONTO_Y) ? r3 : r2;
    if(r1.is_bot() || other.is_bot()) { r.meet_bot(); }
    else {
      const value_t ol = other.lb().value(), ou = other.ub().value();
      r.meet_lb(LB(battery::sub_down<value_t>(xl, ou)));
      r.meet_ub(UB(battery::sub_up<value_t>(xu, ol)));
    }
    return store.embed(d == ONTO_Y ? bc.y : bc.z, r);
  }
  /** `ONTO_X`, the code of `PIR::fdeduce`. */
  if(r2.is_bot() || r3.is_bot()) { r1.meet_bot(); }
  else {
    switch(bc.op) {
      case lala::ADD: {
        r1.meet_lb(LB(battery::add_down<value_t>(yl, zl)));
        r1.meet_ub(UB(battery::add_up<value_t>(yu, zu)));
        break;
      }
      case lala::MUL: {
        if(!r2.lb().is_top() && !r2.ub().is_top() && !r3.lb().is_top() && !r3.ub().is_top()) {
          r1.meet_lb(LB(battery::min(battery::min(battery::mul_down<value_t>(yl, zl), battery::mul_down<value_t>(yl, zu)), battery::min(battery::mul_down<value_t>(yu, zl), battery::mul_down<value_t>(yu, zu)))));
          r1.meet_ub(UB(battery::max(battery::max(battery::mul_up<value_t>(yl, zl), battery::mul_up<value_t>(yl, zu)), battery::max(battery::mul_up<value_t>(yu, zl), battery::mul_up<value_t>(yu, zu)))));
        }
        else if((yl == value_t{0.0} && yu == value_t{0.0}) || (zl == value_t{0.0} && zu == value_t{0.0})) {
          r1.meet_lb(LB(value_t{0.0}));
          r1.meet_ub(UB(value_t{0.0}));
        }
        else {
          switch(3 * (yl >= 0 ? 0 : (yu > 0 ? 1 : 2)) + (zl >= 0 ? 0 : (zu > 0 ? 1 : 2))) {
            case 0: { r1.meet_lb(LB(battery::mul_down<value_t>(yl, zl))); r1.meet_ub(UB(battery::mul_up<value_t>(yu, zu))); break; }
            case 1: { r1.meet_lb(LB(battery::mul_down<value_t>(yu, zl))); r1.meet_ub(UB(battery::mul_up<value_t>(yu, zu))); break; }
            case 2: { r1.meet_lb(LB(battery::mul_down<value_t>(yu, zl))); r1.meet_ub(UB(battery::mul_up<value_t>(yl, zu))); break; }
            case 3: { r1.meet_lb(LB(battery::mul_down<value_t>(yl, zu))); r1.meet_ub(UB(battery::mul_up<value_t>(yu, zu))); break; }
            case 4: {
              r1.meet_lb(LB(battery::min(battery::mul_down<value_t>(yl, zu), battery::mul_down<value_t>(yu, zl))));
              r1.meet_ub(UB(battery::max(battery::mul_up<value_t>(yl, zl), battery::mul_up<value_t>(yu, zu))));
              break;
            }
            case 5: { r1.meet_lb(LB(battery::mul_down<value_t>(yu, zl))); r1.meet_ub(UB(battery::mul_up<value_t>(yl, zl))); break; }
            case 6: { r1.meet_lb(LB(battery::mul_down<value_t>(yl, zu))); r1.meet_ub(UB(battery::mul_up<value_t>(yu, zl))); break; }
            case 7: { r1.meet_lb(LB(battery::mul_down<value_t>(yl, zu))); r1.meet_ub(UB(battery::mul_up<value_t>(yl, zl))); break; }
            case 8:
            default: { r1.meet_lb(LB(battery::mul_down<value_t>(yu, zu))); r1.meet_ub(UB(battery::mul_up<value_t>(yl, zl))); break; }
          }
        }
        break;
      }
      case lala::MIN: {
        r1.meet_lb(LB(battery::min(yl, zl)));
        r1.meet_ub(UB(battery::min(yu, zu)));
        break;
      }
      case lala::MAX: {
        r1.meet_lb(LB(battery::max(yl, zl)));
        r1.meet_ub(UB(battery::max(yu, zu)));
        break;
      }
      default: return false;
    }
  }
  return store.embed(bc.x, r1);
}

} // namespace fbarebones_forward

#endif
