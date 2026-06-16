// ============================================================================
// turbo_tcn_json.h — dump the preprocessed TCN as batched_ac_proto JSON.
//
// Sibling of turbo_dumper.h (which writes the prototype's BINARY fixture). This
// writes the JSON bridge schema consumed by batched_ac_proto/bp/problem.py
// (load_json), for the AC-vs-BC decision-gate measurement (batched_ac_proto
// plan.md T1.1):
//
//   {"variables":  [{"lb": 0, "ub": 7}, ...],
//    "constraints":[{"op": "add", "x": 2, "y": 0, "z": 1}, ...]}
//
// Variable indices are the POST-preprocessing TCN variables; domains are the
// preprocessed root domains (exactly what the histogram export in
// common_solving.hpp iterates). Call AFTER preprocess_tcn has populated
// store + iprop (same call site as turbo_dump_fixture).
//
// Op map (lala::Sig -> proto schema {add,mul,div,mod,min,max,eq,leq}):
//   ADD->add  MUL->mul  MIN->min  MAX->max  EQ->eq  LEQ->leq
//   TDIV/CDIV/FDIV/EDIV->div   (proto accepts then rejects div/mod at compile,
//                               loudly — surfaces coverage gaps, per its README)
//   SUB(x=y-z) is rewritten to add as y=x+z (the proto schema has no sub).
// Any other Sig aborts with a clear message rather than emitting a wrong model.
// ============================================================================
#ifndef TURBO_TCN_JSON_H
#define TURBO_TCN_JSON_H

#include <cstdio>
#include <stdexcept>
#include <string>

#include "lala/logic/ast.hpp"        // for lala::Sig
#include "lala/split_strategy.hpp"   // for VariableOrder/ValueOrder + string_of_*

template<class CP>
inline void turbo_dump_tcn_json(const CP& cp, const std::string& path) {
  const auto& store = *cp.store;
  const auto& iprop = *cp.iprop;
  const int num_vars = store.vars();
  const int num_bc   = iprop.num_deductions();

  std::FILE* f = std::fopen(path.c_str(), "w");
  if(!f) throw std::runtime_error("turbo_tcn_json: cannot open " + path);

  std::fputs("{\"variables\":[", f);
  for(int i = 0; i < num_vars; ++i) {
    auto dom = store[i];
    std::fprintf(f, "%s{\"lb\":%lld,\"ub\":%lld}", (i ? "," : ""),
                 (long long)dom.lb().value(), (long long)dom.ub().value());
  }
  std::fputs("],\"constraints\":[", f);

  bool first = true;
  for(int i = 0; i < num_bc; ++i) {
    auto bc = iprop.load_deduce(i);
    int x = bc.x.vid(), y = bc.y.vid(), z = bc.z.vid();
    const char* op;
    switch(bc.op) {
      case lala::ADD:  op = "add"; break;
      case lala::MUL:  op = "mul"; break;
      case lala::MIN:  op = "min"; break;
      case lala::MAX:  op = "max"; break;
      case lala::EQ:   op = "eq";  break;
      case lala::LEQ:  op = "leq"; break;
      case lala::TDIV:
      case lala::CDIV:
      case lala::FDIV:
      case lala::EDIV: op = "div"; break;
      case lala::SUB:  op = "add"; { int t = x; x = y; y = t; } break; // x=y-z <=> y=x+z
      default:
        std::fclose(f);
        throw std::runtime_error(
          "turbo_tcn_json: unsupported Sig " + std::to_string((int)bc.op) +
          " at constraint " + std::to_string(i) +
          " (extend the op map / proto schema)");
    }
    std::fprintf(f, "%s{\"op\":\"%s\",\"x\":%d,\"y\":%d,\"z\":%d}",
                 (first ? "" : ","), op, x, y, z);
    first = false;
  }
  std::fputs("]", f);   // close "constraints"

  // Objective: Turbo rewrites maximize into minimize (negating the var), so the
  // direction is always "min" on minimize_obj_var. Absent for satisfaction
  // problems. The prototype pins this var's upper bound to a fixed value to make
  // the search FAIL (the gate measures BC-vs-AC fails), standing in for B&B.
  if(!cp.minimize_obj_var.is_untyped()) {
    std::fprintf(f, ",\"objective\":{\"var\":%d,\"dir\":\"min\"}",
                 cp.minimize_obj_var.vid());
  }

  // Search strategies from the FZN annotation (var order, value order, var list)
  // so the prototype branches the same way Turbo does. Empty "vars" => split on
  // the whole store. (May include Turbo's EPS strategy first; the prototype uses
  // the FZN search strategies that follow.)
  std::fputs(",\"search\":[", f);
  const auto& strats = cp.split->strategies_();
  for(size_t s = 0; s < strats.size(); ++s) {
    const auto& st = strats[s];
    std::fprintf(f, "%s{\"var_order\":\"%s\",\"val_order\":\"%s\",\"vars\":[",
                 (s ? "," : ""),
                 lala::string_of_variable_order(st.var_order),
                 lala::string_of_value_order(st.val_order));
    for(size_t i = 0; i < st.vars.size(); ++i)
      std::fprintf(f, "%s%d", (i ? "," : ""), st.vars[i].vid());
    std::fputs("]}", f);
  }
  std::fputs("]}\n", f);

  std::fclose(f);
  std::printf("[turbo_tcn_json] wrote TCN: %d vars, %d constraints, obj_var=%d -> %s\n",
              num_vars, num_bc,
              cp.minimize_obj_var.is_untyped() ? -1 : cp.minimize_obj_var.vid(),
              path.c_str());
}

#endif // TURBO_TCN_JSON_H
