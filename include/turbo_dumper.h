// ============================================================================
// turbo_dumper.h — Production-side fixture dumper for the propagation prototype.
//
// This header is meant to be DROPPED INTO TURBO (e.g., copy it next to
// barebones_dive_and_solve.hpp) and #included from common_solving.hpp's
// preprocess(). It translates turbo's internal post-preprocess state into
// the prototype's binary fixture format.
//
// Integration steps:
//   1. Copy this header to turbo/include/.
//   2. Add to turbo's Configuration:
//        std::string dump_fixture_path;
//   3. Add CLI flag parsing for --dump-fixture <path>.
//   4. In common_solving.hpp's preprocess(), after the existing
//      preprocess_tcn / preprocess_ipc call returns, add:
//
//        if(!config.dump_fixture_path.empty()) {
//          turbo_dump_fixture(*this, config.dump_fixture_path);
//          std::exit(0);  // exit after dump; agent loop uses the fixture
//        }
//
// The dumper writes the SAME binary format documented in
// prototype/include/proto_fixture.h. The prototype reads it back unchanged.
//
// Op translation:
//   Turbo's bytecodes use lala::Sig (enum in lala/logic/ast.hpp). The
//   prototype's Op enum is a subset. Unsupported Sig values cause the dumper
//   to abort with a clear error message — so coverage gaps surface early
//   rather than silently producing wrong fixtures.
// ============================================================================
#ifndef TURBO_DUMPER_H
#define TURBO_DUMPER_H

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "lala/logic/ast.hpp"   // for lala::Sig

namespace turbo_dumper_detail {

// Match the prototype's enum byte values verbatim.
enum ProtoOp : int32_t {
  P_ADD = 0, P_SUB = 1, P_MUL = 2,
  P_TDIV = 3, P_CDIV = 4, P_FDIV = 5, P_EDIV = 6,
  P_MIN = 7, P_MAX = 8, P_EQ = 9, P_LEQ = 10,
};

// Translate lala::Sig → prototype Op. Throws on unsupported.
inline ProtoOp translate_op(lala::Sig sig, int constraint_index) {
  switch(sig) {
    case lala::ADD:  return P_ADD;
    case lala::SUB:  return P_SUB;
    case lala::MUL:  return P_MUL;
    case lala::TDIV: return P_TDIV;
    case lala::CDIV: return P_CDIV;
    case lala::FDIV: return P_FDIV;
    case lala::EDIV: return P_EDIV;
    case lala::MIN:  return P_MIN;
    case lala::MAX:  return P_MAX;
    case lala::EQ:   return P_EQ;
    case lala::LEQ:  return P_LEQ;
    default:
      throw std::runtime_error(
        "turbo_dumper: unsupported Sig " + std::to_string((int)sig) +
        " at constraint " + std::to_string(constraint_index) +
        ". Extend prototype/include/proto_types.h::Op to cover it.");
  }
}

#pragma pack(push, 1)
struct ProtoHeader {
  uint32_t magic;            // 'PROP' little-endian
  uint32_t version;
  int32_t  num_vars;
  int32_t  num_bytecodes;
  int32_t  captured_depth;
  int32_t  threads_per_block;
  int32_t  reserved[2];
};
static_assert(sizeof(ProtoHeader) == 32, "header layout drift");

struct ProtoItv {
  int32_t lb;
  int32_t ub;
};

struct ProtoBytecode {
  int32_t op;
  int32_t x;
  int32_t y;
  int32_t z;
};
#pragma pack(pop)

} // namespace turbo_dumper_detail

// Dump a fixture for the prototype.
//
// `cp` is turbo's AbstractDomains instance (CP<Itv>). At call time, cp.store
// must hold the initial intervals and cp.iprop->bytecodes must hold the
// constraint network — i.e., this should be called AFTER preprocess_tcn
// has populated them.
//
// `captured_depth` defaults to 0 (root). To capture mid-dive fixtures, run
// the solver to the desired depth before calling.
template<class CP>
inline void turbo_dump_fixture(const CP& cp, const std::string& path,
                               int captured_depth = 0,
                               int threads_per_block = 256) {
  using namespace turbo_dumper_detail;

  const auto& store = *cp.store;
  const auto& iprop = *cp.iprop;
  const int num_vars       = store.vars();
  const int num_bytecodes  = iprop.num_deductions();

  std::FILE* f = std::fopen(path.c_str(), "wb");
  if(!f) throw std::runtime_error("turbo_dumper: cannot open " + path);

  ProtoHeader h{};
  h.magic              = 0x50524F50;  // 'PROP'
  h.version            = 1;
  h.num_vars           = num_vars;
  h.num_bytecodes      = num_bytecodes;
  h.captured_depth     = captured_depth;
  h.threads_per_block  = threads_per_block;
  std::fwrite(&h, sizeof(h), 1, f);

  // Store: project each variable's interval.
  for(int i = 0; i < num_vars; ++i) {
    auto dom = store[i];
    ProtoItv pitv;
    pitv.lb = (int32_t)dom.lb().value();
    pitv.ub = (int32_t)dom.ub().value();
    std::fwrite(&pitv, sizeof(pitv), 1, f);
  }

  // Bytecodes: translate Sig and pack into the prototype's layout.
  for(int i = 0; i < num_bytecodes; ++i) {
    auto bc = iprop.load_deduce(i);
    ProtoBytecode pbc;
    pbc.op = (int32_t)translate_op(bc.op, i);
    pbc.x  = bc.x.vid();
    pbc.y  = bc.y.vid();
    pbc.z  = bc.z.vid();
    std::fwrite(&pbc, sizeof(pbc), 1, f);
  }

  std::fclose(f);
  std::printf("[turbo_dumper] wrote fixture: %d vars, %d bytecodes, depth=%d → %s\n",
              num_vars, num_bytecodes, captured_depth, path.c_str());
}

#endif // TURBO_DUMPER_H
