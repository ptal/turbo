// Copyright 2022 Pierre Talbot

#include <iostream>
#include "config.hpp"
#include "XCSP3_parser.hpp"
#include "shared_ptr.hpp"
#include "unique_ptr.hpp"
#include "z.hpp"
#include "vstore.hpp"
#include "interval.hpp"
#include "ipc.hpp"
#include "fixpoint.hpp"
#include "type_inference.hpp"
#include "search_tree.hpp"
#include "value_order.hpp"
#include "variable_order.hpp"
#include "split.hpp"
#include "bab.hpp"

using namespace lala;

using Allocator = battery::StandardAllocator;
using zi = ZInc<int>;
using Itv = Interval<zi>;
using IStore = VStore<Itv, Allocator>;
using IIPC = IPC<IStore>;
using ISplitInputLB = Split<IIPC, InputOrder<IIPC>, LowerBound<IIPC>>;
using IST = SearchTree<IIPC, ISplitInputLB>;
using IBAB = BAB<IST>;
using SF = SFormula<Allocator>;

const static AType sty = 0;
const static AType pty = 1;
const static AType tty = 2;
const static AType split_ty = 3;
const static AType bab_ty = 4;

CUDA void print_variables(const IStore& store) {
  const auto& env = store.environment();
  for(int i = 0; i < env.size(); ++i) {
    const auto& vname = env[i];
    vname.print();
    printf("=");
    store.project(*(env.to_avar(vname))).print();
    printf("  ");
  }
  printf("\n");
}

int main(int argc, char** argv) {
  Configuration config = parse_args(argc, argv);
  try
  {
    auto sf = parse_xcsp3<Allocator>(config.problem_path);
    infer_type(sf.formula(), sty, pty);
    // sf.formula().print(true);
    auto store = make_shared<IStore, Allocator>(std::move(IStore::bot(sty)));
    auto ipc = make_shared<IIPC, Allocator>(IIPC(pty, store));
    auto split = make_shared<ISplitInputLB, Allocator>(ISplitInputLB(split_ty, ipc, ipc));
    auto search_tree = make_shared<IST, Allocator>(tty, ipc, split);
    auto bab = IBAB(bab_ty, search_tree);

    // printf("Abstract domains initialized...\n");

    auto res = bab.interpret(sf);
    if(!res.has_value()) {
      printf("The formula could not be interpreted in the BAB(ST(IPC)) abstract domain.\n");
      exit(EXIT_FAILURE);
    }

    // printf("Logic formula interpreted...\n");

    BInc has_changed = BInc::bot();
    bab.tell(std::move(*res), has_changed);

    // printf("Logic formula joined in the abstract domain...\n");

    // Branch and bound fixpoint algorithm.
    int iterations = 0;
    while(!bab.extract(bab) && has_changed.guard()) {
      iterations++;
      has_changed = BInc::bot();
      // Compute \f$ pop \circ push \circ split \circ bab \circ refine \f$.
      seq_fixpoint(*ipc, has_changed);
      bab.refine(has_changed);
      split->reset();
      seq_refine(*split, has_changed);
      search_tree->refine(has_changed);
    }
    printf("iterations=%d\n", iterations);
    // Found the optimum.
    if(bab.extract(bab)) {
      auto opt_lvar = sf.optimization_lvar();
      auto opt_avar = *(bab.environment().to_avar(opt_lvar));
      printf("optimum=");
      if(sf.mode() == SF::MINIMIZE) {
        bab.optimum().project(opt_avar).lb().print();
      }
      else {
        bab.optimum().project(opt_avar).ub().print();
      }
      printf("\n");
    }
    else {
      printf("The problem is unsatisfiable.");
    }
  }
  catch (exception &e)
  {
    cout.flush();
    cerr << "\n\tUnexpected exception:\n";
    cerr << "\t" << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  return 0;
}
