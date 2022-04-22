// Copyright 2022 Pierre Talbot

#include <iostream>
#include "XCSP3_parser.hpp"
#include "config.hpp"
#include "shared_ptr.hpp"
#include "unique_ptr.hpp"
#include "z.hpp"
#include "vstore.hpp"
#include "interval.hpp"
#include "ipc.hpp"
#include "type_inference.hpp"

using namespace lala;

using zi = ZInc<int>;
using Itv = Interval<zi>;
using IStore = VStore<Itv, battery::StandardAllocator>;
using IStorePtr = battery::shared_ptr<IStore, battery::StandardAllocator>;
using IIPC = IPC<IStore>;

CUDA void print_variables(const IStorePtr& store) {
  const auto& env = store->environment();
  for(int i = 0; i < env.size(); ++i) {
    const auto& vname = env[i];
    vname.print();
    printf(" = ");
    store->project(*(env.to_avar(vname))).print();
    printf("\n");
  }
}

int main(int argc, char** argv) {
  Configuration config = parse_args(argc, argv);
  try
  {
    auto sf = parse_xcsp3<battery::StandardAllocator>(config.problem_path);
    AType sty = 0;
    AType pty = 1;
    infer_type(sf.formula(), sty, pty);
    // sf.formula().print(true);
    IStorePtr istore(new IStore(IStore::bot(sty)));
    IIPC ipc(pty, istore);
    auto res = ipc.interpret(sf.formula());
    if(!res.has_value()) {
      printf("The formula could not be interpreted in the IPC abstract domain.\n");
      exit(EXIT_FAILURE);
    }
    BInc has_changed = BInc::bot;
    ipc.tell(std::move(*res), has_changed);
    printf("Variable store before propagation: \n");
    print_variables(istore);
    while(has_changed.guard()) {
      has_changed.dtell(BInc::bot());
      ipc.refine(has_changed);
    }
    printf("\n\nVariable store after propagation: \n");
    print_variables(istore);

    // Analysis of the abstract element.
    if(ipc.is_top().guard()) {
      printf("The problem is unsatisfiable.");
    }
    else {
      // Extract propagators
      auto seq = battery::get<0>(extract_ty(sf.formula(), pty)).seq();
      bool all_entailed = true;
      printf("Num of props: %d\n", seq.size());
      for(int i = 0; i < seq.size() && all_entailed; ++i) {
        seq[i].print(false);
        auto prop = ipc.interpret(seq[i]);
        assert(prop.has_value());
        if(!ipc.ask(*prop).guard()) {
          printf(" is unknown\n");
          all_entailed = false;
        }
        else {
          printf(" is entailed\n");
        }
      }
      if(all_entailed) {
        printf("The problem is satisfiable (all propagators are entailed).\n");
      }
      else {
        printf("The problem is unknown, there are unknown propagators.\n");
      }
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
