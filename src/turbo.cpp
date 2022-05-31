// Copyright 2022 Pierre Talbot

#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include "config.hpp"
#include "statistics.hpp"
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

// Inspired by https://stackoverflow.com/questions/39513830/launch-cuda-kernel-with-a-timeout/39514902
// Timeout expected in seconds.
void guard_timeout(int timeout, bool& stop) {
  int progressed = 0;
  while (!stop) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    progressed += 1;
    if (progressed >= timeout) {
      stop = true;
    }
  }
}

int main(int argc, char** argv) {
  Configuration config = parse_args(argc, argv);
  try
  {
    XCSP3Core::XCSP3_turbo_callbacks<Allocator> cb;
    parse_xcsp3(config.problem_path, cb);

    bool stop = false;
    std::thread timeout_thread(guard_timeout, config.timeout, std::ref(stop));
    auto t1 = std::chrono::high_resolution_clock::now();

    GlobalStatistics stats(cb.num_variables(), cb.num_constraints());
    auto sf = cb.build_formula();
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
    while(!bab.extract(bab) && has_changed.guard() && !stop) {
      stats.local.nodes++;
      stats.local.depth_max = std::max(stats.local.depth_max, (int)search_tree->depth());
      has_changed = BInc::bot();
      // Compute \f$ pop \circ push \circ split \circ bab \circ refine \f$.
      seq_fixpoint(*ipc, has_changed);
      stats.local.fails += ipc->is_top().guard();
      bab.refine(has_changed);
      split->reset();
      seq_refine(*split, has_changed);
      search_tree->refine(has_changed);
    }
    stats.local.exhaustive = !stop;
    stop = true;
    auto t2 = std::chrono::high_resolution_clock::now();
    stats.duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    timeout_thread.join();
    stats.local.sols = bab.solutions_count().value();
    // Found the optimum.
    if(bab.extract(bab)) {
      auto opt_lvar = sf.optimization_lvar();
      auto opt_avar = *(bab.environment().to_avar(opt_lvar));
      if(sf.mode() == SF::MINIMIZE) {
        stats.local.best_bound = bab.optimum().project(opt_avar).lb().value();
      }
      else {
        stats.local.best_bound = bab.optimum().project(opt_avar).ub().value();
      }
    }
    stats.print_csv();
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
