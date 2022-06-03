// Copyright 2022 Pierre Talbot

#ifndef CP_HPP
#define CP_HPP

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
#include "ast.hpp"

using namespace lala;

const static AType sty = 0;
const static AType pty = 1;
const static AType tty = 2;
const static AType split_ty = 3;
const static AType bab_ty = 4;

/** A discrete constraint programming solver, based on a branch and bound, and propagate and search algorithms. */
template <class Allocator, class IterationStrategy>
class CP_BAB {
  using zi = ZInc<int>;
  using Itv = Interval<zi>;
  using IStore = VStore<Itv, Allocator>;
  using IIPC = IPC<IStore>;
  using ISplitInputLB = Split<IIPC, InputOrder<IIPC>, LowerBound<IIPC>>;
  using IST = SearchTree<IIPC, ISplitInputLB>;
  using IBAB = BAB<IST>;
  using SF = SFormula<Allocator>;

  Statistics stats;
  bool& stop;
  battery::shared_ptr<IStore, Allocator> store;
  battery::shared_ptr<IIPC, Allocator> ipc;
  battery::shared_ptr<ISplitInputLB, Allocator> split;
  battery::shared_ptr<IST, Allocator> search_tree;
  IBAB bab;

public:

  using this_type = CP_BAB<Allocator, IterationStrategy>;
  using TellType = typename IBAB::TellType;

  CUDA CP_BAB(bool& stop): stats(), stop(stop),
    store(battery::make_shared<IStore, Allocator>(std::move(IStore::bot(sty)))),
    ipc(battery::make_shared<IIPC, Allocator>(IIPC(pty, store))),
    split(battery::make_shared<ISplitInputLB, Allocator>(ISplitInputLB(split_ty, ipc, ipc))),
    search_tree(battery::make_shared<IST, Allocator>(tty, ipc, split)),
    bab(bab_ty, search_tree)
  {}

  CUDA thrust::optional<TellType> interpret(SF& sf) {
    infer_type(sf.formula(), sty, pty);
    return bab.interpret(sf);
  }

  CUDA this_type& tell(TellType&& res, BInc& has_changed) {
    bab.tell(std::move(res), has_changed);
    return *this;
  }

private:
  /** Is called just after the IPC refinement. */
  CUDA void refine_node_statistics() {
    stats.nodes++;
    stats.depth_max = std::max(stats.depth_max, (int)search_tree->depth());
    stats.fails += ipc->is_top().guard();
  }

  CUDA void refine_tree_statistics(bool is_underappx) {
    stats.exhaustive = !stop;
    stats.sols = bab.solutions_count().value();
    if(is_underappx) {
      const auto& objective = bab.optimum().project(bab.objective_var());
      if(bab.is_minimization()) {
        stats.best_bound = objective.lb().value();
      }
      else {
        stats.best_bound = objective.ub().value();
      }
    }
  }

public:
  CUDA void refine(int i, BInc& has_changed) {
    SHARED bool is_underappx;
    SHARED bool continue_refining;
    if(i == 0) {
      is_underappx = bab.extract(bab);
      continue_refining = !is_underappx && !stop;
    }
    IterationStrategy::barrier();
    while(continue_refining) {
      // Compute \f$ pop \circ push \circ split \circ bab \circ refine \f$.
      IterationStrategy::fixpoint(*ipc, has_changed);
      if(i == 0) {
        refine_node_statistics();
        bab.refine(has_changed);
        split->reset();
      }
      IterationStrategy::barrier();
      IterationStrategy::iterate(*split, has_changed);
      IterationStrategy::barrier();
      if(i == 0) {
        search_tree->refine(has_changed);
        is_underappx = bab.extract(bab);
        continue_refining = !is_underappx && has_changed.guard() && !stop;
        has_changed = BInc::bot();
      }
      IterationStrategy::barrier();
    }
    if(i == 0) {
      refine_tree_statistics(is_underappx);
    }
  }

  CUDA Statistics statistics() const {
    return stats;
  }
};

#endif
