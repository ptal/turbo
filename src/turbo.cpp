// Copyright 2022 Pierre Talbot

#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include "config.hpp"
#include "statistics.hpp"

#include "vector.hpp"
#include "shared_ptr.hpp"

#include "vstore.hpp"
#include "cartesian_product.hpp"
#include "interval.hpp"
#include "pc.hpp"
#include "terms.hpp"
#include "fixpoint.hpp"
#include "search_tree.hpp"
#include "bab.hpp"

#include "value_order.hpp"
#include "variable_order.hpp"
#include "split.hpp"

#include "flatzinc_parser.hpp"

using namespace lala;
using namespace battery;

using Itv = Interval<local::ZInc>;
using IStore = VStore<Itv, StandardAllocator>;
using IPC = PC<IStore>; // Interval Propagators Completion
using ISplitInputLB = Split<IPC, InputOrder<IPC>, LowerBound<IPC>>;
using IST = SearchTree<IPC, ISplitInputLB>;
using IBAB = BAB<IST>;

const AType sty = 0;
const AType pty = 1;
const AType tty = 2;
const AType split_ty = 3;
const AType bab_ty = 4;

template <class Timepoint>
bool check_timeout(const Configuration& config, GlobalStatistics& stats, const Timepoint& start) {
  if(config.timeout_ms == 0) {
    return true;
  }
  auto now = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
  if(elapsed >= config.timeout_ms) {
    stats.local.exhaustive = false;
    return false;
  }
  return true;
}

void solve(const Configuration& config, GlobalStatistics& stats) {
  auto start = std::chrono::high_resolution_clock::now();

  // I. Parse the FlatZinc model.
  using FormulaPtr = battery::shared_ptr<TFormula<StandardAllocator>, StandardAllocator>;
  FlatZincOutput<StandardAllocator> output;
  FormulaPtr f = parse_flatzinc<StandardAllocator>(config.problem_path, output);
  if(!f) {
    std::cerr << "Could not parse FlatZinc model." << std::endl;
    exit(EXIT_FAILURE);
  }

  printf("%%FlatZinc parsed\n");

  // II. Create the abstract domain.
  int num_vars = num_quantified_vars(*f);
  auto store = make_shared<IStore, StandardAllocator>(sty, num_vars);
  auto ipc = make_shared<IPC, StandardAllocator>(IPC(pty, store));
  auto split = make_shared<ISplitInputLB, StandardAllocator>(ISplitInputLB(split_ty, ipc, ipc));
  auto search_tree = make_shared<IST, StandardAllocator>(IST(tty, ipc, split));
  auto bab = make_shared<IBAB, StandardAllocator>(IBAB(bab_ty, search_tree));

  // III. Interpret the formula in the abstract domain.
  VarEnv<StandardAllocator> env;
  auto r = bab->interpret_in(*f, env);
  if(!r.has_value()) {
    r.print_diagnostics();
    exit(EXIT_FAILURE);
  }
  local::BInc has_changed;
  bab->tell(std::move(r.value()), has_changed);
  stats.variables = store->vars();
  stats.constraints = ipc->num_refinements();

  auto now = std::chrono::high_resolution_clock::now();
  stats.interpretation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();

  printf("%%Formula has been loaded, solving begins...\n");

  // IV. Solve the problem.
  AbstractDeps<StandardAllocator> deps;
  has_changed = true;
  while(has_changed && check_timeout(config, stats, start)) {
    has_changed = false;
    GaussSeidelIteration::fixpoint(*ipc, has_changed);
    split->reset();
    GaussSeidelIteration::iterate(*split, has_changed);
    if(bab->refine(env, has_changed)) {
      output.print_solution(env, bab->optimum());
      stats.print_mzn_separator();
      stats.local.solutions++;
      if(config.stop_after_n_solutions != 0 &&
         stats.local.solutions >= config.stop_after_n_solutions)
      {
        stats.local.exhaustive = false;
        break;
      }
    }
    search_tree->refine(env, has_changed);
  }
  stats.print_mzn_final_separator();
  now = std::chrono::high_resolution_clock::now();
  stats.duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
}

int main(int argc, char** argv) {
  try
  {
    Configuration config = parse_args(argc, argv);
    GlobalStatistics stats(10, 10, false);
    printf("%%"); config.print_commandline(argv[0]);
    solve(config, stats);
    if(config.print_statistics) {
      stats.print_mzn_statistics();
    }
  }
  catch (std::exception &e)
  {
    std::cout.flush();
    std::cerr << "\n\tUnexpected exception:\n";
    std::cerr << "\t" << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
  return 0;
}
