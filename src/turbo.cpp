// Copyright 2022 Pierre Talbot

#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include "config.hpp"
#include "statistics.hpp"
#include "XCSP3_parser.hpp"
#include "cp.hpp"

using namespace lala;


// CUDA void print_variables(const IStore& store) {
//   const auto& env = store.environment();
//   for(int i = 0; i < env.size(); ++i) {
//     const auto& vname = env[i];
//     vname.print();
//     printf("=");
//     store.project(*(env.to_avar(vname))).print();
//     printf("  ");
//   }
//   printf("\n");
// }

// template <class F>
// CUDA_GLOBAL void search_k(F& sf, Statistics& stats, bool& stop) {
//   infer_type(sf.formula(), sty, pty);
//   // sf.formula().print(true);
//   auto store = make_shared<IStore, Allocator>(std::move(IStore::bot(sty)));
//   auto ipc = make_shared<IIPC, Allocator>(IIPC(pty, store));
//   auto split = make_shared<ISplitInputLB, Allocator>(ISplitInputLB(split_ty, ipc, ipc));
//   auto search_tree = make_shared<IST, Allocator>(tty, ipc, split);
//   auto bab = IBAB(bab_ty, search_tree);

//   // printf("Abstract domains initialized...\n");
//   auto res = bab.interpret(sf);
//   if(!res.has_value()) {
//     printf("The formula could not be interpreted in the BAB(ST(IPC)) abstract domain.\n");
//     return;
//   }
// }

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

template <class SF>
CUDA void run_solver(SF& sf, bool& stop, Statistics& stats) {
  CP_BAB<battery::StandardAllocator, GaussSeidelIteration> cp(stop);
  auto x = cp.interpret(sf);
  if(x.has_value()) {
    BInc has_changed = BInc::bot();
    cp.tell(std::move(*x), has_changed);
    cp.refine(0, has_changed);
    stats = cp.statistics();
  }
  else {
    printf("The formula could not be interpreted in the CP_BAB abstract domain.\n");
  }
}

class Bencher {
  std::thread timeout_thread;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
public:
  bool force_stop;
  Bencher(): force_stop(false) {}

  void start(int timeout) {
    timeout_thread = std::thread(guard_timeout, timeout, std::ref(force_stop));
    start_time = std::chrono::high_resolution_clock::now();
  }

  int64_t stop() {
    force_stop = true;
    auto end_time = std::chrono::high_resolution_clock::now();
    int64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    timeout_thread.join();
    return duration;
  }
};

int main(int argc, char** argv) {
  Configuration config = parse_args(argc, argv);
  try
  {
    if(config.arch == GPU) {
      printf("not yet supported\n");
    }
    XCSP3Core::XCSP3_turbo_callbacks<battery::StandardAllocator> cb;
    parse_xcsp3(config.problem_path, cb);
    auto sf = cb.build_formula();
    GlobalStatistics stats(cb.num_variables(), cb.num_constraints());
    Bencher bencher;
    bencher.start(config.timeout);
    run_solver(sf, bencher.force_stop, stats.local);
    stats.duration = bencher.stop();
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
