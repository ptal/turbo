// Copyright 2022 Pierre Talbot

#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include "allocator.hpp"
#include "config.hpp"
#include "statistics.hpp"
#include "XCSP3_parser.hpp"
#include "cp.hpp"
#include "fixpoint.hpp"

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

template <class IterationStrategy, class Allocator>
CUDA void run_solver(SFormula<Allocator>& sf, bool& stop, Statistics& stats) {
  CP_BAB<Allocator, IterationStrategy> cp(stop);
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

#ifdef __NVCC__

CUDA_GLOBAL void gpu_run_solver(const SFormula<battery::ManagedAllocator>& sf, bool& stop, Statistics& stats) {
  SFormula<battery::GlobalAllocatorGPU> sf_in_global(sf);
  run_solver<AsynchronousIterationGPU>(sf_in_global, stop, stats);
}

#endif

template <class Allocator>
class Bencher {
  std::thread timeout_thread;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  battery::shared_ptr<bool, Allocator> force_stop;
public:
  Bencher(): force_stop(battery::make_shared<bool, Allocator>(false)) {}

  void start(int timeout) {
    timeout_thread = std::thread(guard_timeout, timeout, std::ref(*force_stop));
    start_time = std::chrono::high_resolution_clock::now();
  }

  int64_t stop() {
    *force_stop = true;
    auto end_time = std::chrono::high_resolution_clock::now();
    int64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    timeout_thread.join();
    return duration;
  }

  bool& stop_ref() {
    return *force_stop;
  }
};

template <class Allocator>
void bench_solver(Configuration& config) {
  XCSP3Core::XCSP3_turbo_callbacks<Allocator> cb;
  parse_xcsp3(config.problem_path, cb);
  auto sf = cb.build_formula();
  Bencher<Allocator> bencher;
  bencher.start(config.timeout);
  auto stats = battery::make_shared<GlobalStatistics, Allocator>(GlobalStatistics(cb.num_variables(), cb.num_constraints()));
  if(config.arch == CPU) {
    run_solver<GaussSeidelIteration>(*sf, bencher.stop_ref(), stats->local);
  }
  else {
    #ifdef __NVCC__
      if constexpr(std::is_same_v<Allocator, battery::ManagedAllocator>) {
        gpu_run_solver<<<config.or_nodes, config.and_nodes>>>(*sf, bencher.stop_ref(), stats->local);
        CUDIE(cudaDeviceSynchronize());
      }
      else {
        assert(0);
      }
    #endif
  }
  stats->duration = bencher.stop();
  stats->print_csv();
}

int main(int argc, char** argv) {
  Configuration config = parse_args(argc, argv);
  try
  {
    if(config.arch == CPU) {
      bench_solver<battery::StandardAllocator>(config);
    }
    else {
      #ifdef __NVCC__
        bench_solver<battery::ManagedAllocator>(config);
      #else
        printf("Turbo need to be compiled with NVCC in order to run on GPU.");
        exit(EXIT_FAILURE);
      #endif
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
