// Copyright 2021 Pierre Talbot

#ifndef TURBO_STATISTICS_HPP
#define TURBO_STATISTICS_HPP

#include <chrono>
#include <algorithm>
#include "battery/utility.hpp"
#include "battery/allocator.hpp"
#include "lala/logic/ast.hpp"

enum class Timer {
  OVERALL,
  PREPROCESSING,
  SOLVE,
  SEARCH,
  // SPLIT,
  // PUSH,
  // POP,
  FIXPOINT,
  TRANSFER_CPU2GPU,
  TRANSFER_GPU2CPU,
  SELECT_FP_FUNCTIONS,
  WAIT_CPU,
  DIVE,
  NUM_TIMERS
};

template <class Allocator = battery::standard_allocator>
struct TimingStatistics {
  using allocator_type = Allocator;
  template <class Alloc> friend struct TimingStatistics;
private:
  battery::vector<int64_t, allocator_type> timers_ns;

public:

  TimingStatistics(const TimingStatistics&) = default;
  CUDA TimingStatistics():
    timers_ns(static_cast<int>(Timer::NUM_TIMERS), 0)
  {}

  template <class Alloc>
  CUDA TimingStatistics(const TimingStatistics<Alloc>& other):
    timers_ns(other.timers_ns)
  {}

  CUDA int64_t& time_of(Timer timer) {
    return timers_ns[static_cast<int>(timer)];
  }

  CUDA int64_t time_of(Timer timer) const {
    return timers_ns[static_cast<int>(timer)];
  }

  template <class Alloc>
  CUDA void meet(const TimingStatistics<Alloc>& other) {
    for(int i = 0; i < timers_ns.size(); i++) {
      timers_ns[i] += other.timers_ns[i];
    }
  }

  CUDA int64_t time_ms_of(Timer timer) const {
    return timers_ns[static_cast<int>(timer)] / 1000 / 1000;
  }

#ifdef __CUDACC__
  __device__ cuda::std::chrono::system_clock::time_point start_timer_device() const {
    if(threadIdx.x == 0) {
      return cuda::std::chrono::system_clock::now();
    }
    else {
      return cuda::std::chrono::system_clock::time_point{};
    }
  }
#endif

  std::chrono::steady_clock::time_point start_timer_host() const {
    return std::chrono::steady_clock::now();
  }

#ifdef __CUDACC__
  __device__ cuda::std::chrono::system_clock::time_point stop_timer(Timer timer, cuda::std::chrono::system_clock::time_point start) {
    if(threadIdx.x == 0) {
      auto now = cuda::std::chrono::system_clock::now();
      time_of(timer) += cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(
        now - start).count();
      return now;
    }
    return cuda::std::chrono::system_clock::time_point{};
  }
#endif

  std::chrono::steady_clock::time_point stop_timer(Timer timer, std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    time_of(timer) += std::chrono::duration_cast<std::chrono::nanoseconds>(
      now - start).count();
    return now;
  }

  /** Replace the value of the timer with `now - start`. */
  void update_timer(Timer timer, std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    time_of(timer) = std::chrono::duration_cast<std::chrono::nanoseconds>(
      now - start).count();
  }
};

template <class Allocator = battery::standard_allocator>
struct Statistics {
  template <class Alloc> friend struct Statistics;
  using allocator_type = Allocator;
  bool print_statistics; // from config.print_statistics
  size_t variables;
  size_t constraints;
  bool optimization;
  size_t nodes;
  size_t fails;
  size_t solutions;
  int depth_max;
  bool exhaustive;
  size_t eps_num_subproblems;
  size_t eps_solved_subproblems;
  size_t eps_skipped_subproblems;
  size_t num_blocks_done;
  size_t fixpoint_iterations;
  size_t num_deductions;
  TimingStatistics<Allocator> timers;

  CUDA Statistics(size_t variables, size_t constraints, bool optimization, bool print_statistics):
    variables(variables), constraints(constraints), optimization(optimization),
    print_statistics(print_statistics),
    nodes(0), fails(0), solutions(0),
    depth_max(0), exhaustive(true),
    eps_solved_subproblems(0), eps_num_subproblems(1), eps_skipped_subproblems(0),
    num_blocks_done(0), fixpoint_iterations(0), num_deductions(0),
    timers()
  {}

  CUDA Statistics(): Statistics(0,0,false,false) {}
  Statistics(const Statistics&) = default;
  Statistics(Statistics&&) = default;

  template <class Alloc>
  CUDA Statistics(const Statistics<Alloc>& other):
    variables(other.variables), constraints(other.constraints), optimization(other.optimization),
    print_statistics(other.print_statistics),
    nodes(other.nodes), fails(other.fails), solutions(other.solutions),
    depth_max(other.depth_max), exhaustive(other.exhaustive),
    eps_solved_subproblems(other.eps_solved_subproblems), eps_num_subproblems(other.eps_num_subproblems),
    eps_skipped_subproblems(other.eps_skipped_subproblems), num_blocks_done(other.num_blocks_done),
    fixpoint_iterations(other.fixpoint_iterations), num_deductions(other.num_deductions),
    timers(other.timers)
  {}

  template <class Alloc>
  CUDA void meet(const Statistics<Alloc>& other) {
    nodes += other.nodes;
    fails += other.fails;
    solutions += other.solutions;
    depth_max = battery::max(depth_max, other.depth_max);
    exhaustive = exhaustive && other.exhaustive;
    eps_solved_subproblems += other.eps_solved_subproblems;
    eps_skipped_subproblems += other.eps_skipped_subproblems;
    num_blocks_done += other.num_blocks_done;
    fixpoint_iterations += other.fixpoint_iterations;
    num_deductions += other.num_deductions;
    timers.meet(other.timers);
  }

  template <class Alloc>
  CUDA void meet(const TimingStatistics<Alloc>& other) {
    timers.meet(other);
  }

#ifdef __CUDACC__
  __device__ cuda::std::chrono::system_clock::time_point start_timer_device() const {
    return timers.start_timer_device();
  }
#endif

  std::chrono::steady_clock::time_point start_timer_host() const {
    return timers.start_timer_host();
  }

#ifdef __CUDACC__
  __device__ cuda::std::chrono::system_clock::time_point stop_timer(Timer timer, cuda::std::chrono::system_clock::time_point start) {
    return timers.stop_timer(timer, start);
  }
#endif

  std::chrono::steady_clock::time_point stop_timer(Timer timer, std::chrono::steady_clock::time_point start) {
    return timers.stop_timer(timer, start);
  }

  /** Replace the value of the timer with `now - start`. */
  void update_timer(Timer timer, std::chrono::steady_clock::time_point start) {
    timers.update_timer(timer, start);
  }

  CUDA int64_t time_ms_of(Timer timer) const {
    return timers.time_ms_of(timer);
  }

  CUDA void print_stat(const char* name, const char* value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s=\"%s\"\n", name, value);
    }
  }

  CUDA void print_stat(const char* name, size_t value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s=%" PRIu64 "\n", name, value);
    }
  }

  CUDA void print_human_stat(int verbose, const char* name, size_t value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s=%" PRIu64 "\n", name, value);
      if(verbose) {
        printf("%%   [");
        size_t M = 1000 * 1000;
        if(value < M * 1000) {
          printf("%.2fM", static_cast<double>(value) / M);
        }
        else if(value < M * 1000 * 1000) {
          printf("%.2fG", static_cast<double>(value) / (M * 1000));
        }
        else {
          printf("%.2fT", static_cast<double>(value) / (M * 1000 * 1000));
        }
        printf("]\n");
      }
    }
  }

  CUDA void print_stat(const char* name, int value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s=%d\n", name, value);
    }
  }

  CUDA void print_stat_fp_iter(const char* name, size_t num_iterations, size_t value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s_fp_iter_%" PRIu64 "=%" PRIu64 "\n", name, num_iterations, value);
    }
  }

  CUDA void print_stat(const char* name, double value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s=%lf\n", name, value);
    }
  }

  CUDA void print_memory_statistics(int verbose, const char* key, size_t bytes) const {
    print_stat(key, bytes);
    if(verbose) {
      printf("%%   [");
      if(bytes < 1000 * 1000) {
        printf("%.2fKB", static_cast<double>(bytes) / 1000);
      }
      else if(bytes < 1000 * 1000 * 1000) {
        printf("%.2fMB", static_cast<double>(bytes) / (1000 * 1000));
      }
      else {
        printf("%.2fGB", static_cast<double>(bytes) / (1000 * 1000 * 1000));
      }
      printf("]\n");
    }
  }

private:
  CUDA double to_sec(int64_t dur) const {
    return (static_cast<double>(dur / 1000 / 1000) / 1000.);
  }

public:
  CUDA void print_timing_stat(const char* name, Timer timer, size_t or_nodes) const {
    print_stat(name, to_sec(timers.time_of(timer) / or_nodes));
  }

  CUDA void print_timing_stat(const char* name, Timer timer) const {
    print_stat(name, to_sec(timers.time_of(timer)));
  }

  CUDA void print_mzn_statistics(size_t or_nodes = 1, int verbose = 0) const {
    assert(or_nodes != 0);
    print_human_stat(verbose, "nodes", nodes);
    print_stat("failures", fails);
    print_stat("variables", variables);
    print_stat("propagators", constraints);
    print_stat("peakDepth", depth_max);
    print_timing_stat("initTime", Timer::PREPROCESSING, or_nodes);
    print_timing_stat("solveTime", Timer::OVERALL);
    print_stat("num_solutions", solutions);
    print_stat("eps_num_subproblems", eps_num_subproblems);
    print_stat("eps_solved_subproblems", eps_solved_subproblems);
    print_stat("eps_skipped_subproblems", eps_skipped_subproblems);
    print_stat("num_blocks_done", num_blocks_done);
    print_human_stat(verbose, "fixpoint_iterations", fixpoint_iterations);
    print_human_stat(verbose, "num_deductions", num_deductions);

    // Timing statistics
    print_timing_stat("solve_time", Timer::SOLVE, or_nodes);
    print_timing_stat("search_time", Timer::SEARCH, or_nodes);
    // print_timing_stat("split_time", Timer::SPLIT, or_nodes);
    // print_timing_stat("push_time", Timer::PUSH, or_nodes);
    // print_timing_stat("pop_time", Timer::POP, or_nodes);
    print_timing_stat("fixpoint_time", Timer::FIXPOINT, or_nodes);
    print_timing_stat("transfer_cpu2gpu_time", Timer::TRANSFER_CPU2GPU, or_nodes);
    print_timing_stat("transfer_gpu2cpu_time", Timer::TRANSFER_GPU2CPU, or_nodes);
    print_timing_stat("select_fp_functions_time", Timer::SELECT_FP_FUNCTIONS, or_nodes);
    print_timing_stat("wait_cpu_time", Timer::WAIT_CPU, or_nodes);
    print_timing_stat("dive_time", Timer::DIVE, or_nodes);
  }

  CUDA void print_mzn_end_stats() const {
    if(!print_statistics) { return; }
    printf("%%%%%%mzn-stat-end\n");
  }

  CUDA void print_mzn_objective(const auto& obj, bool is_minimization) const {
    if(!print_statistics) { return; }
    printf("%%%%%%mzn-stat: objective=");
    if(is_minimization) {
      obj.lb().template deinterpret<lala::TFormula<battery::standard_allocator>>().print(false);
    }
    else {
      obj.ub().template deinterpret<lala::TFormula<battery::standard_allocator>>().print(false);
    }
    printf("\n");
  }

  CUDA void print_mzn_separator() const {
    printf("----------\n");
  }

  CUDA void print_mzn_final_separator() const {
    if(solutions > 0) {
      if(exhaustive) {
        printf("==========\n");
      }
    }
    else {
      assert(solutions == 0);
      if(exhaustive) {
        printf("=====UNSATISFIABLE=====\n");
      }
      else if(optimization) {
        printf("=====UNBOUNDED=====\n");
      }
      else {
        printf("=====UNKNOWN=====\n");
      }
    }
  }
};

#endif
