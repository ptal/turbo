// Copyright 2022 Pierre Talbot

#ifndef TURBO_CONFIG_HPP
#define TURBO_CONFIG_HPP

#include "battery/allocator.hpp"
#include "battery/string.hpp"

#define OR_NODES 48
#define AND_NODES 256
#define SUBPROBLEMS_POWER 15 // 2^N
#define STACK_KB 100

enum Arch {
  CPU,
  GPU
};

template<class Allocator>
struct Configuration {
  using allocator_type = Allocator;
  bool print_intermediate_solutions; // (only optimization problems).
  int stop_after_n_solutions; // 0 for all solutions (satisfaction problems only).
  bool free_search;
  bool print_statistics;
  bool verbose_solving;
  bool print_ast;
  int timeout_ms;
  int or_nodes;
  int and_nodes; // (only for GPU)
  int subproblems_power;
  int heap_mb;
  int stack_kb;
  Arch arch;
  battery::string<allocator_type> problem_path;

  CUDA Configuration():
    print_intermediate_solutions(false),
    stop_after_n_solutions(1),
    free_search(false),
    verbose_solving(false),
    print_ast(false),
    print_statistics(false),
    timeout_ms(0),
    and_nodes(AND_NODES),
    or_nodes(OR_NODES),
    subproblems_power(SUBPROBLEMS_POWER),
    heap_mb(0),
    stack_kb(STACK_KB),
    arch(
      #ifdef __NVCC__
        GPU
      #else
        CPU
      #endif
    )
  {}

  Configuration(Configuration&&) = default;
  Configuration(const Configuration&) = default;

  template<class Alloc>
  CUDA Configuration(const Configuration<Alloc>& other, const allocator_type& alloc = allocator_type()) :
    print_intermediate_solutions(other.print_intermediate_solutions),
    stop_after_n_solutions(other.stop_after_n_solutions),
    free_search(other.free_search),
    print_statistics(other.print_statistics),
    verbose_solving(other.verbose_solving),
    print_ast(other.print_ast),
    timeout_ms(other.timeout_ms),
    or_nodes(other.or_nodes),
    and_nodes(other.and_nodes),
    subproblems_power(other.subproblems_power),
    heap_mb(other.heap_mb),
    stack_kb(other.stack_kb),
    arch(other.arch),
    problem_path(other.problem_path, alloc)
  {}

  CUDA void print_commandline(const char* program_name) {
    printf("%s -t %d %s-n %d %s%s%s%s%s",
      program_name,
      timeout_ms,
      (print_intermediate_solutions ? "-a ": ""),
      stop_after_n_solutions,
      (print_intermediate_solutions ? "-i ": ""),
      (free_search ? "-f " : ""),
      (print_statistics ? "-s " : ""),
      (verbose_solving ? "-v " : ""),
      (print_ast ? "-ast " : "")
    );
    if(arch == GPU) {
      printf("-arch gpu -or %d -and %d -sub %d -heap %d -stack %d ", or_nodes, and_nodes, subproblems_power, heap_mb, stack_kb);
    }
    else {
      printf("-arch cpu -p %d ", or_nodes);
    }
    printf("%s\n", problem_path.data());
  }
};

void usage_and_exit(const std::string& program_name);
Configuration<battery::standard_allocator> parse_args(int argc, char** argv);

#endif
