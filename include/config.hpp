// Copyright 2022 Pierre Talbot

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "allocator.hpp"
#include "string.hpp"

#define OR_NODES 48
#define AND_NODES 256
#define SUBPROBLEMS_POWER 12 // 2^N

enum Arch {
  CPU,
  GPU
};

template<class Allocator>
struct Configuration {
  bool print_intermediate_solutions; // (only optimization problems).
  int stop_after_n_solutions; // 0 for all solutions (satisfaction problems only).
  bool free_search;
  bool print_statistics;
  bool verbose_solving;
  int timeout_ms;
  int or_nodes;
  int and_nodes; // (only for GPU)
  int subproblems_power;
  Arch arch;
  battery::string<Allocator> problem_path;

  CUDA Configuration():
    print_intermediate_solutions(false),
    stop_after_n_solutions(1),
    free_search(false),
    verbose_solving(false),
    print_statistics(false),
    timeout_ms(0),
    and_nodes(AND_NODES),
    or_nodes(OR_NODES),
    subproblems_power(SUBPROBLEMS_POWER),
    arch(CPU)
  {}

  Configuration(Configuration&&) = default;
  Configuration(const Configuration&) = default;

  template<class Alloc>
  CUDA Configuration(const Configuration<Alloc>& other) :
    print_intermediate_solutions(other.print_intermediate_solutions),
    stop_after_n_solutions(other.stop_after_n_solutions),
    free_search(other.free_search),
    print_statistics(other.print_statistics),
    verbose_solving(other.verbose_solving),
    timeout_ms(other.timeout_ms),
    or_nodes(other.or_nodes),
    and_nodes(other.and_nodes),
    subproblems_power(other.subproblems_power),
    arch(other.arch),
    problem_path(other.problem_path)
  {}

  CUDA void print_commandline(const char* program_name) {
    printf("%s -t %d %s-n %d %s%s%s%s",
      program_name,
      timeout_ms,
      (print_intermediate_solutions ? "-a ": ""),
      stop_after_n_solutions,
      (print_intermediate_solutions ? "-i ": ""),
      (free_search ? "-f " : ""),
      (print_statistics ? "-s " : ""),
      (verbose_solving ? "-v " : "")
    );
    if(arch == GPU) {
      printf("-or %d -and %d -sub %d ", or_nodes, and_nodes, subproblems_power);
    }
    else {
      printf("-p %d ", or_nodes);
    }
    printf("%s\n", problem_path.data());
  }
};

void usage_and_exit(const std::string& program_name);
Configuration<battery::StandardAllocator> parse_args(int argc, char** argv);

#endif
