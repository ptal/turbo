// Copyright 2022 Pierre Talbot

#ifndef TURBO_CONFIG_HPP
#define TURBO_CONFIG_HPP

#include "battery/allocator.hpp"
#include "battery/string.hpp"

#ifdef __CUDACC__
  #include <cuda.h>
#endif

#define SUBPROBLEMS_POWER 12 // 2^N
#define STACK_KB 32

enum class Arch {
  CPU,
  GPU
};

enum class InputFormat {
  XCSP3,
  FLATZINC
};

template<class Allocator>
struct Configuration {
  using allocator_type = Allocator;
  bool print_intermediate_solutions; // (only optimization problems).
  size_t stop_after_n_solutions; // 0 for all solutions (satisfaction problems only).
  bool free_search;
  bool print_statistics;
  bool verbose_solving;
  bool print_ast;
  size_t timeout_ms;
  size_t or_nodes;
  size_t and_nodes; // (only for GPU)
  size_t subproblems_power;
  size_t stack_kb;
  Arch arch;
  battery::string<allocator_type> problem_path;
  battery::string<allocator_type> version;
  battery::string<allocator_type> hardware;

  CUDA Configuration(const allocator_type& alloc = allocator_type{}):
    print_intermediate_solutions(false),
    stop_after_n_solutions(1),
    free_search(false),
    verbose_solving(false),
    print_ast(false),
    print_statistics(false),
    timeout_ms(0),
    and_nodes(0),
    or_nodes(0),
    subproblems_power(SUBPROBLEMS_POWER),
    stack_kb(STACK_KB),
    arch(
      #ifdef __CUDACC__
        Arch::GPU
      #else
        Arch::CPU
      #endif
    ),
    problem_path(alloc),
    version(alloc),
    hardware(alloc)
  {}

  Configuration(Configuration<allocator_type>&&) = default;
  Configuration(const Configuration<allocator_type>&) = default;

  template<class Alloc>
  CUDA Configuration(const Configuration<Alloc>& other, const allocator_type& alloc = allocator_type{}) :
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
    stack_kb(other.stack_kb),
    arch(other.arch),
    problem_path(other.problem_path, alloc),
    version(other.version, alloc),
    hardware(other.hardware, alloc)
  {}

  template <class Alloc2>
  CUDA Configuration<allocator_type>& operator=(const Configuration<Alloc2>& other) {
    print_intermediate_solutions = other.print_intermediate_solutions;
    stop_after_n_solutions = other.stop_after_n_solutions;
    free_search = other.free_search;
    verbose_solving = other.verbose_solving;
    print_ast = other.print_ast;
    print_statistics = other.print_statistics;
    timeout_ms = other.timeout_ms;
    and_nodes = other.and_nodes;
    or_nodes = other.or_nodes;
    subproblems_power = other.subproblems_power;
    stack_kb = other.stack_kb;
    arch = other.arch;
    problem_path = other.problem_path;
    version = other.version;
    hardware = other.hardware;
  }

  CUDA void print_commandline(const char* program_name) {
    printf("%s -t %lu %s-n %lu %s%s%s%s%s",
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
    if(arch == Arch::GPU) {
      printf("-arch gpu -or %lu -and %lu -sub %lu -stack %lu ", or_nodes, and_nodes, subproblems_power, stack_kb);
    }
    else {
      printf("-arch cpu -p %zu ", or_nodes);
    }
    if(version.size() != 0) {
      printf("-version %s ", version.data());
    }
    if(hardware.size() != 0) {
      printf("-hardware \"%s\" ", hardware.data());
    }
    printf("%s\n", problem_path.data());
  }

  CUDA void print_mzn_statistics() const {
    printf("%%%%%%mzn-stat: problem_path=\"%s\"\n", problem_path.data());
    printf("%%%%%%mzn-stat: solver=\"Turbo\"\n");
    printf("%%%%%%mzn-stat: version=\"%s\"\n", (version.size() == 0) ? "1.1.2" : version.data());
    printf("%%%%%%mzn-stat: hardware=\"%s\"\n", (hardware.size() == 0) ? "Intel Core i9-10900X@3.7GHz;24GO DDR4;NVIDIA RTX A5000" : hardware.data());
    printf("%%%%%%mzn-stat: arch=\"%s\"\n", arch == Arch::GPU ? "gpu" : "cpu");
    printf("%%%%%%mzn-stat: free_search=\"%s\"\n", free_search ? "yes" : "no");
    printf("%%%%%%mzn-stat: or_nodes=%lu\n", or_nodes);
    printf("%%%%%%mzn-stat: timeout_ms=%lu\n", timeout_ms);
    if(arch == Arch::GPU) {
      printf("%%%%%%mzn-stat: and_nodes=%lu\n", and_nodes);
      printf("%%%%%%mzn-stat: stack_size=%lu\n", stack_kb * 1000);
      #ifdef CUDA_VERSION
        printf("%%%%%%mzn-stat: cuda_version=%d\n", CUDA_VERSION);
      #endif
      #ifdef __CUDA_ARCH__
        printf("%%%%%%mzn-stat: cuda_architecture=%d\n", __CUDA_ARCH__);
      #endif
    }
  }

  CUDA InputFormat input_format() const {
    if(problem_path.ends_with(".fzn")) {
      return InputFormat::FLATZINC;
    }
    else if(problem_path.ends_with(".xml")) {
      return InputFormat::XCSP3;
    }
    else {
      printf("ERROR: Unknown input format for the file %s [supported extension: .xml and .fzn].\n", problem_path.data());
      exit(EXIT_FAILURE);
    }
  }
};

void usage_and_exit(const std::string& program_name);
Configuration<battery::standard_allocator> parse_args(int argc, char** argv);

#endif
