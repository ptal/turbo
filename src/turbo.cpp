// Copyright 2022 Pierre Talbot

#include <iostream>
#include "cpu_solving.hpp"

#ifdef REDUCE_PTX_SIZE
#ifndef DISABLE_FULL_GPU_SOLVING
#include "gpu_dive_and_solve.hpp"
#endif
#endif

#include "hybrid_dive_and_solve.hpp"

using namespace battery;

int main(int argc, char** argv) {
  try
  {
    Configuration<standard_allocator> config = parse_args(argc, argv);
    if(config.verbose_solving) {
      printf("%% ");
      config.print_commandline(argv[0]);
    }
    if(config.arch == Arch::CPU) {
      cpu_solve(config);
    }
#ifdef REDUCE_PTX_SIZE
#ifndef DISABLE_FULL_GPU_SOLVING
    else if(config.arch == Arch::GPU) {
      gpu_dive_and_solve(config);
    }
#endif
#endif
    else if(config.arch == Arch::HYBRID) {
      hybrid_dive_and_solve(config);
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
