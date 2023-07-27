// Copyright 2022 Pierre Talbot

#include <iostream>
#include "cpu_solving.hpp"
#include "gpu_solving.hpp"

using namespace battery;

int main(int argc, char** argv) {
  try
  {
    Configuration<standard_allocator> config = parse_args(argc, argv);
    if(config.verbose_solving) {
      printf("%% ");
      config.print_commandline(argv[0]);
    }
    if(config.arch == CPU) {
      cpu_solve(config);
    }
    else if(config.arch == GPU) {
      gpu_solve(config);
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
