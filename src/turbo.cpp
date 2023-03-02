// Copyright 2022 Pierre Talbot

#include <iostream>
#include "cpu_solving.hpp"
#include "gpu_solving.hpp"

int main(int argc, char** argv) {
  try
  {
    Configuration config = parse_args(argc, argv);
    GlobalStatistics stats(10, 10, false);
    printf("%%");
    config.print_commandline(argv[0]);
    if(config.arch == CPU) {
      cpu_solve(config, stats);
    }
    else if(config.arch == GPU) {
      gpu_solve(config, stats);
    }
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
