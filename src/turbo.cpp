// Copyright 2022 Pierre Talbot

#include <iostream>
#include "barebones_dive_and_solve.hpp"

using namespace battery;

int main(int argc, char** argv) {
  try
  {
    Configuration<standard_allocator> config = parse_args(argc, argv);
    if(config.print_statistics) {
      printf("%%%%%%mzn-stat: command_line=\"");
      config.print_commandline(argv[0]);
      printf("\"\n");
    }
    barebones::barebones_dive_and_solve(config);
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
