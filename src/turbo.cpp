// Copyright 2022 Pierre Talbot

#include <iostream>
#include "XCSP3_parser.hpp"
#include "config.hpp"

using namespace lala;

int main(int argc, char** argv) {
  Configuration config = parse_args(argc, argv);
  try
  {
    auto sf = parse_xcsp3<StandardAllocator>(config.problem_path, 0, 1);
    sf.formula().print(false);
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
