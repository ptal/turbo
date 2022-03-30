// Copyright 2022 Pierre Talbot

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>

#define OR_NODES 48
#define AND_NODES 256
#define SUBPROBLEMS_POWER 12 // 2^N

struct Configuration {
  int timeout;
  int and_nodes;
  int or_nodes;
  int subproblems_power;
  std::string problem_path;

  Configuration();
};

void usage_and_exit(char** argv);
Configuration parse_args(int argc, char** argv);

#endif
