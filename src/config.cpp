// Copyright 2022 Pierre Talbot

#include "config.hpp"
#include "statistics.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>

Configuration::Configuration(): timeout(std::numeric_limits<int>::max()), arch(CPU),
  and_nodes(AND_NODES), or_nodes(OR_NODES), subproblems_power(SUBPROBLEMS_POWER) {}

void usage_and_exit(char** argv) {
  std::cout << "usage: " << argv[0] << " [timeout in seconds] [-arch <cpu|gpu>] [-only_csv_header] [-or 48] [-and 256] [-sub 12] xcsp3instance.xml" << std::endl;
  std::cout << "\tGiven -sub N, we generate 2^N subproblems." << std::endl;
  exit(EXIT_FAILURE);
}

// https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
class InputParser {
private:
  std::vector <std::string> tokens;
public:
  InputParser (int &argc, char **argv) {
    for (int i=1; i < argc; ++i) {
      tokens.push_back(std::string(argv[i]));
    }
  }

  const std::string& getCmdOption(const std::string &option) const {
    std::vector<std::string>::const_iterator itr = std::find(tokens.begin(), tokens.end(), option);
    if (itr != tokens.end() && ++itr != tokens.end()){
      return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
  }

  bool cmdOptionExists(const std::string &option) const {
    return std::find(tokens.begin(), tokens.end(), option) != tokens.end();
  }
};

Configuration parse_args(int argc, char** argv) {
  int num_params = 0;
  Configuration config;
  InputParser input(argc, argv);

  if(input.cmdOptionExists("-only_csv_header")) {
    GlobalStatistics::print_csv_header();
    exit(EXIT_SUCCESS);
  }

  const std::string& or_nodes = input.getCmdOption("-or");
  const std::string& and_nodes = input.getCmdOption("-and");
  const std::string& subproblems_power = input.getCmdOption("-sub");
  const std::string& architecture = input.getCmdOption("-arch");

  if(!or_nodes.empty()) {
    config.or_nodes = std::stoi(or_nodes);
    ++num_params;
  }
  if(!and_nodes.empty()) {
    config.and_nodes = std::stoi(and_nodes);
    ++num_params;
  }
  if(!subproblems_power.empty()) {
    config.subproblems_power = std::stoi(subproblems_power);
    ++num_params;
  }
  if(!architecture.empty()) {
    ++num_params;
    if(architecture == "cpu") {
      config.arch = CPU;
    }
    else if(architecture == "gpu") {
      config.arch = GPU;
    }
    else {
      printf("unknown architecture -arch %s\n", architecture);
      usage_and_exit(argv);
    }
  }

  if(argc < num_params + 2) {
    usage_and_exit(argv);
  }
  if(argc >= num_params + 2) {
    config.problem_path = std::string(argv[argc-1]);
  }
  if(argc >= num_params + 3) {
    config.timeout = std::atoi(argv[1]);
  }
  return config;
}
