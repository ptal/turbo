// Copyright 2022 Pierre Talbot

#include "config.hpp"
#include "statistics.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>

void usage_and_exit(const std::string& program_name) {
  std::cout << "usage: " << program_name << " [-t 2000] [-a] [-n 10] [-i] [-f] [-s] [-v] [-p <i>] [-arch <cpu|gpu>] [-p 48] [-or 48] [-and 256] [-sub 12] [xcsp3instance.xml | fzninstance.fzn]" << std::endl;
  std::cout << "\t-t 2000: Run the solver with a timeout of 2000 milliseconds." << std::endl;
  std::cout << "\t-a: Instructs the solver to report all solutions in the case of satisfaction problems, or print intermediate solutions of increasing quality in the case of optimisation problems." << std::endl;
  std::cout << "\t-n 10: Instructs the solver to stop after reporting 10 solutions (only used with satisfaction problems)." << std::endl;
  std::cout << "\t-i: Instructs the solver to print intermediate solutions of increasing quality (only used with optimisation problems)." << std::endl;
  std::cout << "\t-f: Instructs the solver to conduct a “free search”, i.e., ignore any search annotations. The solver is not required to ignore the annotations, but it is allowed to do so." << std::endl;
  std::cout << "\t-s: Print statistics during and after the search for solutions." << std::endl;
  std::cout << "\t-v: Print log messages (verbose solving) to the standard error stream." << std::endl;
  std::cout << "\t-p 48: On CPU, run with 48 parallel threads. On GPU, equivalent to `-or 48`." << std::endl;
  std::cout << "\t-arch <cpu|gpu>: Choose the architecture on which the problem will be solved." << std::endl;
  std::cout << "\t-or 48: Run the subproblems on 48 streaming multiprocessors (SMs) (only for GPU architecture)." << std::endl;
  std::cout << "\t-and 256: Run each subproblem in 256 threads within a SM (only for GPU architecture)." << std::endl;
  std::cout << "\t-sub 12: Create 2^12 subproblems to be solved in turns by the 'OR threads' (embarrasingly parallel search)." << std::endl;
  exit(EXIT_FAILURE);
}

// https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
class InputParser {
private:
  std::string program_name;
  std::vector <std::string> tokens;
  int tokens_read;
public:
  InputParser (int argc, char **argv): tokens_read(0), program_name(argv[0]) {
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

  bool read_int(const std::string& option, int& result) {
    const std::string& value = getCmdOption(option);
    if(!value.empty()) {
      result = std::stoi(value);
      tokens_read += 2;
      return true;
    }
    return false;
  }

  bool read_bool(const std::string& option, bool& result) {
    result = cmdOptionExists(option);
    if(result) {
      ++tokens_read;
      return true;
    }
    return false;
  }

  bool read_string(const std::string& option, std::string& result) {
    result = getCmdOption(option);
    if(!result.empty()) {
      tokens_read += 2;
      return true;
    }
    return false;
  }

  void read_input_file(std::string& result) {
    if(tokens.size() <= tokens_read) {
      usage_and_exit(program_name);
    }
    result = tokens.back();
  }
};

Configuration parse_args(int argc, char** argv) {
  Configuration config;
  InputParser input(argc, argv);

  if(input.cmdOptionExists("-or") && input.cmdOptionExists("-p")) {
    std::cerr << "The options -or and -p cannot be used at the same time" << std::endl;
    usage_and_exit(argv[0]);
  }
  input.read_int("-p", config.or_nodes);
  input.read_int("-or", config.or_nodes);
  input.read_int("-and", config.and_nodes);
  input.read_int("-sub", config.subproblems_power);
  input.read_int("-t", config.timeout_ms);
  bool all_sols;
  input.read_bool("-a", all_sols);
  if(all_sols) {
    config.stop_after_n_solutions = 0;
  }
  input.read_int("-n", config.stop_after_n_solutions);
  input.read_bool("-i", config.print_intermediate_solutions);
  input.read_bool("-f", config.free_search);
  input.read_bool("-v", config.verbose_solving);
  input.read_bool("-s", config.print_statistics);

  std::string architecture;
  if(input.read_string("-arch", architecture)) {
    if(architecture == "cpu") {
      config.arch = CPU;
    }
    else if(architecture == "gpu") {
      config.arch = GPU;
    }
    else {
      std::cerr << "unknown architecture -arch " << architecture << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  input.read_input_file(config.problem_path);
  return config;
}
