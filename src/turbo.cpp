// Copyright 2021 Pierre Talbot, Frédéric Pinel

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <cstdlib>

#include "solver.hpp"
#include "propagators.hpp"

#include "XCSP3CoreParser.h"

#include "XCSP3_turbo_callbacks.hpp"

void usage_and_exit(char** argv) {
  std::cout << "usage: " << argv[0] << " [timeout in seconds] [-or 48] [-and 256] [-sub 12] xcsp3instance.xml" << std::endl;
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
    std::vector<std::string>::const_iterator itr;
    itr = std::find(tokens.begin(), tokens.end(), option);
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
  const std::string& or_nodes = input.getCmdOption("-or");
  const std::string& and_nodes = input.getCmdOption("-and");
  const std::string& subproblems_power = input.getCmdOption("-sub");

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

int main(int argc, char** argv) {
  Configuration config = parse_args(argc, argv);
  try
  {
    ModelBuilder* model_builder = new ModelBuilder();
    XCSP3_turbo_callbacks cb(model_builder);
    XCSP3CoreParser parser(&cb);
    parser.parse(config.problem_path.c_str()); // fileName is a string
    Constraints constraints = model_builder->build_constraints();
    VStore* vstore = model_builder->build_store();
    Var minimize_x = model_builder->build_minimize_obj();
    solve(vstore, constraints, minimize_x, config);
    vstore->free_names();
    vstore->~VStore();
    free2(vstore);
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
