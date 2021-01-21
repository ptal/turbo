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

#include "solver.cuh"
#include "constraints.cuh"

#include "XCSP3CoreParser.h"

#include "XCSP3_turbo_callbacks.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "usage: " << argv[0] << " xcsp3instance.xml" <<std::endl;
    exit(EXIT_FAILURE);
  }

  try
  {
    ModelBuilder* model_builder = new ModelBuilder();
    XCSP3_turbo_callbacks cb(model_builder);
    XCSP3CoreParser parser(&cb);
    parser.parse(argv[1]); // fileName is a string
    std::vector<std::string>& var2name = model_builder->name_of_vars();
    Constraints constraints = model_builder->build_constraints();
    VStore* vstore = model_builder->build_store();
    const char** var2name_raw = new const char*[var2name.size()];
    for(int i = 0; i < var2name.size(); ++i) {
      var2name_raw[i] = var2name[i].c_str();
    }
    vstore->print(var2name_raw);
    constraints.print(var2name_raw);
    solve(vstore, constraints, var2name_raw);
    delete[] var2name_raw;
    vstore->free();
    CUDIE(cudaFree(vstore));
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
