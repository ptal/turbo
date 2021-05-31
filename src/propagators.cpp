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

#include "propagators.hpp"

CUDA_GLOBAL void init_logical_or(Propagator** p, int uid, Propagator* left, Propagator* right) {
  *p = new LogicalOr(left, right);
  (*p)->uid = uid;
}

CUDA_GLOBAL void init_logical_and(Propagator** p, int uid, Propagator* left, Propagator* right) {
  *p = new LogicalAnd(left, right);
  (*p)->uid = uid;
}

CUDA_GLOBAL void init_reified_prop(Propagator** p, int uid, Var b, Propagator* rhs, Propagator* not_rhs) {
  *p = new ReifiedProp(b, rhs, not_rhs);
  (*p)->uid = uid;
}

CUDA_GLOBAL void init_linear_ineq(Propagator** p, int uid, const Array<Var> vars, const Array<int> constants, int max) {
  *p = new LinearIneq(vars, constants, max, global_allocator);
  (*p)->uid = uid;
}
