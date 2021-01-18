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

#ifndef VSTORE_HPP
#define VSTORE_HPP

#include <limits>
#include "cuda_helper.hpp"

// A variable with a negative index represents the negation `-x`.
// The conversion is automatically handled in `VStore::operator[]`.
typedef size_t Var;

struct Interval {
  int lb;
  int ub;

  Interval():
    lb(std::numeric_limits<int>::min()),
    ub(std::numeric_limits<int>::max()) {}

  CUDA Interval(int lb, int ub): lb(lb), ub(ub) {}

  CUDA Interval join(Interval b) {
    lb = ::max<int>(lb, b.lb);
    ub = ::min<int>(ub, b.ub);
    return *this;
  }

  CUDA Interval neg() {
    return {-ub, -lb};
  }

  CUDA bool operator==(int x) {
    return lb == x && ub == x;
  }
};

struct VStore {
  Interval* data;
  size_t size;

  VStore(int nvar) {
    size = nvar;
    CUDIE(cudaMallocManaged(&data, sizeof(*data) * nvar));
  }

  void free() {
    CUDIE(cudaFree(data));
    size = 0;
  }

  ~VStore() {
    free();
  }

  CUDA void print_store() {
    for(int i=0; i < size; ++i) {
      printf("%d = [%d..%d]\n", i, data[i].lb, data[i].ub);
    }
  }

  // lb <= x <= ub
  CUDA void dom(Var x, Interval itv) {
    data[x] = itv;
  }

  CUDA void update(int i, Interval itv) {
    if (i<0) {
      data[-i].lb = -itv.ub;
      data[-i].ub = -itv.lb;
    } else {
      data[i] = itv;
    }
  }

  CUDA Interval operator[](int i) {
    return i < 0 ? data[-i].neg() : data[i];
  }
};

#endif
