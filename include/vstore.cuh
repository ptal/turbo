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

#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include "cuda_helper.hpp"

// A variable with a negative index represents the negation `-x`.
// The conversion is automatically handled in `VStore::operator[]`.
typedef int Var;
typedef const char** Var2Name;

struct Interval {
  int lb;
  int ub;

  CUDA Interval(): lb(limit_min()), ub(limit_max()) {}

  CUDA Interval(int lb, int ub): lb(lb), ub(ub) {}

  CUDA Interval join(Interval b) {
    lb = ::max<int>(lb, b.lb);
    ub = ::min<int>(ub, b.ub);
    return *this;
  }

  CUDA bool is_assigned() {
    return lb == ub;
  }

  CUDA Interval neg() {
    return {-ub, -lb};
  }

  CUDA bool operator==(int x) {
    return lb == x && ub == x;
  }

  CUDA bool operator!=(const Interval& other) {
    return lb != other.lb || ub != other.ub;
  }

  CUDA void print() {
    printf("[%d..%d]", lb, ub);
  }
};

class VStore {
  Interval* data;
  size_t n;

public:

  CUDA VStore() : data(nullptr), n(0) {}

  VStore(int nvar) {
    n = nvar;
    CUDIE(cudaMallocManaged(&data, sizeof(*data) * n));
  }

  VStore(const VStore& other) {
    n = other.n;
    CUDIE(cudaMallocManaged(&data, sizeof(*data) * n));
    for(int i = 0; i < n; ++i) {
      data[i] = other.data[i];
    }
  }

  __device__ VStore& operator=(const VStore& other) {
    if (this != &other) {
      // Memory optimisation to reuse memory if already allocated.
      if (this->n != n) {
        n = other.n;
        data = new Interval[n];
      }
      for(int i = 0; i < n; ++i) {
        data[i] = other.data[i];
      }
    }
    return *this;
  }

  VStore(VStore&& other) :
    data(other.data), n(n) {
      other.data = nullptr;
      other.n = 0;
  }

  // Free the data array on host.
  __host__ void free() {
    CUDIE(cudaFree(data));
    n = 0;
  }

  __device__ ~VStore() {
    delete[] data;
    n = 0;
  }

  CUDA bool all_assigned() {
    for(int i = 0; i < n; ++i) {
      if(!data[i].is_assigned()) {
        return false;
      }
    }
    return true;
  }

  CUDA static void print_var(Var x, Var2Name var2name) {
    printf("%s%s", (x < 0 ? "-" : ""), var2name[abs(x)]);
  }

  CUDA void print(Var2Name var2name) {
    for(int i=0; i < n; ++i) {
      print_var(i, var2name);
      printf(" = ");
      data[i].print();
      printf("\n");
    }
  }

  // lb <= x <= ub
  CUDA void dom(Var x, Interval itv) {
    data[x] = itv;
  }

  // Precondition: i >= 0
  CUDA bool update_raw(int i, Interval itv) {
    bool has_changed = data[i] != itv;
    if (has_changed) {
      Interval it = data[i];
      data[i] = itv.join(data[i]);
      printf("Update %d with %d..%d (old = %d..%d, new = %d..%d)\n",
        i, itv.lb, itv.ub, it.lb, it.ub, data[i].lb, data[i].ub);
    }
    return has_changed;
  }

  CUDA bool update(int i, Interval itv) {
    if (i<0) {
      return update_raw(-i, itv.neg());
    } else {
      return update_raw(i, itv);
    }
  }

  CUDA Interval operator[](int i) {
    return i < 0 ? data[-i].neg() : data[i];
  }

  CUDA size_t size() { return n; }
};

#endif
