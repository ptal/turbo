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
#include <cassert>
#include <vector>
#include <string>
#include "cuda_helper.hpp"

// A variable with a negative index represents the negation `-x`.
// The conversion is automatically handled in `VStore::view_of`.
typedef int Var;

struct Interval {
  int lb;
  int ub;

  CUDA Interval(): lb(limit_min()), ub(limit_max()) {}

  CUDA Interval(int lb, int ub): lb(lb), ub(ub) {}

  CUDA void inplace_join(const Interval &b) {
    lb = ::max<int>(lb, b.lb);
    ub = ::min<int>(ub, b.ub);
  }

  CUDA bool is_assigned() const {
    return lb == ub;
  }

  CUDA bool is_top() const {
    return lb > ub;
  }

  CUDA Interval neg() const {
    return {-ub, -lb};
  }

  CUDA bool operator==(int x) const {
    return lb == x && ub == x;
  }

  CUDA bool operator!=(const Interval& other) const {
    return lb != other.lb || ub != other.ub;
  }

  CUDA void print() const {
    printf("[%d..%d]", lb, ub);
  }
};

class VStore {
  Interval* data;
  size_t n;

  // The names don't change during solving. We want to avoid useless copies.
  // Unfortunately, static member are not supported in CUDA, so we use an instance variable which is never copied.
  char** names;
  size_t names_len;

public:

  void init_names(std::vector<std::string>& vnames) {
    names_len = vnames.size();
    CUDIE(cudaMallocManaged(&names, sizeof(*names) * names_len));
    for(int i=0; i < names_len; ++i) {
      int len = vnames[i].size();
      CUDIE(cudaMallocManaged(&names[i], sizeof(char) * (len + 1)));
      for(int j=0; j < len; ++j) {
        names[i][j] = vnames[i][j];
      }
      names[i][len] = '\0';
    }
  }

  void free_names() {
    for(int i = 0; i < names_len; ++i) {
      CUDIE(cudaFree(names[i]));
    }
    CUDIE(cudaFree(names));
  }

  VStore(int nvar) {
    n = nvar;
    CUDIE(cudaMallocManaged(&data, sizeof(*data) * n));
  }

  CUDA VStore(const VStore& other) {
    n = other.n;
    names = other.names;
    names_len = other.names_len;
    MALLOC_CHECK(cudaMalloc(&data, sizeof(*data) * n));
    for(int i = 0; i < n; ++i) {
      data[i] = other.data[i];
    }
  }

  CUDA void reset(const VStore& other) {
    assert(n == other.n);
    for(int i = 0; i < n; ++i) {
      data[i] = other.data[i];
    }
  }

  CUDA ~VStore() {
    cudaFree(data);
  }

  CUDA bool all_assigned() const {
    for(int i = 0; i < n; ++i) {
      if(!data[i].is_assigned()) {
        return false;
      }
    }
    return true;
  }

  CUDA bool is_top() const {
    for(int i = 0; i < n; ++i) {
      if(data[i].is_top()) {
        return true;
      }
    }
    return false;
  }

  CUDA bool is_top(Var x) const {
    return view_of(x).is_top();
  }

  CUDA const char* name_of(Var x) const {
    return names[abs(x)];
  }

  CUDA void print_var(Var x) const {
    printf("%s%s", (x < 0 ? "-" : ""), names[abs(x)]);
  }

  CUDA void print_view(Var* vars) const {
    for(int i=0; vars[i] != -1; ++i) {
      print_var(vars[i]);
      printf(" = ");
      data[vars[i]].print();
      printf("\n");
    }
  }

  CUDA void print() const {
    // The first variable is the fake one, c.f. `ModelBuilder` constructor.
    for(int i=1; i < n; ++i) {
      print_var(i);
      printf(" = ");
      data[i].print();
      printf("\n");
    }
  }

  // lb <= x <= ub
  CUDA void dom(Var x, Interval itv) {
    data[x] = itv;
  }

  CUDA bool update_lb(Var i, int lb) {
    if(i >= 0) {
      if (data[i].lb < lb) {
        LOG(printf("Update LB(%s) with %d (old = %d) in %p\n", names[i], lb, data[i].lb, this));
        data[i].lb = lb;
        return true;
      }
    }
    else {
      if (data[-i].ub > -lb) {
        LOG(printf("Update UB(%s) with %d (old = %d) in %p\n", names[-i], -lb, data[-i].ub, this));
        data[-i].ub = -lb;
        return true;
      }
    }
    return false;
  }

  CUDA bool update_ub(Var i, int ub) {
    if(i >= 0) {
      if (data[i].ub > ub) {
        LOG(printf("Update UB(%s) with %d (old = %d) in %p\n", names[i], ub, data[i].ub, this));
        data[i].ub = ub;
        return true;
      }
    }
    else {
      if (data[-i].lb < -ub) {
        LOG(printf("Update LB(%s) with %d (old = %d) in %p\n", names[-i], -ub, data[-i].lb, this));
        data[-i].lb = -ub;
        return true;
      }
    }
    return false;
  }

  CUDA bool update(Var i, Interval itv) {
    bool has_changed = update_lb(i, itv.lb);
    has_changed |= update_ub(i, itv.ub);
    return has_changed;
  }

  CUDA bool assign(Var i, int v) {
    return update(i, {v, v});
  }

  CUDA Interval view_of(Var i) const {
    return i < 0 ? data[-i].neg() : data[i];
  }

  CUDA int lb(Var i) const {
    return view_of(i).lb;
  }

  CUDA int ub(Var i) const {
    return view_of(i).ub;
  }

  CUDA size_t size() const { return n; }
};

#endif
