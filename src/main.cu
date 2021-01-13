#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cmath.h>

#define CUDIE(result) { \
        cudaError_t e = (result); \
        if (e != cudaSuccess) { \
                std::cerr << __FILE__ << ":" << __LINE__; \
                std::cerr << " CUDA runtime error: " << cudaGetErrorString(e) << '\n'; \
                exit((int)e); \
        }}

#define CUDIE0() CUDIE(cudaGetLastError())

// A variable with a negative index represents the negation `-x`.
// The conversion is automatically handled in `VStore::operator[]`.
typedef size_t Var;

struct Interval {
  int lb;
  int ub;

  inline void join(Interval b) {
    lb = max(lb, b.lb);
    ub = min(ub, b.ub);
  }

  Interval neg() {
    return {-ub, -lb};
  }

  bool operator==(int x) {
    return lb == x && ub == x;
  }
};

struct VStore {
  Interval* data;
  size_t size;

  VStore(int nvar) {
    size = nvar;
    CUDIE(cudaMallocManaged(&data, sizeof(Interval) * nvar));
  }

  void print_store() {
    for(int i=0; i < size; ++i) {
      printf("%d = [%d..%d]\n", i, data[i].lb, data[i].ub);
    }
  }

  // lb <= x <= ub
  void dom(Var x, Interval itv) {
    data[x] = itv;
  }

  Interval& operator[](int i) {
    return i < 0 ? data[-i].neg() : data[i];
  }
};

/// x + y <= c
struct XplusYleqC {
  Var x;
  Var y;
  int c;

  XplusYleqC(Var x, Var y, int c) : x(x), y(y), c(c) {}

  void propagate(VStore vstore)
  {
    vstore[x].join({vstore[x].lb, c - vstore[y].lb});
    vstore[y].join({vstore[y].lb, c - vstore[x].lb});
  }

  bool is_entailed(VStore vstore) {
    return vstore[x].ub + vstore[y].ub <= c;
  }

  bool is_disentailed(VStore vstore) {
    return vstore[x].lb + vstore[y].lb > c;
  }
};

/// b <=> left /\ right
struct ReifiedLogicalAnd {
  Var b;
  XplusYleqC left;
  XplusYleqC right;

  ReifiedLogicalAnd(Var b, XplusYleqC left, XplusYleqC right) :
    b(b), left(left), right(right) {}

  void propagate(VStore vstore) {
    if vstore[b] == 0 {
      XplusYleqC c1(-left.x, -left.y, -left.c-1);
      c1.propagate()
      XplusYleqC c2(-right.x, -right.y, -right.c-1);
      c2.propagate();
    }
    else if vstore[b] == 1 {
      left.propagate(vstore);
      right.propagate(vstore);
    }
    else if left.is_entailed(vstore) && right.is_entailed(vstore) {
      vstore[b] = 1;
    }
    else if left.is_disentailed(vstore) || right.is_disentailed(vstore) {
      vstore[b] = 0;
    }
  }
}

int main() {

  // I. Declare the variable's domains.
  int nvar = 4;
  int x = 0;
  int y = 1;
  int z = 2;
  int b = 3;
  VStore vstore = new_vstore(nvar);

  dom(vstore, x, {0, 2});
  dom(vstore, y, {1, 3});
  dom(vstore, z, {2, 4});
  dom(vstore, b, {0,1});

  print_store(vstore);

  // II. Declare the constraints
  XplusYleqC c1(x,y,2);
  XplusYleqC c2(y,z,2);
  ReifiedLogicalAnd c3(b, c1, c2);

  // III. Solve the problem.
  c3.propagate();

  print_store(vstore);

  return 0;
}
