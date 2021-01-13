#include <iostream>
#include <algorithm>
#include <stdio.h>

#define CUDIE(result) { \
        cudaError_t e = (result); \
        if (e != cudaSuccess) { \
                std::cerr << __FILE__ << ":" << __LINE__; \
                std::cerr << " CUDA runtime error: " << cudaGetErrorString(e) << '\n'; \
                exit((int)e); \
        }}

#define CUDIE0() CUDIE(cudaGetLastError())

struct Interval {
  int lb;
  int ub;
};

inline void join(Interval* a, Interval b) {
  a->lb = std::max(a->lb, b.lb);
  a->ub = std::min(a->ub, b.ub);
}

struct VStore {
  Interval* data;
  size_t size;
};

VStore new_vstore(int nvar) {
  Interval* data;
  CUDIE(cudaMallocManaged(data, sizeof(Interval) * nvar));
  {data, nvar}
}

typedef int Var;
typedef Interval* VStore;

void print_store(VStore vstore) {
  for(int i=0; i < vstore.size; ++i) {
    printf("%d = [%d..%d]\n", i, vstore[i].lb, vstore[i].ub);
  }
}

// lb <= x <= ub
void dom(VStore vstore, Var x, Interval itv) {
  vstore.data[x] = itv;
}

// x + y <= c
void x_plus_y_leq_c(VStore vstore, Var x, Var y, int c)
{
  join(&vstore[x], {vstore.data[x].lb, c - vstore[y].lb});
  join(&vstore[y], {vstore.data[y].lb, c - vstore[x].lb});
}

int main() {
  VStore vstore = new_vstore(2);
  int x = 0;
  int y = 1;
  dom(vstore, x, {0, 2});
  dom(vstore, y, {1, 3});
  print_store(vstore);
  x_plus_y_leq_c(vstore, x, y, 2);
  print_store(vstore);
  return 0;
}
