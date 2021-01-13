#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <new>

#define CUDIE(result) { \
        cudaError_t e = (result); \
        if (e != cudaSuccess) { \
                std::cerr << __FILE__ << ":" << __LINE__; \
                std::cerr << " CUDA runtime error: " << cudaGetErrorString(e) << '\n'; \
                exit((int)e); \
        }}

#define CUDIE0() CUDIE(cudaGetLastError())

template<typename T>__device__ __host__ T min(T a, T b) { return a<=b ? a : b; }
template<typename T>__device__ __host__ T max(T a, T b) { return a>=b ? a : b; }

// A variable with a negative index represents the negation `-x`.
// The conversion is automatically handled in `VStore::operator[]`.
typedef size_t Var;

struct Interval {
  int lb;
  int ub;

  __host__ __device__
  Interval join(Interval b) {
    lb = max<int>(lb, b.lb);
    ub = min<int>(ub, b.ub);
    return *this;
  }

  __host__ __device__
  Interval neg() {
    return {-ub, -lb};
  }

  __host__ __device__
  bool operator==(int x) {
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

  /*VStore(const VStore& s) {
    // use : size{s.size}, ... ?
    size = s.size;
    data = s.data;
  }*/

  __host__ __device__
  void print_store() {
    for(int i=0; i < size; ++i) {
      printf("%d = [%d..%d]\n", i, data[i].lb, data[i].ub);
    }
  }

  // lb <= x <= ub
  __host__ __device__
  void dom(Var x, Interval itv) {
    data[x] = itv;
  }

  __host__ __device__
  void update(int i, Interval itv) {
    if (i<0) {
      data[-i].lb = -itv.ub;
      data[-i].ub = -itv.lb;
    } else { 
      data[i] = itv;
    }
  }

  __host__ __device__
  Interval operator[](int i) {
    return i < 0 ? data[-i].neg() : data[i];
  }
};

/// x + y <= c
struct XplusYleqC {
  Var x;
  Var y;
  int c;

  __device__ __host__
  XplusYleqC(Var x, Var y, int c) : x(x), y(y), c(c) {}

  __device__ __host__
  void propagate(VStore& vstore)
  {
    vstore.update(x, 
        vstore[x].join({vstore[x].lb, c - vstore[y].lb}));
    vstore.update(y, 
        vstore[y].join({vstore[y].lb, c - vstore[x].lb}));
  }

  __device__ __host__
  bool is_entailed(VStore& vstore) {
    return vstore[x].ub + vstore[y].ub <= c;
  }

  __device__ __host__
  bool is_disentailed(VStore& vstore) {
    return vstore[x].lb + vstore[y].lb > c;
  }
};


/// b <=> left /\ right
struct ReifiedLogicalAnd {
  Var b;
  XplusYleqC left;
  XplusYleqC right;

  __device__ __host__
  ReifiedLogicalAnd(Var b, XplusYleqC left, XplusYleqC right) :
    b(b), left(left), right(right) {}

  __device__ __host__
  void propagate(VStore& vstore) {
    if (vstore[b] == 0) {
      XplusYleqC c1(-left.x, -left.y, -left.c-1);
      c1.propagate(vstore);
      XplusYleqC c2(-right.x, -right.y, -right.c-1);
      c2.propagate(vstore);
    }
    else if (vstore[b] == 1) {
      left.propagate(vstore);
      right.propagate(vstore);
    }
    else if (left.is_entailed(vstore) && right.is_entailed(vstore)) {
      vstore.update(b, {1, 1});
    }
    else if (left.is_disentailed(vstore) || right.is_disentailed(vstore)) {
      vstore.update(b, {0, 0});
    }
  }
};

__global__
void propagate_k(ReifiedLogicalAnd c, VStore* vstore) {
  c.propagate(*vstore);
}


// struct LogicalOr {
//   XplusYleqC left;
//   XplusYleqC right;

//   LogicalOr(XplusYleqC left, XplusYleqC right)
// }


int main() {

  // I. Declare the variable's domains.
  int nvar = 4;
  int x = 0;
  int y = 1;
  int z = 2;
  int b = 3;

  void* v;
  CUDIE(cudaMallocManaged(&v, sizeof(VStore)));
  VStore* vstore = new(v) VStore(nvar);

  vstore->dom(x, {0, 2});
  vstore->dom(y, {1, 3});
  vstore->dom(z, {2, 4});
  vstore->dom(b, {0,1});

  vstore->print_store();

  // II. Declare the constraints
  XplusYleqC c1(x,y,2);
  XplusYleqC c2(y,z,2);
  ReifiedLogicalAnd c3(b, c1, c2);

  // III. Solve the problem.
  //c3.propagate(*vstore);
  propagate_k<<<1,1>>>(c3, vstore);
  CUDIE(cudaDeviceSynchronize());

  vstore->print_store();

  return 0;
}
