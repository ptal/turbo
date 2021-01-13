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

template<typename T>__device__ __host__ T min(T a, T b) { return a<=b ? a : b; }
template<typename T>__device__ __host__ T max(T a, T b) { return a>=b ? a : b; }

struct Interval {
  int lb;
  int ub;
};

__host__ __device__
void join(Interval* a, Interval b) {
  a->lb = max<int>(a->lb, b.lb);
  a->ub = min<int>(a->ub, b.ub);
}

struct VStore {
  Interval* data;
  size_t size;
};

VStore* new_vstore(size_t nvar) {
  VStore *vsd;
  CUDIE(cudaMallocManaged(&vsd, sizeof(*vsd)));
  Interval* data;
  CUDIE(cudaMallocManaged(&data, sizeof(Interval) * nvar));
  vsd->data = data;
  vsd->size = nvar;
  return vsd;
}

typedef size_t Var;

void print_store(VStore vstore) {
  for(int i=0; i < vstore.size; ++i) {
    printf("%d = [%d..%d]\n", i, vstore.data[i].lb, vstore.data[i].ub);
  }
}

// lb <= x <= ub
void dom(VStore vstore, Var x, Interval itv) {
  vstore.data[x] = itv;
}

// x + y <= c
__global__
void x_plus_y_leq_c(VStore* vstore, Var x, Var y, int c)
{
  join(&vstore->data[x], {vstore->data[x].lb, c - vstore->data[y].lb});
  join(&vstore->data[y], {vstore->data[y].lb, c - vstore->data[x].lb});
}

__device__ VStore vstore_d;

int main() {
  VStore* vstore = new_vstore(2);
  int x = 0;
  int y = 1;
  dom(*vstore, x, {0, 2});
  dom(*vstore, y, {1, 3});
  print_store(*vstore);
  x_plus_y_leq_c<<<1,1>>>(vstore, x, y, 2);
  CUDIE0();
  CUDIE(cudaDeviceSynchronize());
  // page fault expected:
  print_store(*vstore);
  return 0;
}
