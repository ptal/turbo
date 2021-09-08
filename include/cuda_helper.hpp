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

#ifndef CUDA_HELPER_HPP
#define CUDA_HELPER_HPP

#include <cstdio>
#include <iostream>

#ifdef __NVCC__
  #define CUDA __device__ __host__
  #define CUDA_DEVICE __device__
  #define CUDA_VAR __device__ __managed__
  #define CUDA_GLOBAL __global__
  #define INLINE __forceinline__

  #define CUDIE(result) { \
    cudaError_t e = (result); \
    if (e != cudaSuccess) { \
      printf("%s:%d CUDA runtime error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      assert(0); \
    }}

  #define CUDIE0() CUDIE(cudaGetLastError())

  #include <cuda/atomic>
  using atomic_int = cuda::atomic<int, cuda::thread_scope_block>;
  using atomic_bool = cuda::atomic<bool, cuda::thread_scope_block>;
  #define memory_order_relaxed cuda::memory_order_relaxed
#else
  #define INLINE
  #include <atomic>
  using atomic_int = std::atomic_int;
  using atomic_bool = std::atomic_bool;
  #define memory_order_relaxed std::memory_order_relaxed
#endif

template<typename T>CUDA T min(T a, T b) { return a<=b ? a : b; }
template<typename T>CUDA T max(T a, T b) { return a>=b ? a : b; }

CUDA static constexpr int limit_min() noexcept { return -__INT_MAX__ - 1; }
CUDA static constexpr int limit_max() noexcept { return __INT_MAX__; }

#ifdef DEBUG
  #define TRACE
  #define LOG(X) X
#else
  #define LOG(X)
#endif

#ifdef TRACE
  #define INFO(X) X
#else
  #define INFO(X)
#endif

#define MALLOC_CHECK(M) { \
  cudaError_t rc = M; \
  if (rc != cudaSuccess) { \
    printf("Could not allocate the stack (error = %d)\n", rc); \
    assert(0); \
  }}

template<typename T> CUDA void swap(T* a, T* b) {
  T c = *a;
  *a = *b;
  *b = c;
}

template<typename T>
CUDA void malloc2(T* &data, int n) {
  MALLOC_CHECK(cudaMalloc(&data, sizeof(T) * n));
}

template<typename T>
void malloc2_managed(T* &data, int n) {
  CUDIE(cudaMallocManaged(&data, sizeof(T) * n));
}

template<typename T>
CUDA void free2(T* data) {
  cudaFree(data);
}

#endif // CUDA_HELPER_HPP
