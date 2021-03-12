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

#ifndef MEMORY_HPP
#define MEMORY_HPP

#include "cuda_helper.hpp"

struct ground_type_tag {};
struct polymorphic_type_tag {};

class SharedAllocator {
  char* mem;
  size_t offset;
public:
  __device__ SharedAllocator(char* mem):
    mem(mem), offset(0) {}

  __device__ SharedAllocator() = delete;

  __device__ void* allocate(size_t bytes) {
    void* m = (void*)&mem[offset];
    offset += bytes;
    return m;
  }
};

__device__ void* operator new(size_t bytes, SharedAllocator& p) {
  return p.allocate(bytes);
}

__device__ void* operator new[](size_t bytes, SharedAllocator& p) {
  return p.allocate(bytes);
}


class ManagedAllocator {
public:
  CUDA void* allocate(size_t bytes) {
    void* data;
    CUDIE(cudaMallocManaged(&data, bytes));
    return data;
  }

  CUDA void deallocate(void* data) {
    cudaFree(data);
  }
};

CUDA void* operator new(size_t bytes, ManagedAllocator& p) {
  return p.allocate(bytes);
}

CUDA void* operator new[](size_t bytes, ManagedAllocator& p) {
  return p.allocate(bytes);
}

class GlobalAllocator {
public:
  __device__ void* allocate(size_t bytes) {
    void* data;
    MALLOC_CHECK(cudaMalloc(&data, bytes));
    return data;
  }

  __device__ void deallocate(void* data) {
    cudaFree(data);
  }
};

__device__ void* operator new(size_t bytes, GlobalAllocator& p) {
  return p.allocate(bytes);
}

__device__ void* operator new[](size_t bytes, GlobalAllocator& p) {
  return p.allocate(bytes);
}

template<typename T>
class Pointer {
  T* ptr;
public:
  CUDA Pointer(): ptr(nullptr)
  {}

  CUDA Pointer(T* ptr): ptr(ptr) {}

  template<typename Allocator = ManagedAllocator>
  CUDA Pointer(const T& from, Allocator& allocator = Allocator()):
    ptr(new(allocator) T(from, allocator))
  {}

  template<typename Allocator = ManagedAllocator>
  CUDA Pointer(const T& from, ground_type_tag, Allocator& allocator = Allocator()):
    ptr(new(allocator) T(from))
  {}

  template<typename Allocator = ManagedAllocator>
  CUDA Pointer(const Pointer<T>& from, Allocator& allocator = Allocator()):
    ptr(new(allocator) T(*from.ptr, allocator))
  {}

  template<typename Allocator = ManagedAllocator>
  CUDA Pointer(const Pointer<T>& from, ground_type_tag, Allocator& allocator = Allocator()):
    ptr(new(allocator) T(*from.ptr))
  {}

  template<typename Allocator>
  CUDA Pointer<T> clone_in(Allocator& allocator) {
    return Pointer(ptr->clone_in(allocator));
  }

  CUDA T* operator->() const { assert(ptr != nullptr); return ptr; }
  CUDA T& operator*() const { assert(ptr != nullptr); return *ptr; }
};

template<typename T>
class Array {
  T* data;
  size_t n;
public:

  template<typename Allocator = ManagedAllocator>
  CUDA Array(int n, Allocator& allocator = Allocator()):
    n(n), data(new(allocator) T[n]) {}

  template<typename Allocator = ManagedAllocator>
  CUDA Array(int n, const T* from, Allocator& allocator = Allocator()):
    n(n), data(new(allocator) T[n])
  {
    for(int i = 0; i < n; ++i) {
      new(&data[i]) T(from[i], allocator);
    }
  }

  template<typename Allocator = ManagedAllocator>
  CUDA Array(int n, const T* from, ground_type_tag, Allocator& allocator = Allocator()):
    n(n), data(new(allocator) T[n])
  {
    for(int i = 0; i < n; ++i) {
      new(&data[i]) T(from[i]);
    }
  }

  template <typename Allocator = ManagedAllocator>
  CUDA Array(const Array<T>& from, Allocator& allocator = Allocator()):
    n(from.n), data(new(allocator) T[n])
  {
    for(int i = 0; i < n; ++i) {
      new(&data[i]) T(from[i], allocator);
    }
  }

  template <typename Allocator = ManagedAllocator>
  CUDA Array(const Array<T>& from, ground_type_tag, Allocator& allocator = Allocator()):
    n(from.n), data(new(allocator) T[n])
  {
    for(int i = 0; i < n; ++i) {
      new(&data[i]) T(from[i]);
    }
  }

  CUDA size_t size() const { return n; }
  CUDA T& operator[](size_t i) { return data[i]; }
  CUDA const T& operator[](size_t i) const { return data[i]; }
  CUDA T* data() { return data; }
  CUDA const T* data() const { return data; }
};

// Special constructor for array of polymorphic pointers.
template<typename T>
class Array<Pointer<T>> {
  template <typename Allocator = ManagedAllocator>
  CUDA Array(const Array<Pointer<T>>& from, polymorphic_type_tag, Allocator& allocator = Allocator()):
    n(from.n), data(new(allocator) Pointer<T>[n])
  {
    for(int i = 0; i < n; ++i) {
      data[i] = from[i].clone_in(allocator);
    }
  }
};

#endif // MEMORY_HPP
