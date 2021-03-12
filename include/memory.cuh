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

#include <vector>
#include <cassert>
#include "cuda_helper.hpp"

struct ground_type_tag_t {};
struct polymorphic_type_tag_t {};

extern ground_type_tag_t ground_type_tag;
extern polymorphic_type_tag_t polymorphic_type_tag;

class SharedAllocator {
  char* mem;
  size_t offset;
public:
  __device__ SharedAllocator(char* mem);
  __device__ SharedAllocator() = delete;
  __device__ void* allocate(size_t bytes);
};

__device__ void* operator new(size_t bytes, SharedAllocator& p);
__device__ void* operator new[](size_t bytes, SharedAllocator& p);

class ManagedAllocator {
public:
  void* allocate(size_t bytes);
  void deallocate(void* data);
};

__device__ extern ManagedAllocator managed_allocator;

void* operator new(size_t bytes, ManagedAllocator& p);
void* operator new[](size_t bytes, ManagedAllocator& p);

// class GlobalAllocator {
// public:
//   __device__ void* allocate(size_t bytes) {
//     void* data;
//     MALLOC_CHECK(cudaMalloc(&data, bytes));
//     return data;
//   }

//   __device__ void deallocate(void* data) {
//     cudaFree(data);
//   }
// };

// __device__ void* operator new(size_t bytes, GlobalAllocator& p) {
//   return p.allocate(bytes);
// }

// __device__ void* operator new[](size_t bytes, GlobalAllocator& p) {
//   return p.allocate(bytes);
// }

template<typename T>
class Pointer {
  T* ptr;
public:
  CUDA Pointer(): ptr(nullptr)
  {}

  CUDA Pointer(T* ptr): ptr(ptr) {}

  template<typename Allocator = ManagedAllocator>
  CUDA Pointer(const T& from, Allocator& allocator = managed_allocator):
    ptr(new(allocator) T(from, allocator))
  {}

  template<typename Allocator = ManagedAllocator>
  CUDA Pointer(const T& from, ground_type_tag_t, Allocator& allocator = managed_allocator):
    ptr(new(allocator) T(from))
  {}

  template<typename Allocator = ManagedAllocator>
  CUDA Pointer(const Pointer<T>& from, Allocator& allocator = managed_allocator):
    ptr(new(allocator) T(*from.ptr, allocator))
  {}

  template<typename Allocator = ManagedAllocator>
  CUDA Pointer(const Pointer<T>& from, ground_type_tag_t, Allocator& allocator = managed_allocator):
    ptr(new(allocator) T(*from.ptr))
  {}

  CUDA Pointer<T> clone_in(SharedAllocator& allocator) const {
    return Pointer(ptr->clone_in(allocator));
  }

  CUDA T* operator->() const { assert(ptr != nullptr); return ptr; }
  CUDA T& operator*() const { assert(ptr != nullptr); return *ptr; }

  CUDA void reset(T* ptr) {
    this->ptr = ptr;
  }
};

template<typename T>
class ArrayBase {
protected:
  T* array;
  size_t n;

  CUDA ArrayBase(size_t n, T* array): n(n), array(array) {}
public:
  CUDA size_t size() const { return n; }
  CUDA T& operator[](size_t i) { assert(i < n); return array[i]; }
  CUDA const T& operator[](size_t i) const { assert(i < n); return array[i]; }
  CUDA T* data() { return array; }
  CUDA const T* data() const { return array; }
};

template<typename T>
class Array: public ArrayBase<T> {
  typedef ArrayBase<T> base_type;
public:
  template<typename Allocator = ManagedAllocator>
  CUDA Array(int n, Allocator& allocator = managed_allocator):
    base_type(n, new(allocator) T[n]) {}

  template<typename Allocator = ManagedAllocator>
  CUDA Array(int n, const T* from, Allocator& allocator = managed_allocator):
    base_type(n, new(allocator) T[n])
  {
    for(int i = 0; i < this->n; ++i) {
      new(&this->array[i]) T(from[i], allocator);
    }
  }

  template<typename Allocator = ManagedAllocator>
  CUDA Array(int n, const T* from, ground_type_tag_t, Allocator& allocator = managed_allocator):
    base_type(n, new(allocator) T[n])
  {
    for(int i = 0; i < this->n; ++i) {
      new(&this->array[i]) T(from[i]);
    }
  }

  template <typename Allocator = ManagedAllocator>
  CUDA Array(const Array<T>& from, Allocator& allocator = managed_allocator):
    base_type(from.n, new(allocator) T[from.n])
  {
    for(int i = 0; i < this->n; ++i) {
      new(&this->array[i]) T(from[i], allocator);
    }
  }

  template <typename Allocator = ManagedAllocator>
  CUDA Array(const Array<T>& from, ground_type_tag_t, Allocator& allocator = managed_allocator):
    base_type(from.n, new(allocator) T[from.n])
  {
    for(int i = 0; i < this->n; ++i) {
      new(&this->array[i]) T(from[i]);
    }
  }

  template <typename Allocator = ManagedAllocator>
  Array(const std::vector<T>& from, Allocator& allocator = managed_allocator):
    base_type(from.size(), new(allocator) T[from.size()])
  {
    for(int i = 0; i < this->n; ++i) {
      new(&this->array[i]) T(from[i], allocator);
    }
  }

  template <typename Allocator = ManagedAllocator>
  Array(const std::vector<T>& from, ground_type_tag_t, Allocator& allocator = managed_allocator):
    base_type(from.size(), new(allocator) T[from.size()])
  {
    for(int i = 0; i < this->n; ++i) {
      new(&this->array[i]) T(from[i]);
    }
  }

  template <typename Allocator = ManagedAllocator>
  CUDA Array(const T& from, int n, Allocator& allocator = managed_allocator):
    base_type(n, new(allocator) T[n])
  {
    for(int i = 0; i < this->n; ++i) {
      new(&this->array[i]) T(from, allocator);
    }
  }
};

// Special constructor for array of polymorphic pointers.
template<typename T>
class Array<Pointer<T>>: public ArrayBase<Pointer<T>> {
  typedef ArrayBase<Pointer<T>> base_type;
public:
  CUDA Array(const Array<Pointer<T>>& from, polymorphic_type_tag_t, SharedAllocator& allocator):
    base_type(from.n, new(allocator) Pointer<T>[from.n])
  {
    for(int i = 0; i < this->n; ++i) {
      this->array[i] = from[i].clone_in(allocator);
    }
  }

  template <typename Allocator = ManagedAllocator>
  CUDA Array(int n, Allocator& allocator = managed_allocator):
    base_type(n, new(allocator) Pointer<T>[n])
  {}
};

#endif // MEMORY_HPP
