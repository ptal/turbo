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
#include <type_traits>
#include "cuda_helper.hpp"

class SharedAllocator {
  int* mem;
  size_t offset;
  size_t capacity;
public:
  __device__ SharedAllocator(int* mem, size_t capacity);
  __device__ SharedAllocator() = delete;
  __device__ void* allocate(size_t bytes);
};

__device__ void* operator new(size_t bytes, SharedAllocator& p);
__device__ void* operator new[](size_t bytes, SharedAllocator& p);
__device__ void operator delete(void* ptr, SharedAllocator& p);
__device__ void operator delete[](void* ptr, SharedAllocator& p);

class ManagedAllocator {
public:
  void* allocate(size_t bytes);
  void deallocate(void* data);
};

extern ManagedAllocator managed_allocator;

void* operator new(size_t bytes, ManagedAllocator& p);
void* operator new[](size_t bytes, ManagedAllocator& p);
void operator delete(void* ptr, ManagedAllocator& p);
void operator delete[](void* ptr, ManagedAllocator& p);

class GlobalAllocator {
public:
  __device__ void* allocate(size_t bytes);
  __device__ void deallocate(void* data);
};

__device__ extern GlobalAllocator global_allocator;

__device__ void* operator new(size_t bytes, GlobalAllocator& p);
__device__ void* operator new[](size_t bytes, GlobalAllocator& p);
__device__ void operator delete(void* ptr, GlobalAllocator& p);
__device__ void operator delete[](void* ptr, GlobalAllocator& p);

template<typename T>
class Pointer;

template<typename T>
struct TypeAllocatorDispatch {
  template<typename Allocator>
  __device__ static T* build(const T& from, Allocator& allocator) {
    if constexpr(std::is_constructible<T, const T&, Allocator&>{}) {
      return new(allocator) T(from, allocator);
    }
    else {
      return new(allocator) T(from);
    }
    return nullptr; // to suppress a warning.
  }

  template<typename Allocator>
  __device__ static void build(T* placement, const T& from, Allocator& allocator) {
    if constexpr(std::is_constructible<T, const T&, Allocator&>{}) {
      new(placement) T(from, allocator);
    }
    else {
      new(placement) T(from);
    }
  }
};

template<typename T>
struct TypeAllocatorDispatch<Pointer<T>> {
  __device__ static void build(Pointer<T>* placement, const Pointer<T>& from, SharedAllocator& allocator) {
    if constexpr(std::is_polymorphic_v<T>) {
      new(placement) Pointer(from->clone_in(allocator));
    }
    else {
      new(placement) T(from, allocator);
    }
  }
};

template<typename T>
class Pointer {
  T* ptr;
public:
  typedef T ptr_type;
  typedef Pointer<T> this_type;

  CUDA Pointer(): ptr(nullptr) {}
  CUDA Pointer(T* ptr): ptr(ptr) {}

  template<typename Allocator>
  __device__ Pointer(const T& from, Allocator& allocator):
    ptr(TypeAllocatorDispatch<T>::build(from, allocator))
  {}

  template<typename Allocator>
  __device__ Pointer(const Pointer<T>& from, Allocator& allocator):
    ptr(from.ptr == nullptr
        ? nullptr
        : TypeAllocatorDispatch<T>::build(*from.ptr, allocator))
  {}

  Pointer(const T& from): ptr(new(managed_allocator) T(from)) {}
  Pointer(const Pointer<T>& from): ptr(
    from.ptr == nullptr
     ? nullptr
     : new(managed_allocator) T(*from.ptr)) {}

  CUDA T* operator->() const { assert(ptr != nullptr); return ptr; }
  CUDA T& operator*() const { assert(ptr != nullptr); return *ptr; }

  CUDA void reset(T* ptr) {
    this->ptr = ptr;
  }
};

template<typename T>
class Array {
  typedef Array<T> this_type;

  T* array;
  size_t n;
public:

  template<typename Allocator>
  __device__ Array(int n, Allocator& allocator):
    n(n), array(new(allocator) T[n]) {}

  template<typename Allocator>
  __device__ Array(int n, const T* from, Allocator& allocator):
    n(n), array(new(allocator) T[n])
  {
    for(int i = 0; i < n; ++i) {
      TypeAllocatorDispatch<T>::build(&array[i], from[i], allocator);
    }
  }

  template <typename Allocator>
  __device__ Array(const Array<T>& from, Allocator& allocator):
    n(from.n), array(new(allocator) T[from.n])
  {
    for(int i = 0; i < n; ++i) {
      TypeAllocatorDispatch<T>::build(&array[i], from[i], allocator);
    }
  }

  template <typename Allocator>
  __device__ Array(const T& from, int n, Allocator& allocator):
    n(n), array(new(allocator) T[n])
  {
    for(int i = 0; i < n; ++i) {
      TypeAllocatorDispatch<T>::build(&array[i], from, allocator);
    }
  }

  Array(int n): n(n), array(new(managed_allocator) T[n]) {}
  Array(const T* from, int n): n(n), array(new(managed_allocator) T[n]) {
    for(int i = 0; i < n; ++i) {
      new(&array[i]) T(from[i]);
    }
  }

  Array(const Array<T>& from): n(from.n), array(new(managed_allocator) T[from.n]) {
    for(int i = 0; i < n; ++i) {
      new(&array[i]) T(from[i]);
    }
  }

  Array(const T& from, int n): n(n), array(new(managed_allocator) T[n]) {
    for(int i = 0; i < n; ++i) {
      new(&array[i]) T(from);
    }
  }

  Array(const std::vector<T>& from):
    n(from.size()), array(new(managed_allocator) T[from.size()])
  {
    for(int i = 0; i < n; ++i) {
      new(&array[i]) T(from[i]);
    }
  }

  CUDA size_t size() const { return n; }
  CUDA T& operator[](size_t i) { assert(i < n); return array[i]; }
  CUDA const T& operator[](size_t i) const { assert(i < n); return array[i]; }
  CUDA T* data() { return array; }
  CUDA const T* data() const { return array; }
};

#endif // MEMORY_HPP
