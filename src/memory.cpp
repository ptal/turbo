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

#include "memory.hpp"

ManagedAllocator managed_allocator;
__device__ GlobalAllocator global_allocator;

__device__ SharedAllocator::SharedAllocator(int* mem, size_t capacity):
  mem(mem), capacity(capacity), offset(0) {}

__device__ void* SharedAllocator::allocate(size_t bytes) {
  assert(offset < capacity);
  void* m = (void*)&mem[offset];
  offset += bytes / sizeof(int);
  offset += offset % sizeof(int*);
  return m;
}

__device__ void* operator new(size_t bytes, SharedAllocator& p) {
  return p.allocate(bytes);
}

__device__ void* operator new[](size_t bytes, SharedAllocator& p) {
  return p.allocate(bytes);
}

// For now, we don't support freeing the memory.
__device__ void operator delete(void* ptr, SharedAllocator& p) {}
__device__ void operator delete[](void* ptr, SharedAllocator& p) {}

void* ManagedAllocator::allocate(size_t bytes) {
  void* data;
  cudaMallocManaged(&data, bytes);
  return data;
}

void ManagedAllocator::deallocate(void* data) {
  cudaFree(data);
}

void* operator new(size_t bytes, ManagedAllocator& p) {
  return p.allocate(bytes);
}

void* operator new[](size_t bytes, ManagedAllocator& p) {
  return p.allocate(bytes);
}

void operator delete(void* ptr, ManagedAllocator& p) {
  p.deallocate(ptr);
}

void operator delete[](void* ptr, ManagedAllocator& p) {
  p.deallocate(ptr);
}

__device__ void* GlobalAllocator::allocate(size_t bytes) {
  void* data;
  MALLOC_CHECK(cudaMalloc(&data, bytes));
  return data;
}

__device__ void GlobalAllocator::deallocate(void* data) {
  cudaFree(data);
}

__device__ void* operator new(size_t bytes, GlobalAllocator& p) {
  return p.allocate(bytes);
}

__device__ void* operator new[](size_t bytes, GlobalAllocator& p) {
  return p.allocate(bytes);
}

__device__ void operator delete(void* ptr, GlobalAllocator& p) {
  p.deallocate(ptr);
}

__device__ void operator delete[](void* ptr, GlobalAllocator& p) {
  p.deallocate(ptr);
}