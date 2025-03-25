// Copyright 2025 Pierre Talbot

#ifndef TURBO_MEMORY_GPU_HPP
#define TURBO_MEMORY_GPU_HPP

#include "battery/allocator.hpp"
#include "config.hpp"

namespace bt = ::battery;

#ifdef __CUDACC__

/** Depending on the problem, we can store the abstract elements in different memories.
 * The "worst" is everything in global memory (GLOBAL) when the problem is too large for the shared memory.
 * The "best" is when the ternary constraint network (both the store of variables and the propagators) can be stored in shared memory (TCN_SHARED).
 * A third possibility is to store only the variables' domains in the shared memory (STORE_SHARED).
*/
enum class MemoryKind {
  GLOBAL,
  STORE_SHARED,
  TCN_SHARED
};

/** The shared memory must be configured by hand before the kernel is launched.
 * This class encapsulates information about the size of the store and propagators, and help creating the allocators accordingly.
*/
struct MemoryConfig {
  MemoryKind mem_kind;
  size_t shared_bytes;
  size_t store_bytes;
  size_t prop_bytes;

  MemoryConfig() = default;
  MemoryConfig(const MemoryConfig&) = default;

  MemoryConfig(size_t store_bytes, size_t prop_bytes):
    mem_kind(MemoryKind::GLOBAL),
    shared_bytes(0),
    store_bytes(store_bytes),
    prop_bytes(prop_bytes)
  {}

  MemoryConfig(const void* kernel, size_t store_bytes, size_t prop_bytes):
    store_bytes(store_bytes),
    prop_bytes(prop_bytes)
  {
    int maxSharedMemPerSM;
    cudaDeviceGetAttribute(&maxSharedMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    int alignment = 128; // just in case...
    if(store_bytes + prop_bytes + alignment < maxSharedMemPerSM) {
      shared_bytes = store_bytes + prop_bytes + alignment;
      mem_kind = MemoryKind::TCN_SHARED;
    }
    else if(store_bytes + alignment < maxSharedMemPerSM) {
      shared_bytes = store_bytes + alignment;
      mem_kind = MemoryKind::STORE_SHARED;
    }
    else {
      shared_bytes = 0;
      mem_kind = MemoryKind::GLOBAL;
    }
    if(shared_bytes != 0) {
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    }
  }

  CUDA bt::pool_allocator make_global_pool(size_t bytes) {
    void* mem_pool = bt::global_allocator{}.allocate(bytes);
    return bt::pool_allocator(static_cast<unsigned char*>(mem_pool), bytes);
  }

  CUDA bt::pool_allocator make_shared_pool(unsigned char* shared_mem) {
    return bt::pool_allocator(shared_mem, shared_bytes);
  }

  CUDA bt::pool_allocator make_prop_pool(bt::pool_allocator shared_mem) {
    if(mem_kind == MemoryKind::TCN_SHARED) {
      return shared_mem;
    }
    else {
      return make_global_pool(prop_bytes);
    }
  }

  CUDA bt::pool_allocator make_store_pool(bt::pool_allocator shared_mem) {
    if(mem_kind == MemoryKind::TCN_SHARED || mem_kind == MemoryKind::STORE_SHARED) {
      return shared_mem;
    }
    else {
      return make_global_pool(store_bytes);
    }
  }

  template <class Config, class Stat>
  CUDA void print_mzn_statistics(const Config& config, const Stat& stats) const {
    stats.print_stat("memory_configuration",
      mem_kind == MemoryKind::GLOBAL ? "global" : (
      mem_kind == MemoryKind::STORE_SHARED ? "store_shared" : "tcn_shared"));
    stats.print_memory_statistics(config.verbose_solving, "shared_mem", shared_bytes);
    stats.print_memory_statistics(config.verbose_solving, "store_mem", store_bytes);
    stats.print_memory_statistics(config.verbose_solving, "propagator_mem", prop_bytes);
    stats.print_mzn_end_stats();
  }
};

template <class T>
__global__ void gpu_sizeof_kernel(size_t* size) {
  *size = sizeof(T);
}

template <class T>
size_t gpu_sizeof() {
  auto s = bt::make_unique<size_t, bt::managed_allocator>();
  gpu_sizeof_kernel<T><<<1, 1>>>(s.get());
  CUDAEX(cudaDeviceSynchronize());
  return *s;
}

void check_support_managed_memory() {
  int attr = 0;
  int dev = 0;
  CUDAEX(cudaDeviceGetAttribute(&attr, cudaDevAttrManagedMemory, dev));
  if (!attr) {
    std::cerr << "The GPU does not support managed memory." << std::endl;
    exit(EXIT_FAILURE);
  }
}

void check_support_concurrent_managed_memory() {
  int attr = 0;
  int dev = 0;
  CUDAEX(cudaDeviceGetAttribute(&attr, cudaDevAttrConcurrentManagedAccess, dev));
  if (!attr) {
#ifdef NO_CONCURRENT_MANAGED_MEMORY
    printf("%% WARNING: The GPU does not support concurrent access to managed memory, hence we fall back on pinned memory.\n");
  /** Set cudaDeviceMapHost to allow cudaMallocHost() to allocate pinned memory
   * for concurrent access between the device and the host. It must be called
   * early, before any CUDA management functions, so that we can fall back to
   * using the pinned_allocator instead of the managed_allocator.
   * This is required on Windows, WSL, macOS, and NVIDIA GRID.
   * See also PR #18.
   */
    unsigned int flags = 0;
    CUDAEX(cudaGetDeviceFlags(&flags));
    flags |= cudaDeviceMapHost;
    CUDAEX(cudaSetDeviceFlags(flags));
#else
    printf("%% To run Turbo on this GPU you need to build Turbo with the option NO_CONCURRENT_MANAGED_MEMORY.\n");
    exit(EXIT_FAILURE);
#endif
  }
}

/** Wait the solving ends because of a timeout, CTRL-C or because the kernel finished. */
template<class CP, class Timepoint>
bool wait_solving_ends(cuda::std::atomic_flag& stop, CP& root, const Timepoint& start) {
  cudaEvent_t event;
  cudaEventCreateWithFlags(&event,cudaEventDisableTiming);
  cudaEventRecord(event);
  while(!must_quit(root) && check_timeout(root, start) && cudaEventQuery(event) == cudaErrorNotReady) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if(cudaEventQuery(event) == cudaErrorNotReady) {
    stop.test_and_set();
    root.prune();
    return true;
  }
  else {
    cudaError error = cudaDeviceSynchronize();
    if(error == cudaErrorIllegalAddress) {
      printf("%% ERROR: CUDA kernel failed due to an illegal memory access. This might be due to a stack overflow because it is too small. Try increasing the stack size with the options -stack. If it does not work, please report it as a bug.\n");
      exit(EXIT_FAILURE);
    }
    CUDAEX(error);
    return false;
  }
}

#endif
#endif
