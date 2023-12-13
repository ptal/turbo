// Copyright 2023 Pierre Talbot

#ifndef TURBO_ABORT_HPP
#define TURBO_ABORT_HPP

// #include <cuda/std/chrono>
#include "battery/memory.hpp"

using namespace lala;

template <class A>
class Abort {
  // using time_point_type = std::chrono::time_point<std::chrono::system_clock>;
  // time_point_type start_time;
  // size_t max_duration_sec;

  using sub_type = A;
  using sub_ptr = abstract_ptr<sub_type>;

  BInc<battery::atomic_memory_grid> gpu_abort;
  BInc<battery::atomic_memory_multi_grid> cpu_abort;
  sub_ptr sub;

  CUDA local::BInc is_top() {
    return join(sub->is_top(), join(gpu_abort, cpu_abort));
  }

  CUDA int num_refinements() const {
    return sub->num_refinements();
  }

  template <class Mem>
  CUDA void refine(size_t i, BInc<Mem>& has_changed) {
    sub->refine(i, has_changed);
  }

  CUDA void abort() {
    cpu_abort.tell_top();
    #ifdef __CUDA_ARCH__
      gpu_abort.tell_top();
    #endif
  }
};

#endif
