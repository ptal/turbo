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

#ifndef TURBO_STATUS_HPP
#define TURBO_STATUS_HPP

#include "cuda_helper.hpp"
#include "memory.hpp"

enum Status {
  UNKNOWN = 0,
  IDLE = 1,
  ENTAILED = 2,
  DISENTAILED = 3
};

CUDA inline Status join_status(Status status1, Status status2) {
  return max<Status>(status1, status2);
}

CUDA inline const char* string_of_status(Status status) {
  static const char* unknown = "unknown";
  static const char* idle = "idle";
  static const char* entailed = "entailed";
  static const char* disentailed = "disentailed";
  switch (status) {
    case UNKNOWN: return unknown;
    case IDLE: return idle;
    case ENTAILED: return entailed;
    default: return disentailed;
  }
}

class PropagatorsStatus {
  Vector<Status> status;
  bool changed;

public:
  template<typename Allocator = ManagedAllocator>
  CUDA PropagatorsStatus(size_t n, Allocator& allocator = Allocator()):
    status(n), changed(false)
  {
    for(int i = 0; i < n; ++i) {
      status[i] = UNKNOWN;
    }
  }

  CUDA PropagatorsStatus() = delete;
  CUDA PropagatorsStatus(const PropagatorsStatus&) = delete;

  CUDA inline size_t size() const { return status.size(); }

  CUDA inline const Status& operator[](size_t i) const {
    return status[i];
  }

  CUDA inline void inplace_join(size_t i, Status s) {
    if (status[i] < s) {
      changed = true;
    }
    status[i] = join_status(status[i], s);
  }

  CUDA Status join() const {
    int unk = 0;
    int idle = 0;
    for(int i = 0; i < size(); ++i) {
      switch (status[i]) {
        case UNKNOWN: ++unk; break;
        case IDLE: ++idle; break;
        case ENTAILED: break;
        case DISENTAILED: return DISENTAILED;
      }
      //printf("join: %d %s\n", i, string_of_status(status[i]));
    }
    if (unk == 0) {
      if (idle == 0) {
        return ENTAILED;
      }
      return IDLE;
    }
    return UNKNOWN;
  }

  CUDA void reset_changed() {
    changed = false;
  }

  CUDA bool has_changed() {
    return changed;
  }

  CUDA void wake_up_all() {
    for(int i = 0; i < size(); ++i) {
      if (status[i] == IDLE) {
        status[i] = UNKNOWN;
      }
    }
  }

  CUDA void reset() {
    for(int i = 0; i < size(); ++i) {
      status[i] = UNKNOWN;
    }
  }
};

#endif
