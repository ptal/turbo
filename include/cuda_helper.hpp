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

#ifdef __CUDACC__
#define CUDA __device__ __host__

#define CUDA_VAR __device__ __managed__
#define CUDA_GLOBAL __global__

#define CUDIE(result) { \
        cudaError_t e = (result); \
        if (e != cudaSuccess) { \
                std::cerr << __FILE__ << ":" << __LINE__; \
                std::cerr << " CUDA runtime error: " << cudaGetErrorString(e) << '\n'; \
                exit((int)e); \
        }}

#define CUDIE0() CUDIE(cudaGetLastError())
#else
#define CUDA
#define CUDA_VAR
#define CUDA_GLOBAL
#define CUDIE(result)
#define CUDIE0()
#endif

template<typename T>CUDA T min(T a, T b) { return a<=b ? a : b; }
template<typename T>CUDA T max(T a, T b) { return a>=b ? a : b; }

#endif
