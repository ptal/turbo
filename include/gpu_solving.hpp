// Copyright 2023 Pierre Talbot

#ifndef GPU_SOLVING_HPP
#define GPU_SOLVING_HPP

#include "common_solving.hpp"

using F = TFormula<ManagedAllocator>;
using FormulaPtr = battery::shared_ptr<F, ManagedAllocator>;

#ifdef __NVCC__

__global__ void gpu_solve_kernel(FormulaPtr f) {
  // f->seq(0).print();
  // printf(" %s ", string_of_sig(f->sig()));
  // f->seq(1).print();
  // printf("\n");
  f->print();
  printf("\n");
}

#endif

void gpu_solve(const Configuration& config, GlobalStatistics& stats) {
  #ifndef __NVCC__
    std::cout << "You must use the NVCC compiler to compile Turbo on GPU." << std::endl;
  #else
  auto start = std::chrono::high_resolution_clock::now();

  // I. Parse the FlatZinc model.
  FlatZincOutput<ManagedAllocator> output;
  FormulaPtr f = parse_flatzinc<ManagedAllocator>(config.problem_path, output);
  if(!f) {
    std::cerr << "Could not parse FlatZinc model." << std::endl;
    exit(EXIT_FAILURE);
  }

  printf("%%FlatZinc parsed\n");

  size_t stack_limit;
  cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize);
  std::cout << "CUDA stack limit: " << stack_limit << std::endl;
  cudaDeviceSetLimit(cudaLimitStackSize, stack_limit*10);

  // FormulaPtr f2 = battery::make_shared<F, ManagedAllocator>(
  //   F::make_binary(
  //     F::make_binary(F::make_lvar(0, "x"), IN, F::make_binary(F::make_z(1), ADD, F::make_z(1))),
  //     AND,
  //     F::make_binary(F::make_lvar(0, "x"), IN, F::make_binary(F::make_z(1), ADD, F::make_z(1)))));
  // f2->print();
  // printf("\n");
  // gpu_solve_kernel<<<1, 1>>>(f2);
  // CUDIE(cudaDeviceSynchronize());

  f->print();
  printf("\n");
  gpu_solve_kernel<<<1, 1>>>(f);
  CUDIE(cudaDeviceSynchronize());
#endif
}

#endif
