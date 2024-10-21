// Copyright 2024 Pierre Talbot

#ifndef TURBO_HYBRID_DIVE_AND_SOLVE_HPP
#define TURBO_HYBRID_DIVE_AND_SOLVE_HPP

#include "common_solving.hpp"
#include <mutex>
#include <thread>

namespace bt = ::battery;

/**
 * "Dive and solve" is a new algorithm to parallelize the standard "propagate and search" algorithm of constraint programming.
 * Given a depth `d`, we create `2^d` subproblems that we solve in parallel.
 * We create as many CPU threads as blocks on the GPU (option `config.or_nodes`).
 * A CPU thread takes the next subproblem available and run the "propagate and search" algorithm on it.
 * The CPU thread offloads the propagation to the GPU, but take care of splitting and backtracking in the search tree, as well as maintaining the best bound found, and the statistics.
 * Therefore, a kernel with only 1 block is executed each time we propagate a node.
 * Since many CPU threads execute in parallel, there are many kernels running in parallel.
 *
 * We call a task solving a subproblem a "cube" (this loosely follows the terminology of SAT solving with "cube and conquer").
 * By CPU cube, we refer to the local state of a CPU thread for solving a subproblem.
 * By GPU cube, we refer to the local state of a GPU block for solving a subproblem.
 * Note that at each moment, there are at most `config.or_nodes` cubes active in parallel.
 */

#ifdef __CUDACC__

/** We only need a few bytes for the shared memory. */
#define DEFAULT_SHARED_MEM_BYTES 100

/** A CPU cube is the data used by a CPU thread to solve a subproblem. */
struct CPUCube {
  using cube_type = AbstractDomains<
    Itv, bt::standard_allocator, UniqueLightAlloc<bt::standard_allocator,0>, bt::managed_allocator>;
  /** A CPU cube is fully allocated on the CPU, but for the store of variables `cube.store` which is allocated in managed memory.
   * Indeed, when exchanging information between a GPU cube and a CPU cube, only the store of variables need to be transfered.
   */
  cube_type cube;

  /** Each cube runs kernel in distinct stream (otherwise they would all be sequentialized on the default stream). */
  cudaStream_t stream;

  /** We keep a snapshot of the root to reinitialize the CPU cube after each subproblem has been solved. */
  typename cube_type::IST::snapshot_type<bt::standard_allocator> root_snapshot;

  /** This flag becomes `true` when the thread has finished its execution (it exits `dive_and_solve` function). */
  std::atomic_flag finished;

  /** This is the path to the subproblem needed to be solved by this cube.
   * This member is initialized in `CPUData` constructor.
   */
  size_t subproblem_idx;

  CPUCube(const CP<Itv>& root)
   : cube(root)
   , root_snapshot(cube.search_tree->template snapshot<bt::standard_allocator>())
  {
    cudaStreamCreate(&stream);
  }

  ~CPUCube() {
    cudaStreamDestroy(stream);
  }
};

/** A GPU cube is the data used by a GPU block to solve a subproblem.
 * We only allocate the necessary structures to perform propagation: the store of variables and the propagators.
*/
struct GPUCube {
  /** We use atomic to store the interval's lower and upper bounds. */
  using Itv1 = Interval<ZLB<int, bt::atomic_memory_block>>;

  /** We use a `pool_allocator`, this allows to easily switch between global memory and shared memory, if the store of variables can fit inside. */
  using IStore = VStore<Itv1, bt::pool_allocator>;

  /** The store of propagators also uses a `pool_allocator` of global memory. This was necessary due to the slow copy of propagators between CPU and GPU.
   * Indeed, a propagator is a tree-shaped object (like an AST) that contain many pointers, and thus the copy calls the allocation function a lot. */
  using IPC = PC<IStore, bt::pool_allocator>;

  /** The store of variables is only accessible on GPU. */
  abstract_ptr<IStore> store_gpu;

  /** The propagators is only accessible on GPU but the array of propagators is shared among all blocks.
   * Since the propagators are state-less, we avoid duplicating them in each block.
   */
  abstract_ptr<IPC> ipc_gpu;

  /** We also have a store of variables in managed memory to communicate the results of the propagation to the CPU.
   * Note that this store is the same than the one in the corresponding CPU cube (`cube.store`).
   * This member is initialized in `CPUData` constructor.
   */
  abstract_ptr<VStore<Itv, bt::managed_allocator>> store_cpu;

  /** The cumulative number of iterations required to reach a fixpoint.
   * By dividing this statistic by the number of nodes, we get the average number of iterations per node.
   */
  size_t fp_iterations;

  GPUCube() {}

  /** Initialize the store of variables and propagators from existing store and propagators.
   * If `pc_shared` is `true`, the propagators if this cube will be shared with `pc`.
   * Otherwise, a full copy of the propagators is made.
   * We generally want to share the propagators, hence the cube 0 copies the propagators, and all other cubes share them.
  */
  template <class StoreType, class PCType>
  __device__ void allocate(StoreType& store, PCType& pc, size_t bytes, bool pc_shared) {
    void* mem_pool = bt::global_allocator{}.allocate(bytes);
    bt::pool_allocator pool(static_cast<unsigned char*>(mem_pool), bytes);
    AbstractDeps<bt::global_allocator, bt::pool_allocator> deps(pc_shared, bt::global_allocator{}, pool);
    ipc_gpu = bt::allocate_shared<IPC, bt::pool_allocator>(pool, pc, deps);
    store_gpu = deps.template extract<IStore>(store.aty());
  }

  __device__ void deallocate() {
    // NOTE: .reset() does not work because it does not reset the allocator, which is itself allocated in global memory.
    store_gpu = abstract_ptr<IStore>();
    ipc_gpu = abstract_ptr<IPC>();
  }
};

/** This kernel initializes the GPU cubes. See the constructor of `CPUData` for more information. */
template <class Store, class IPC>
__global__ void allocate_gpu_cubes(GPUCube* gpu_cubes,
  size_t n, Store* store, IPC* ipc)
{
  assert(threadIdx.x == 0 && blockIdx.x == 0);
  size_t bytes = store->get_allocator().total_bytes_allocated()
    + sizeof(GPUCube::IStore) + sizeof(GPUCube::IPC) + 1000;
  gpu_cubes[0].allocate(*store, *ipc, bytes + ipc->get_allocator().total_bytes_allocated(), false);
  for(int i = 1; i < n; ++i) {
    gpu_cubes[i].allocate(*gpu_cubes[0].store_gpu, *gpu_cubes[0].ipc_gpu, bytes, true);
  }
}

__global__ void deallocate_gpu_cubes(GPUCube* gpu_cubes, size_t n) {
  assert(threadIdx.x == 0 && blockIdx.x == 0);
  for(int i = 0; i < n; ++i) {
    gpu_cubes[i].deallocate();
  }
}

/** This is the data shared among all CPU threads. */
struct CPUData {
  /** We generate the subproblems lazily.
   * Suppose we generate `2^3` subproblems, we represent the first subproblem as `000`, the second as `001`, the third as `010`, and so on.
   * A `0` means to turn left in the search tree, and a `1` means to turn right.
   * Incrementing this integer will generate the path of the next subproblem.
   */
  ZLB<size_t, bt::atomic_memory<>> next_subproblem;

  /** This is the best bound found so far, globally, across all threads. */
  Itv best_bound;

  /** A flag to stop the threads, for example because of a timeout or CTRL-C. */
  std::atomic_flag cpu_stop;

  /** Due to multithreading, we must protect `stdout` when printing.
   * The model of computation in this work is lock-free, but it seems unavoidable for printing.
  */
  std::mutex print_lock;

  /** The initial problem is only accessible from the CPU.
   * It is used at the beginning to count the bytes used by the store and propagators.
   * And at the end, to merge all the statistics and solutions from all threads, and print them.
   */
  CP<Itv> root;

  /** For each block, how much shared memory are we using? */
  size_t shared_mem_bytes;

  /** Each CPU thread has its own local state to solve a subproblem, called a cube. */
  bt::vector<CPUCube> cpu_cubes;

  /** Each GPU block has its own local state to solve a subproblem, called a cube. */
  bt::vector<GPUCube, bt::managed_allocator> gpu_cubes;

  /** We create as many cubes as CPU threads and GPU blocks (option `config.or_nodes`).
   * All CPU cubes are initialized to different subproblems, from 0 to `config.or_nodes - 1`.
   * Hence the next subproblem to solve is `config.or_nodes`.
   * The GPU cubes are initialized to the same state than the CPU cubes.
   * Further, we connect the CPU and GPU cubes by sharing their store of variables (`gpu_cubes[i].store_cpu` and `cpu_cubes[i].cube.store`).
   * Also, we share the propagators of `gpu_cubes[0].ipc_gpu` with all other cubes `gpu_cubes[i].ipc_gpu` (with i >= 1).
  */
  CPUData(const CP<Itv>& root, size_t shared_mem_bytes)
   : next_subproblem(root.config.or_nodes)
   , best_bound(Itv::top())
   , cpu_stop(false)
   , root(root)
   , shared_mem_bytes(shared_mem_bytes)
   , cpu_cubes(root.config.or_nodes, this->root)
   , gpu_cubes(root.config.or_nodes)
  {
    for(int i = 0; i < root.config.or_nodes; ++i) {
      cpu_cubes[i].subproblem_idx = i;
      gpu_cubes[i].store_cpu = cpu_cubes[i].cube.store;
    }
    /** This is a temporary object to initialize the first cube with the store and propagators. */
    AbstractDomains<Itv, bt::standard_allocator,
      bt::statistics_allocator<UniqueLightAlloc<bt::managed_allocator, 0>>,
      bt::statistics_allocator<UniqueLightAlloc<bt::managed_allocator, 1>>>
    managed_cp(root);
    allocate_gpu_cubes<<<1, 1>>>(gpu_cubes.data(), gpu_cubes.size(), managed_cp.store.get(), managed_cp.ipc.get());
    CUDAEX(cudaDeviceSynchronize());
  }

  ~CPUData() {
    deallocate_gpu_cubes<<<1, 1>>>(gpu_cubes.data(), gpu_cubes.size());
    CUDAEX(cudaDeviceSynchronize());
  }

  CPUData() = delete;
  CPUData(const CPUData&) = delete;
  CPUData(CPUData&&) = delete;
};

void dive_and_solve(CPUData& global, size_t cube_idx);
size_t dive(CPUData& global, size_t cube_idx);
void solve(CPUData& global, size_t cube_idx);
bool propagate(CPUData& global, size_t cube_idx);
bool update_global_best_bound(CPUData& global, size_t cube_idx);
void update_local_best_bound(CPUData& global, size_t cube_idx);
void reduce_cubes(CPUData& global);
template <class Alloc>
void configure_gpu(Configuration<Alloc>& config, size_t shared_mem_bytes);
__global__ void gpu_propagate(GPUCube* cube, size_t shared_bytes);

#endif // __CUDACC__

/** This is the point of entry, we preprocess the problem, create the threads solving the problem, wait for their completion or an interruption, merge and print the statistics. */
void hybrid_dive_and_solve(const Configuration<battery::standard_allocator>& config)
{
#ifndef __CUDACC__
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
#else
  auto start = std::chrono::high_resolution_clock::now();
  /** We start with some preprocessing to reduce the number of variables and constraints. */
  CP<Itv> cp(config);
  cp.preprocess();
  size_t alignment_overhead = 200;
  size_t shared_mem_bytes = DEFAULT_SHARED_MEM_BYTES + alignment_overhead + (cp.store->vars() * sizeof(GPUCube::Itv1));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if(shared_mem_bytes >= deviceProp.sharedMemPerBlock || config.only_global_memory) {
    shared_mem_bytes = DEFAULT_SHARED_MEM_BYTES;
    printf("%%%%%%mzn-stat: memory_configuration=\"global\"\n");
  }
  else {
    printf("%%%%%%mzn-stat: memory_configuration=\"store_shared\"\n");
  }
  printf("%%%%%%mzn-stat: shared_mem=%" PRIu64 "\n", shared_mem_bytes);
  configure_gpu(cp.config, shared_mem_bytes);

  /** Block the signal CTRL-C to notify the threads if we must exit. */
  block_signal_ctrlc();

  /** This is the main data structure, we create all the data for each thread and GPU block. */
  CPUData global(cp, shared_mem_bytes);

  /** Start the algorithm in parallel with as many CPU threads as GPU blocks. */
  std::vector<std::thread> threads;
  for(int i = 0; i < global.root.config.or_nodes; ++i) {
    threads.push_back(std::thread(dive_and_solve, std::ref(global), i));
  }

  /** We wait that either the solving is interrupted, or that all threads have finished. */
  size_t terminated = 0;
  while(terminated < threads.size()) {
    if(must_quit() || !check_timeout(global.root, start)) {
      global.cpu_stop.test_and_set();
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    terminated = 0;
    for(int i = 0; i < global.cpu_cubes.size(); ++i) {
      if(global.cpu_cubes[i].finished.test()) {
        ++terminated;
      }
    }
  }
  for(auto& t : threads) {
    t.join();
  }

  /** We reduce all the statistics of all threads. */
  reduce_cubes(global);
  global.root.print_final_solution();
  global.root.print_mzn_statistics();
#endif
}

#ifdef __CUDACC__

/** This is the main algorithm.
 * Each CPU thread will run one instance of this algorithm with a different `cube_idx`.
 * It interleaves two steps until all subproblems have been solved or we reached another stopping condition (timeout, CTRL-C, pruning conditions):
 *  1) Dive (function `dive`): Given a root node, follow a path in the search tree to reach a subproblem.
 *  2) Solve (function `solve`): Solve the subproblem using the propagate and search algorithm.
*/
void dive_and_solve(CPUData& global, size_t cube_idx)
{
  size_t num_subproblems = global.root.stats.eps_num_subproblems;
  size_t& subproblem_idx = global.cpu_cubes[cube_idx].subproblem_idx;
  auto& cube = global.cpu_cubes[cube_idx].cube;
  /** In each iteration, we will solve one subproblem obtained after a diving phase. */
  while(subproblem_idx < num_subproblems && !global.cpu_stop.test()) {
    if(global.root.config.verbose_solving) {
      std::lock_guard<std::mutex> print_guard(global.print_lock);
      printf("%% Cube %d solves subproblem num %zu\n", cube_idx, subproblem_idx);
    }
    /** The first step is to "dive" by committing to a search path. */
    size_t remaining_depth = dive(global, cube_idx);
    /** If we reached the subproblem without reaching a leaf node, we start the solving phase. */
    if(remaining_depth == 0) {
      solve(global, cube_idx);
      /** If we didn't stop solving because of an external interruption, we increase the number of subproblems solved. */
      if(!global.cpu_stop.test()) {
        cube.stats.eps_solved_subproblems += 1;
      }
    }
    /** We reached a leaf node before the subproblem was reached, it means a whole subtree should be skipped. */
    else if(!global.cpu_stop.test()) {
      /** To skip all the paths of the subtree obtained, we perform bitwise operations.
       * Suppose the current path is "00" turn left two times, and the following search tree:
       *         *         depth = 0
       *        / \
       *      0    1       depth = 1
       *    / \   / \
       *   00 01 10 11     depth = 2
       *
       * If we detect a leaf node at depth 1, after only one left turn, we must skip the remaining of the subtree, in particular to avoid exploring the path "01".
       * To achieve that, we take the current path "00", shift it to the right by 1 (essentially erasing the path that has not been explored), increment it to go to the next subtree (at the same depth), and shift it back to the left to reach the first subproblem of the subtree.
       * Cool huh?
       */
      size_t next_subproblem_idx = ((subproblem_idx >> remaining_depth) + size_t{1}) << remaining_depth;
      global.next_subproblem.meet(ZLB<size_t, bt::local_memory>(next_subproblem_idx));
      /** It is possible that other threads skip similar subtrees.
        * Hence, we only count the subproblems skipped by the thread solving the left most subproblem. */
      if((subproblem_idx & ((size_t{1} << remaining_depth) - size_t{1})) == size_t{0}) {
        cube.stats.eps_skipped_subproblems += next_subproblem_idx - subproblem_idx;
      }
    }
    /** We prepare the cube to solve the next problem.
     * We restore the search tree to the root, and reset the EPS search strategy.
     * We also update the subproblem index to the next subproblem to solve. */
    if(!global.cpu_stop.test()) {
      /** To avoid that several cubes solve the same subproblem, we use an atomic post-increment. */
      subproblem_idx = global.next_subproblem.atomic()++;
      /** The following commented code is completely valid and does not use atomic post-increment.
       * But honestly, we kinda need more performance so... let's avoid reexploring subproblems. */
      // subproblem_idx = global.next_subproblem.value();
      // global.next_subproblem.meet(ZLB<size_t, bt::local_memory>(subproblem_idx + size_t{1}));
      if(subproblem_idx < num_subproblems) {
        cube.search_tree->restore(global.cpu_cubes[cube_idx].root_snapshot);
        cube.eps_split->reset();
      }
    }
  }
  /** If we did not get interrupted, this thread has explored all available subproblems.
   * We record this information in `num_blocks_done` simply because it helps to detect unbalanced workloads (many threads finished but not some others).
   */
  if(!global.cpu_stop.test()) {
    cube.stats.num_blocks_done = 1;
  }
  global.cpu_cubes[cube_idx].finished.test_and_set();
}

/** Given a root problem, we follow a predefined path in the search tree to reach a subproblem.
 * This is call the "dive" operation.
 * The path is given by `subproblem_idx` in the CPU cube (see `CPUData::next_subproblem` for more info).
 * For all dives, the initial problem must be the same and the EPS search strategy must be static (always splitting the tree in the same way).
 * Therefore, don't be tempted to add the objective to the initial problem because it might lead to ignoring some subproblems since the splitting decisions will differ.
 *
 * \return The remaining depth of the path if we reach a leaf node before the end of the path.
 */
size_t dive(CPUData& global, size_t cube_idx) {
  auto& cube = global.cpu_cubes[cube_idx].cube;
  bool stop_diving = false;
  size_t remaining_depth = cube.config.subproblems_power;
  /** The number of iterations depends on the length of the diving path. */
  while(remaining_depth > 0 && !stop_diving && !global.cpu_stop.test()) {
    remaining_depth--;
    bool is_leaf_node = propagate(global, cube_idx);
    /** If we reach a leaf node before the end of the path, we stop and the remaining depth is reported to the caller. */
    if(is_leaf_node) {
      stop_diving = true;
    }
    else {
      /** We create two branches according to the EPS search strategy. */
      auto branches = cube.eps_split->split();
      assert(branches.size() == 2);
      /** We commit to one of the branches depending on the current value on the path.
       * Suppose the depth is 3, the path is "010" we are currently at `remaining_depth = 1`.
       * We must extract the bit "1", and we do so by standard bitwise manipulation.
       * Whenever the branch_idx is 0 means to take the left branch, and 1 means to take the right branch.
       */
      size_t branch_idx = (global.cpu_cubes[cube_idx].subproblem_idx & (size_t{1} << remaining_depth)) >> remaining_depth;
      /** We immediately commit to the branch.
       * It has the effect of reducing the domain of the variables in `cube.store` (and `gpu_cube.store_cpu` since they are aliased).
       */
      cube.ipc->deduce(branches[branch_idx]);
    }
  }
  return remaining_depth;
}

/** We solve a cube using the propagate and search algorithm.
 * We explore the whole search tree of the cube, only stopping earlier due to pruning conditions or external stop signal (e.g. `global.cpu_stop`).
 */
void solve(CPUData& global, size_t cube_idx) {
  auto& cpu_cube = global.cpu_cubes[cube_idx].cube;
  bool has_changed = true;
  while(has_changed && !global.cpu_stop.test()) {
    /** Before propagating, we update the local bound with the best known global bound. */
    update_local_best_bound(global, cube_idx);
    /** We propagate on GPU, and manage the leaf nodes (solution and failed nodes). */
    propagate(global, cube_idx);
    /** We pop a new node from the search tree, ready to explore.
     * If the search tree becomes empty, `has_changed` will be `false`.
     */
    has_changed = cpu_cube.search_tree->deduce();
  }
}

/** Propagate the cube `cube_idx` on the GPU.
 * Check if we reached a leaf node (solution or failed) on the CPU.
 * Branching on unknown nodes is a task left to the caller.
 *
 * \return `true` if we reached a leaf node.
 */
bool propagate(CPUData& global, size_t cube_idx) {
  auto& cube = global.cpu_cubes[cube_idx].cube;
  bool is_leaf_node = false;
  /** The propagation is done by a single block on the GPU. */
  gpu_propagate<<<1, static_cast<unsigned int>(global.root.config.and_nodes), global.shared_mem_bytes, global.cpu_cubes[cube_idx].stream>>>
    (&global.gpu_cubes[cube_idx], global.shared_mem_bytes);
  CUDAEX(cudaStreamSynchronize(global.cpu_cubes[cube_idx].stream));
  /** `on_node` updates the statistics and verifies whether we should stop (e.g. option `--cutnodes`). */
  bool is_pruned = cube.on_node();
  /** If the abstract domain is `bottom`, we reached a leaf node where the problem is unsatisfiable. */
  if(cube.ipc->is_bot()) {
    is_leaf_node = true;
    cube.on_failed_node();
  }
  /** When the problem is "extractable", then all variables are assigned to a single value.
   * It means that we have reached a solution.
   */
  else if(cube.search_tree->template is_extractable<AtomicExtraction>()) {
    is_leaf_node = true;
    /** We save the new best solution found.
     * The "branch-and-bound" (bab) abstract domain has a local store of variable to store the best solution.
     * It adds a new bound constraint to the root of the search tree, such that, on backtracking the best bound is enforced.
     */
    cube.bab->deduce();
    bool print_solution = cube.is_printing_intermediate_sol();
    if(cube.bab->is_optimization()) {
      /** We share the new best bound with the other cubes. */
      print_solution &= update_global_best_bound(global, cube_idx);
    }
    /** If we print all intermediate solutions, and really found a better bound (no other thread found a better one meanwhile), we print the current solution. */
    if(print_solution) {
      cube.print_solution();
    }
    /** We update the statistics, and check if we must terminate (e.g. we stop after N solutions). */
    is_pruned |= cube.update_solution_stats();
  }
  if(is_pruned) {
    /** We notify all threads that we must stop. */
    global.cpu_stop.test_and_set();
  }
  return is_leaf_node;
}

/** This kernel executes the propagation loop on the GPU until a fixpoint is reached.
 * 1) Transfer the store of variables from the CPU to the GPU.
 * 2) Execute the fixpoint engine.
 * 3) Transfer the store of variables from the GPU to the CPU.
 */
__global__ void gpu_propagate(GPUCube* gpu_cube, size_t shared_bytes) {
  using FPEngine = BlockAsynchronousIterationGPU<bt::pool_allocator>;
  extern __shared__ unsigned char shared_mem[];
  assert(blockIdx.x == 0);
  auto group = cooperative_groups::this_thread_block();
  bt::unique_ptr<bt::pool_allocator, bt::global_allocator> shared_mem_pool_ptr;
  bt::pool_allocator& shared_mem_pool = bt::make_unique_block(shared_mem_pool_ptr, shared_mem, shared_bytes);
  // If we booked more than the default shared memory, it means we allocate the store in shared memory.
  if(threadIdx.x == 0 && shared_bytes > DEFAULT_SHARED_MEM_BYTES) {
    gpu_cube->store_gpu->reset_data(shared_mem_pool);
  }
  group.sync();
  gpu_cube->store_cpu->copy_to(group, *gpu_cube->store_gpu);
  // No need to sync here, make_unique_block already contains sync.
  bt::unique_ptr<FPEngine, bt::global_allocator> fp_engine_ptr;
  FPEngine& fp_engine = bt::make_unique_block(fp_engine_ptr, group, shared_mem_pool);
  size_t fp_iterations = fp_engine.fixpoint(*(gpu_cube->ipc_gpu));
  // No need to sync because all threads are always synchronized in the last iteration of the fixpoint loop.
  gpu_cube->store_gpu->copy_to(group, *gpu_cube->store_cpu);
  if(threadIdx.x == 0) {
    gpu_cube->fp_iterations += fp_iterations;
  }
  group.sync(); // to avoid `fp_engine` to be deleted before all warps exit `fp_engine.fixpoint`.
}

/** We update the bound found by the current cube so it is visible to all other cubes.
 * Note that this operation might not always succeed, which is okay, the best bound is still saved locally in `gpu_cubes` and then reduced at the end (in `reduce_cubes`).
 * The worst that can happen is that a best bound is found twice, which does not prevent the correctness of the algorithm.
 *
 * \return `true` if the best bound has changed. Can return `false` if the best bound was updated by another thread meanwhile.
 */
bool update_global_best_bound(CPUData& global, size_t cube_idx) {
  const auto& cube = global.cpu_cubes[cube_idx].cube;
  assert(cube.bab->is_optimization());
  // We retrieve the best bound found by the current cube.
  auto local_best = cube.bab->optimum().project(cube.bab->objective_var());
  // We update the global `best_bound`.
  if(cube.bab->is_maximization()) {
    return global.best_bound.meet_lb(dual<Itv::LB>(local_best.ub()));
  }
  else {
    return global.best_bound.meet_ub(dual<Itv::UB>(local_best.lb()));
  }
}

/** This function essentially does the converse operation of `update_global_best_bound`.
 * We directly update the store with the global best bound.
 * This function should be called in each node, since the best bound is erased on backtracking (it is not included in the snapshot).
 */
void update_local_best_bound(CPUData& global, size_t cube_idx) {
  if(global.cpu_cubes[cube_idx].cube.bab->is_optimization()) {
    auto& cube = global.cpu_cubes[cube_idx].cube;
    VarEnv<bt::standard_allocator> empty_env{};
    auto best_formula = cube.bab->template deinterpret_best_bound<bt::standard_allocator>(
      cube.bab->is_maximization()
      ? Itv(dual<Itv::UB>(global.best_bound.lb()))
      : Itv(dual<Itv::LB>(global.best_bound.ub())));
    IDiagnostics diagnostics;
    bool r = interpret_and_tell(best_formula, empty_env, *cube.store, diagnostics);
    assert(r);
  }
}

/** After solving, we merge all the statistics and best solutions from all cubes together, before printing them. */
void reduce_cubes(CPUData& global) {
  for(int i = 0; i < global.cpu_cubes.size(); ++i) {
    /** `meet` is the merge operation. */
    global.root.meet(global.cpu_cubes[i].cube);
    global.root.stats.fixpoint_iterations += global.gpu_cubes[i].fp_iterations;
  }
}

/** We configure the GPU according to the user configuration:
 * 1) Increase the stack size if needed.
 * 2) Increase the global memory allocation (we set the limit to around 90% of the global memory).
 * 3) Guess the "best" number of threads per block and the number of blocks per SM, if not provided.
 */
template <class Alloc>
void configure_gpu(Configuration<Alloc>& config, size_t shared_mem_bytes) {
  int hint_num_blocks;
  int hint_num_threads;
  CUDAE(cudaOccupancyMaxPotentialBlockSize(&hint_num_blocks, &hint_num_threads, (void*) gpu_propagate, shared_mem_bytes));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  size_t total_global_mem = deviceProp.totalGlobalMem;
  size_t num_sm = deviceProp.multiProcessorCount;
  config.and_nodes = (config.and_nodes == 0) ? hint_num_threads : config.and_nodes;
  config.or_nodes = (config.or_nodes == 0) ? hint_num_blocks : config.or_nodes;
  // The stack allocated depends on the maximum number of threads per SM, not on the actual number of threads per block.
  size_t total_stack_size = num_sm * deviceProp.maxThreadsPerMultiProcessor * config.stack_kb * 1000;
  size_t remaining_global_mem = total_global_mem - total_stack_size;
  remaining_global_mem -= remaining_global_mem / 10; // We leave 10% of global memory free for CUDA allocations, not sure if it is useful though.
  CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, config.stack_kb*1000));
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, remaining_global_mem));
  print_memory_statistics("stack_memory", total_stack_size);
  print_memory_statistics("heap_memory", remaining_global_mem);
  printf("%% and_nodes=%zu\n", config.and_nodes);
  printf("%% or_nodes=%zu\n", config.or_nodes);
  int num_blocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, (void*) gpu_propagate, config.and_nodes, shared_mem_bytes);
  printf("%% max_blocks_per_sm=%d\n", num_blocks);
}

#endif // __CUDACC__
#endif // TURBO_HYBRID_DIVE_AND_SOLVE_HPP
