// Copyright 2022 Pierre Talbot

#include "config.hpp"
#include "statistics.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>

void usage_and_exit(const std::string& program_name) {
  std::cout << "usage: " << program_name << " [-t 2000] [-timeout_s 2] [-a] [-n 10] [-i] [-f] [-s] [-v] [-p <i>] [-arch <cpu|hybrid|gpu|barebones|fbarebones>] [-p 48] [-or 48] [-sub 12] [-stack 100] [-fp <ac1|wac1>] [-wac1_threshold 0] [-jet_affine <v1|v2>] [-jet_interval_prop <fixpoint|forward|forward_backward>] [-jet_symbolic] [-jet_sym_rounds 0] [-jet_sym_layout <slot|neuron>] [-pir_fdeduce <v1|v2>] [-split_on <input|relu|mixed>] [-relu_order <earliest|widest|smear|abs>] [-relu_score <smear|gap>] [-relu_leaf <input|point>] [-jet_sym_clip_hidden] [-jet_sym_root_only] [-jet_sym_skip_final_interval] [-jet_candidates] [-jet_cand_every 0] [-jet_cand_grad <on|off>] [-jet_disjuncts <interleaved|sequential>] [-jet_leaf_check <strict|lenient>] [-jet_backward <off|root|spec|all>] [-jet_linear <both|forward|backward|auto>] [-jet_linear_n0 256] [-jet_bwd_rows 32] [-jet_bwd_every 0] [-jet_bwd_rounds 4] [-jet_bwd_clip] [-jet_bwd_et] [-under <jet|scip>] [-scip_mode <observe|race>] [-scip_cex <exact|float>] [-scip_threads 4] [-scip_max_unstable 20] [-scip_time 10] [-scip_time_ratio 0.02] [-scip_time_min 0.5] [-scip_time_max 30] [-scip_margin 1e-6] [-scip_queue 1024] [-jet_var_select <default|maxdom|rr|smear|abs>] [-abs_beta 0.5] [-abs_psi <std|cv>] [-abs_grad <interval|relax>] [-abs_dive_count <on|off>] [-forward_only] [-eps_var_order <input_order|first_fail|anti_first_fail|smallest|largest>] [-eps_value_order <min|max|split|reverse_split>] [-seed 0] [-network_analysis] [-cutnodes 0] [-disable_simplify] [-force_ternarize] [-globalmem] [-version 1.0.0] [xcsp3instance.xml | fzninstance.fzn]" << std::endl;
  std::cout << "\t-t 2000: Run the solver with a timeout of 2000 milliseconds." << std::endl;
  std::cout << "\t-timeout 2000: Same as -t, but if both -t and -timeout are specified, -timeout overrides -t." << std::endl;
  std::cout << "\t-timeout_s 2: The timeout in seconds (a decimal number, e.g. 116 or 0.5, as the timeouts of VNN-COMP); it overrides -t and -timeout." << std::endl;
  std::cout << "\t-a: Instructs the solver to report all solutions in the case of satisfaction problems, or print intermediate solutions of increasing quality in the case of optimisation problems." << std::endl;
  std::cout << "\t-n 10: Instructs the solver to stop after reporting 10 solutions (only used with satisfaction problems)." << std::endl;
  std::cout << "\t-i: Instructs the solver to print intermediate solutions of increasing quality (only used with optimisation problems)." << std::endl;
  std::cout << "\t-f: Instructs the solver to conduct a “free search”, i.e., ignore any search annotations. The solver is not required to ignore the annotations, but it is allowed to do so." << std::endl;
  std::cout << "\t-s: Print statistics during and after the search for solutions." << std::endl;
  std::cout << "\t-v: Print log messages (verbose solving) to the standard error stream." << std::endl;
  std::cout << "\t-ast: Print the AST of the model (useful to debug)." << std::endl;
  std::cout << "\t-p 48: On CPU, multithreading is not yet implemented. On GPU, equivalent to `-or 48`." << std::endl;
  std::cout << "\t-arch <cpu|gpu|hybrid|barebones|fbarebones>: Choose the architecture on which the problem will be solved." << std::endl;
  std::cout << "\t-fp <ac1|wac1>: Choose the fixpoint strategy (default: ac1 on CPU, wac1 on GPU):" << std::endl;
  std::cout << "\t\t ac1: All propagators are executed in parallel at each iteration." << std::endl;
  std::cout << "\t\t wac1: Behave as ac1 when the number of active propagators is less than wac1_threshold. Otherwise,  each warp must reach a local fixpoint before executing the next 32 propagators (not compatible with -arch cpu)." << std::endl;
  std::cout << "\t-wac1_threshold 4096: Threshold below which we select AC1 instead of WAC1 (default: 0)." << std::endl;
  std::cout << "\t-jet_affine <v1|v2>: Deduction of the affine neurons with -arch jet (default: v1):" << std::endl;
  std::cout << "\t\t v1: The original propagator of JET." << std::endl;
  std::cout << "\t\t v2: Same projection with less arithmetic, and the backward pass skipped when no term can be tightened (see include/jet_affine_v2.hpp)." << std::endl;
  std::cout << "\t-jet_interval_prop <fixpoint|forward|forward_backward>: The interval propagation of -arch jet (default: fixpoint):" << std::endl;
  std::cout << "\t\t fixpoint: The deductions of the neurons project forward and backward (affine, residual and ReLU deductions)." << std::endl;
  std::cout << "\t\t forward: Forward only: the deduction of a neuron never tightens the neurons it reads (as -forward_only), but the symbolic rounds of -jet_symbolic still clip the inputs. The comparisons of the property still bound the output neurons. -forward_only implies it." << std::endl;
  std::cout << "\t\t forward_backward: No fixpoint: one forward sweep (the layers in order, each neuron computed from the layers it reads, forward projections only), the comparisons of the property once, then one backward sweep (the layers in reverse order, each neuron narrows the layers it reads); the narrowed inputs are not propagated forward again. With -jet_symbolic, it replaces the interval fixpoint before and between the symbolic rounds (which clip)." << std::endl;
  std::cout << "\t-jet_symbolic: With -arch jet, alternate the interval fixpoint with symbolic rounds: affine bounds over the inputs, DeepPoly ReLU relaxation and input clipping (see include/jet_symbolic.hpp). With -arch fbarebones, it is ignored with a warning." << std::endl;
  std::cout << "\t-jet_sym_rounds 0: Maximum number of symbolic rounds per search node, 0 to continue until a round changes nothing (default: 0)." << std::endl;
  std::cout << "\t-jet_sym_layout <slot|neuron>: Memory layout of the symbolic forms of -jet_symbolic, both give the same results (default: slot):" << std::endl;
  std::cout << "\t\t slot: A slot of all the neurons is contiguous, so that the threads of a warp access consecutive addresses." << std::endl;
  std::cout << "\t\t neuron: All the slots of a neuron are contiguous (the first version)." << std::endl;
  std::cout << "\t-pir_fdeduce <v1|v2>: Deduction of the bytecodes with -arch fbarebones, both give bitwise the same results (default: v1):" << std::endl;
  std::cout << "\t\t v1: PIR::fdeduce." << std::endl;
  std::cout << "\t\t v2: PIR::fdeduce_v2 (lala-pc): for x = c * y with a constant c, the same forward and backward projections as PIR::fdeduce, with 2 products instead of 8 forward and 2 divisions instead of 8 for y = x / c." << std::endl;
  std::cout << "\t-split_on <input|relu|mixed>: With -arch jet or fbarebones, the variables the search branches on (default: input):" << std::endl;
  std::cout << "\t\t input: The input neurons, at the middle of their domain (the original search)." << std::endl;
  std::cout << "\t\t relu: The phase of a ReLU y = max(p, 0) whose phase is unknown: p <= 0, then p >= 0 (see include/relu_split.hpp). When every ReLU has a phase, the search continues as chosen by -relu_leaf." << std::endl;
  std::cout << "\t\t mixed: (-arch jet only) The inputs and the ReLUs of unknown phase together, as one set of variables, chosen by -jet_var_select smear or abs (abs if another one is given): the score of an input is half its smear (a bisection halves its width), that of a ReLU is given by -relu_score; with abs, the restricted candidate list is over both, and in it the variable bisected the fewest times on the branch (a ReLU: never), then the highest score. -relu_leaf is not used: when no ReLU is left, the inputs are split until their width is at most -epsilon." << std::endl;
  std::cout << "\t-relu_order <earliest|widest|smear|abs>: With -split_on relu, the ReLU to split (default: earliest):" << std::endl;
  std::cout << "\t\t earliest: The first one in the order of the network (first layer first)." << std::endl;
  std::cout << "\t\t widest: The one whose pre-activation (-arch fbarebones) or output (-arch jet) has the widest domain." << std::endl;
  std::cout << "\t\t smear: (-arch jet only) The one of maximum score (-relu_score), the earliest one on a tie; the derivatives are given by -abs_grad." << std::endl;
  std::cout << "\t\t abs: (-arch jet only) The adaptive bisection of Granvilliers (CP 2012) over the ReLUs: the restricted candidate list of the ReLUs of score at least s_l + alpha (s_u - s_l), alpha = 1 / (1 + beta psi) (-abs_beta, -abs_psi), and in it the one of the earliest layer (a ReLU is split at most once on a branch, so the fewest bisections of the paper would not choose), then the highest score, then the earliest one." << std::endl;
  std::cout << "\t-relu_score <smear|gap>: With -relu_order smear or abs, or -split_on mixed, the score of a ReLU of unknown phase, g * m, g the largest |d(row)/d(y)| over the rows of the property (outputs with a bound and comparisons) (default: smear):" << std::endl;
  std::cout << "\t\t smear: m is the width of y (the smear of the variable y)." << std::endl;
  std::cout << "\t\t gap: m is the largest gap of the relaxation of the ReLU at the node: the width of y when the node ran no linear relaxation (interval propagation only), -l u / (u - l) when it ran the symbolic pass or the backward contractor (l < 0 < u the bounds of the pre-activation)." << std::endl;
  std::cout << "\t-relu_leaf <input|point>: With -split_on relu, the search once every ReLU has a phase (default: input):" << std::endl;
  std::cout << "\t\t input: The input split of -split_on input (-var_order, -value_order): the inputs are split until their width is at most -epsilon, and a solution is then looked for with the underapproximation of the value order (e.g. split: the middle of the box)." << std::endl;
  std::cout << "\t\t point: The inputs are fixed to points at once with the underapproximation of the value order: split, all at their middle; lb_split, ub_split, mid_split, one input at a time (chosen by -var_order) at its lower bound, upper bound or middle; mid_lb_split, mid_ub_split, one input at a time at its middle, then at its lower (upper) bound. The node is a solution if the point is one, and is counted unknown otherwise (the input space is not split)." << std::endl;
  std::cout << "\t-jet_sym_clip_hidden: With -jet_symbolic, also clip the inputs with the forms of the hidden neurons whose bounds are tighter than their forms, e.g. the phase of a ReLU split (see clip_hidden in include/jet_symbolic.hpp)." << std::endl;
  std::cout << "\t-jet_sym_skip_final_interval: With -jet_symbolic and -jet_sym_rounds k > 0, no interval propagation after the k-th symbolic round: e.g. with -jet_sym_rounds 1, a node is the interval propagation, one symbolic round (forms and clipping), and nothing more; the clipped inputs are not propagated to the other neurons (-arch jet). By default, the interval propagation follows every symbolic round that changes a domain." << std::endl;
  std::cout << "\t-jet_candidates: With -arch jet, test candidate points of the box of a node as counterexamples before the search goes on, without making them nodes of the search tree (see include/jet_candidates.hpp): the middle of the box (within the vnnlib bounds rounded inward), and, when the node ran a symbolic round (-jet_symbolic), one corner per constraint of the counterexample region, chosen by the signs of the coefficients of its symbolic forms. A point is reported only if the outward-rounded evaluation of the network on it satisfies every constraint of the region on the whole interval of its neurons (strict test); otherwise the search goes on unchanged. With -jet_sym_root_only, the nodes do not run symbolic rounds: only the middle is tested." << std::endl;
  std::cout << "\t-jet_disjuncts <interleaved|sequential>: With -arch jet, how the disjuncts of a disjunctive property (expanded into DNF) are searched (default: interleaved):" << std::endl;
  std::cout << "\t\t interleaved: One search for all the disjuncts: the subproblems of the disjuncts are taken in turn by the blocks, so that every disjunct progresses from the start and a counterexample of one disjunct is not delayed by the search of another one. The answer is sat as soon as a block finds a counterexample, unsat when all the subproblems of all the disjuncts are done." << std::endl;
  std::cout << "\t\t sequential: One search per disjunct, in the order of the property, each one finished before the next one starts (with -jet_candidates, the candidates of the roots of all the disjuncts are tested first)." << std::endl;
  std::cout << "\t-jet_leaf_check <strict|lenient>: With -arch jet, how a leaf of the search (the variables of the strategy are points and the propagation is not empty) is decided (default: strict):" << std::endl;
  std::cout << "\t\t strict: its point (the middle of the node within the vnnlib bounds rounded inward) is reported as a counterexample only if the network evaluated on it in double precision, rounded outward, satisfies every constraint of the counterexample region on the whole interval of its neurons (the strict test of -jet_candidates, see include/jet_candidates.hpp); otherwise the leaf is counted unknown and the search goes on." << std::endl;
  std::cout << "\t\t lenient: the leaf is a counterexample as soon as its propagation is not empty, which only says that the outward-rounded intervals of the store meet the region (the behaviour before this option): a point near the boundary of the region may be reported although it is not a counterexample." << std::endl;
  std::cout << "\t-jet_cand_grad <on|off>: With -jet_candidates, in a node without the symbolic forms of the node (no -jet_symbolic, -jet_sym_root_only), the corners are chosen by the signs of the middle of the interval Jacobian of their rows (include/jet_jacobian.hpp) instead of not being tested (default: on)." << std::endl;
  std::cout << "\t-jet_cand_every k: With -jet_candidates, test the candidates at the nodes of depth multiple of k (default: 0, at depth 0 only, i.e. the nodes of the dive to each subproblem)." << std::endl;
  std::cout << "\t-jet_backward <off|root|spec|all>: With -arch jet, the backward contractor (see include/jet_backward.hpp): back-substitution of linear rows through the ReLU and sigmoid relaxations (CROWN), alternated with the interval fixpoint, keeping no symbolic form (default: off):" << std::endl;
  std::cout << "\t\t root: On the root of every disjunct, once, before the search: the rows of the unstable neurons (layer by layer), of the output neurons and of the comparisons of the property." << std::endl;
  std::cout << "\t\t spec: As root, then in every node the rows of the output neurons and of the comparisons only (the hidden neurons keep the bounds of the root and of the interval propagation)." << std::endl;
  std::cout << "\t\t all: As spec, and all the rows in the nodes of depth 0 or multiple of -jet_bwd_every." << std::endl;
  std::cout << "\t-jet_linear <both|forward|backward|auto>: With -arch jet, which linear-relaxation contractors may run (default: both, each one as enabled by -jet_symbolic and -jet_backward): forward disables the backward contractor, backward disables the symbolic pass, auto keeps the symbolic pass when the network has at most -jet_linear_n0 inputs (default 256) and the backward contractor otherwise." << std::endl;
  std::cout << "\t-jet_bwd_rows R: With -jet_backward, the number of rows of one node back-substituted together by the threads of a block (a group of rows sharing the reads of the weights, not a batch of nodes), even (default: 32)." << std::endl;
  std::cout << "\t-jet_bwd_every k: With -jet_backward all, all the rows in the nodes of depth multiple of k, the rows of the property only in the others (default: 0, depth 0 only)." << std::endl;
  std::cout << "\t-jet_bwd_rounds k: With -jet_backward, the maximal number of backward passes in the fixpoint of a node, each one followed by the interval fixpoint when it changed the store (default: 4)." << std::endl;
  std::cout << "\t-jet_bwd_clip: With -jet_backward, clip the inputs with the rows that reached them (Clip-and-Verify): sum a_k x_k + c <= row <= its upper bound in the store." << std::endl;
  std::cout << "\t-jet_bwd_et: With -jet_backward, stop the back-substitution of the row of a neuron as soon as its bound decides the phase of the neuron (GPUPoly-like early termination)." << std::endl;
  std::cout << "\t-under <jet|scip>: With -arch jet, how a counterexample is looked for in the nodes (default: jet):" << std::endl;
  std::cout << "\t\t jet: the underapproximation of JET: the candidate points of -jet_candidates and the strict test of the leaves (-jet_leaf_check)." << std::endl;
  std::cout << "\t\t scip: the first node of each path of the search with at most -scip_max_unstable unstable ReLUs, whose candidate points (-jet_candidates) were tested without counterexample, is copied to a queue and solved by SCIP in exact mode on the CPU (see include/jet_scip.hpp), by -scip_threads worker processes in parallel, while the GPU goes on with its search (it does not wait for SCIP). A point found by SCIP is a counterexample only if it passes the strict test. With -scip_mode race (the default), a node that SCIP proves without counterexample ends the search of its subtree by the GPU, and the unknown leaves under it are not counted. The strict test of the leaves stays. Requires a build with -DWITH_SCIP=ON." << std::endl;
  std::cout << "\t-scip_threads n: With -under scip, the number of SCIP workers (default: 4)." << std::endl;
  std::cout << "\t-scip_max_unstable k: With -under scip, a node is sent to SCIP when it has at most k unstable ReLUs and no ancestor was sent (default: 20)." << std::endl;
  std::cout << "\t-scip_cex <exact|float>: With -under scip, how a solution of SCIP becomes a counterexample (default: exact):" << std::endl;
  std::cout << "\t\t exact: the exact rational solution is a counterexample of the real-valued network as it is, when it lies in the region of the property rounded inward (checked in rational arithmetic); it is printed exactly, and its neurons are reported rounded outward." << std::endl;
  std::cout << "\t\t float: the inputs are rounded to the nearest floats, and the point must pass the strict test of JET (double arithmetic rounded outward)." << std::endl;
  std::cout << "\t-scip_mode <observe|race>: With -under scip, what the answers of SCIP do (default: race):" << std::endl;
  std::cout << "\t\t observe: SCIP only looks for counterexamples; its other answers prune nothing." << std::endl;
  std::cout << "\t\t race: SCIP also proves the nodes: the MIP of a node is solved over the region of the property rounded outward, and a node without solution of margin >= 0 is closed (proved); the block searching its subtree then leaves it, and the unknown leaves under a closed node are not counted (also when SCIP closes it after the GPU left the subtree, at the end of the solving). The GPU and SCIP race on the subtree: the first one to decide it wins." << std::endl;
  std::cout << "\t-scip_time_ratio r, -scip_time_min a, -scip_time_max b: With -under scip, the time limit of SCIP on one node is clamp(r * T, a, b) seconds, T the timeout of the solving (-t); without timeout or with r = 0, -scip_time (default: 0.02, 0.5, 30)." << std::endl;
  std::cout << "\t-scip_time s: With -under scip, the time limit of SCIP on one node, in seconds (default: 10)." << std::endl;
  std::cout << "\t-jet_var_select <default|maxdom|rr|smear|abs>: With -arch jet, the input to bisect when the search splits the inputs (-split_on input, and below the leaves of -split_on relu -relu_leaf input) (default: default). Only the inputs wider than -epsilon whose interval has a float strictly inside are candidates:" << std::endl;
  std::cout << "\t\t default: the variable order of the strategy (-var_order), as before." << std::endl;
  std::cout << "\t\t maxdom: the widest candidate." << std::endl;
  std::cout << "\t\t rr: round-robin, the candidate bisected the fewest times on the current branch (then the lowest index)." << std::endl;
  std::cout << "\t\t smear: the maximum smear (Kearfott and Novoa, Algorithm 681 INTBIS, ACM TOMS 1990): max over the comparisons of the property of |d(comparison)/d(input)| times the width of the input, the derivatives as given by -abs_grad." << std::endl;
  std::cout << "\t\t abs: the adaptive bisection of Granvilliers (Adaptive Bisection of Numerical CSPs, CP 2012): the restricted candidate list of the candidates of smear at least s_l + alpha (s_u - s_l), alpha = 1 / (1 + beta psi), and in it the candidate bisected the fewest times on the current branch (then the highest smear, then the lowest index)." << std::endl;
  std::cout << "\t-abs_beta b: With -jet_var_select abs or -relu_order abs, beta in alpha = 1 / (1 + beta psi) (default: 0.5, the value of the paper)." << std::endl;
  std::cout << "\t-abs_psi <std|cv>: With -jet_var_select abs or -relu_order abs, psi is the standard deviation of the smear values of the candidates (std, the paper, default) or their coefficient of variation (cv: standard deviation over mean, independent of the scale of the outputs)." << std::endl;
  std::cout << "\t-abs_grad <interval|relax>: With -jet_var_select smear or abs, the derivatives: the interval Jacobian of the comparisons of the property over the box of the node (interval, the paper, default), or the coefficients of the symbolic forms of the node (relax, with -jet_symbolic; the interval Jacobian in a node without forms). With -relu_order smear or abs and -split_on mixed, relax is the coefficients of the linear relaxation back-substituted from the rows (CROWN, on the bounds of the pre-activations of the node), for the ReLUs and, with -split_on mixed, the inputs." << std::endl;
  std::cout << "\t-abs_dive_count <on|off>: With -jet_var_select rr or abs, the number of bisections of an input also counts those of the dive, estimated as log2 of its width at the root of the disjunct over its current width (default: on)." << std::endl;
  std::cout << "\t-scip_margin m: With -under scip, SCIP maximizes the margin by which the point satisfies the property and stops at a margin of m (default: 1e-6); a point on the boundary fails the strict test." << std::endl;
  std::cout << "\t-scip_queue q: With -under scip, the number of slots of the queue from the GPU to SCIP (default: 1024, reduced to fit 256 MB); a node is dropped when its slot is busy." << std::endl;
  std::cout << "\t-jet_sym_root_only: With -jet_symbolic, do the symbolic rounds on the root node only, once before the search; the nodes of the search run the interval fixpoint only (-arch jet)." << std::endl;
  std::cout << "\t-forward_only: With -arch jet or fbarebones, propagate forward only: the deduction of a neuron never tightens the neurons it reads (no backward projection of the affine, residual and ReLU deductions), and the symbolic rounds of -jet_symbolic clip nothing: the forms still tighten the bounds of the neurons, but the property only checks emptiness with them, it never narrows the inputs or the hidden neurons. The comparisons of the property still bound the output neurons. With -arch fbarebones, the bytecodes of the network project only onto the variable they compute (found by evaluating the bytecodes from the inputs), the bytecodes of the property keep all their projections, and the simplification of the model is disabled (-disable_simplify), since it runs every projection on the root. With -arch jet, -forward_only is -jet_interval_prop forward and, with -jet_symbolic, no clipping." << std::endl;
  std::cout << "\t-or 48: Run the subproblems on 48 streaming multiprocessors (SMs) (only for GPU architecture). Default: -or 0 for automatic selection of the number of SMs." << std::endl;
  std::cout << "\t-sub 12: Create 2^12 subproblems to be solved in turns by the blocks (embarrasingly parallel search). The special value `-1` leaves Turbo to decide on the number of subproblems (at least 30 * number of blocks). Default: -sub -1." << std::endl;
  std::cout << "\t-subfactor 300: Create B * 300 subproblems to be solved in turns by `B` blocks (embarrasingly parallel search). Default: -subfactor 300." << std::endl;
  std::cout << "\t-epsilon: Choose the epsilon value for the precision of the floating point intervals." << std::endl;
  std::cout << "\t-var_order <input_order|first_fail|anti_first_fail|smallest|largest|random>: Choose the variable ordering strategy for tree search." << std::endl;
  std::cout << "\t-value_order <indomain_min|indomain_max|indomain_split|indomain_reverse_split>: Choose the value ordering strategy for tree search." << std::endl;
  std::cout << "\t-eps_var_order <input_order|first_fail|anti_first_fail|smallest|largest|random>: Choose the variable ordering strategy for subproblems decomposition (default: same as main search strategy)." << std::endl;
  std::cout << "\t-eps_value_order <min|max|split|reverse_split>: Choose the value ordering strategy for subproblems decomposition (default: same as main search strategy)." << std::endl;
  std::cout << "\t-seed 0: Set the seed for the random number generator (default: 0)." << std::endl;
  std::cout << "\t-disable_network_analysis: Do not analyse the constraint network and output statistics." << std::endl;
  std::cout << "\t-stack 100: Use a maximum of 100KB of stack size per thread stored in global memory (only for GPU architectures)." << std::endl;
  std::cout << "\t-version 1.0.0: A version identifier that is printed as statistics to know which version of Turbo was used to solve an instance. It is only for documentation and replicability purposes." << std::endl;
  std::cout << "\t-hardware \"Intel Core i9-10900X@3.7GHz;24GO DDR4;NVIDIA RTX A5000\": The description of the hardware on which the solver is executed (\"CPU;RAM;GPU\"). It is only for documentation and replicability purposes." << std::endl;
  std::cout << "\t-cutnodes 1000: Stop the solver when 1000 nodes have been explored in a subproblem (0 for no limit)." << std::endl;
  std::cout << "Benchmarking options (should normally not be used):" << std::endl;
  std::cout << "\t-force_ternarize: Force the transformation of the formula in ternary normal form, even with IPC abstract domain (note that it is enabled by default with PIR abstract domain)." << std::endl;
  std::cout << "\t-disable_simplify: Disable the simplification step." << std::endl;
  std::cout << "\t-globalmem: Store all data abstract elements in the global memory and do not try to optimise using shared memory." << std::endl;
  std::cout << "\t-vnnlib_path <path>: Path to the VNNLIB file." << std::endl;
  std::cout << "\t-onnx_path <path>: Path to the ONNX file." << std::endl;
  exit(EXIT_FAILURE);
}

// https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
class InputParser {
private:
  std::string program_name;
  std::string version;
  std::vector <std::string> tokens;
  int tokens_read;
public:
  InputParser (int argc, char **argv): tokens_read(0), program_name(argv[0]) {
    for (int i=1; i < argc; ++i) {
      tokens.push_back(std::string(argv[i]));
    }
  }

  const std::string& getCmdOption(const std::string &option) const {
    std::vector<std::string>::const_iterator itr = std::find(tokens.begin(), tokens.end(), option);
    if (itr != tokens.end() && ++itr != tokens.end()){
      return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
  }

  bool cmdOptionExists(const std::string &option) const {
    return std::find(tokens.begin(), tokens.end(), option) != tokens.end();
  }

  bool read_size_t(const std::string& option, size_t& result) {
    const std::string& value = getCmdOption(option);
    if(!value.empty()) {
      sscanf(value.c_str(), "%zu", &result);
      tokens_read += 2;
      return true;
    }
    return false;
  }

  bool read_int(const std::string& option, int& result) {
    const std::string& value = getCmdOption(option);
    if(!value.empty()) {
      result = std::stoi(value);
      tokens_read += 2;
      return true;
    }
    return false;
  }

  bool read_double(const std::string& option, double& result) {
    const std::string& value = getCmdOption(option);
    if(!value.empty()) {
      result = std::stod(value);
      tokens_read += 2;
      return true;
    }
    return false;
  }

  bool read_bool(const std::string& option, bool& result) {
    result = cmdOptionExists(option);
    if(result) {
      ++tokens_read;
      return true;
    }
    return false;
  }

  bool read_occurrence(const std::string& option, int& result) {
    result = std::count(tokens.begin(), tokens.end(), option);
    if(result > 0) {
      tokens_read += result;
      return true;
    }
    return false;
  }

  bool read_string(const std::string& option, std::string& result) {
    result = getCmdOption(option);
    if(!result.empty()) {
      tokens_read += 2;
      return true;
    }
    return false;
  }

  void read_input_file(std::string& result) {
    if(tokens.size() <= tokens_read) {
      usage_and_exit(program_name);
    }
    result = tokens.back();
  }

#ifdef WITH_NNV
  void read_vnnlib_file(std::string& result){
    // Make sure there are still unread tokens
    if (tokens.size() <= tokens_read) {
        usage_and_exit(program_name);
    }
    // Take the next unread token
    result = tokens[tokens_read];
    ++tokens_read; // mark it as read
  }

  void read_onnx_file(std::string& result){
    read_input_file(result); 
  }
#endif 
};

Configuration<battery::standard_allocator> parse_args(int argc, char** argv) {
  Configuration<battery::standard_allocator> config;
  InputParser input(argc, argv);

  if(input.cmdOptionExists("-or") && input.cmdOptionExists("-p")) {
    std::cerr << "The options -or and -p cannot be used at the same time" << std::endl;
    usage_and_exit(argv[0]);
  }
  input.read_int("-sub", config.subproblems_power);
  input.read_size_t("-subfactor", config.subproblems_factor);
  input.read_size_t("-p", config.or_nodes);
  input.read_size_t("-or", config.or_nodes);
  input.read_size_t("-t", config.timeout_ms);
  input.read_size_t("-timeout", config.timeout_ms);
  /** `-timeout_s 116`: the timeout in seconds (e.g. the timeouts of VNN-COMP); it overrides -t and -timeout. */
  double timeout_s = -1.0;
  if(input.read_double("-timeout_s", timeout_s)) {
    if(timeout_s < 0) {
      std::cerr << "-timeout_s must be nonnegative" << std::endl;
      exit(EXIT_FAILURE);
    }
    config.timeout_ms = static_cast<size_t>(std::llround(timeout_s * 1000.0));
  }
  input.read_size_t("-stack", config.stack_kb);
  input.read_size_t("-n", config.stop_after_n_solutions);
  input.read_size_t("-cutnodes", config.stop_after_n_nodes);
  input.read_size_t("-seed", config.seed);
  input.read_bool("-i", config.print_intermediate_solutions);
  bool all_sols;
  input.read_bool("-a", all_sols);
  if(all_sols) {
    config.stop_after_n_solutions = 0;
    config.print_intermediate_solutions = true;
  }
  input.read_bool("-f", config.free_search);

  input.read_occurrence("-v", config.verbose_solving);
  input.read_bool("-ast", config.print_ast);
  input.read_bool("-s", config.print_statistics);
  input.read_bool("-globalmem", config.only_global_memory);
  input.read_bool("-disable_simplify", config.disable_simplify);
  input.read_bool("-force_ternarize", config.force_ternarize);
  input.read_bool("-disable_network_analysis", config.disable_network_analysis);

  std::string architecture;
  if(input.read_string("-arch", architecture)) {
    if(architecture == "cpu") {
      config.arch = Arch::CPU;
    }
    else if(architecture == "hybrid") {
      config.arch = Arch::HYBRID;
    }
    else if(architecture == "gpu") {
      config.arch = Arch::GPU;
    }
    else if(architecture == "barebones") {
      config.arch = Arch::BAREBONES;
    }
    else if (architecture == "fbarebones") {
      config.arch = Arch::FBAREBONES;
    }
    else if (architecture == "jet") {
      config.arch = Arch::JET;
    }
    else {
      std::cerr << "Unknown architecture -arch " << architecture << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string fixpoint;
  if(input.read_string("-fp", fixpoint)) {
    if(fixpoint == "ac1") {
      config.fixpoint = FixpointKind::AC1;
    }
    else if(fixpoint == "wac1") {
      config.fixpoint = FixpointKind::WAC1;
    }
    else {
      std::cerr << "Unknown fixpoint -fp " << fixpoint << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  input.read_size_t("-wac1_threshold", config.wac1_threshold);
  std::string jet_affine;
  if(input.read_string("-jet_affine", jet_affine)) {
    if(jet_affine == "v1") {
      config.jet_affine = 1;
    }
    else if(jet_affine == "v2") {
      config.jet_affine = 2;
    }
    else {
      std::cerr << "Unknown -jet_affine " << jet_affine << " (expected v1 or v2)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  input.read_bool("-jet_symbolic", config.jet_symbolic);
  input.read_int("-jet_sym_rounds", config.jet_sym_rounds);
  std::string jet_sym_layout;
  if(input.read_string("-jet_sym_layout", jet_sym_layout)) {
    if(jet_sym_layout == "neuron") {
      config.jet_sym_layout = 0;
    }
    else if(jet_sym_layout == "slot") {
      config.jet_sym_layout = 1;
    }
    else {
      std::cerr << "Unknown -jet_sym_layout " << jet_sym_layout << " (expected slot or neuron)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string jet_interval_prop;
  if(input.read_string("-jet_interval_prop", jet_interval_prop)) {
    if(jet_interval_prop == "fixpoint") {
      config.jet_interval_prop = JET_INTERVAL_FIXPOINT;
    }
    else if(jet_interval_prop == "forward") {
      config.jet_interval_prop = JET_INTERVAL_FORWARD;
    }
    else if(jet_interval_prop == "forward_backward") {
      config.jet_interval_prop = JET_INTERVAL_FORWARD_BACKWARD;
    }
    else {
      std::cerr << "Unknown -jet_interval_prop " << jet_interval_prop << " (expected fixpoint, forward or forward_backward)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string jet_disjuncts;
  if(input.read_string("-jet_disjuncts", jet_disjuncts)) {
    if(jet_disjuncts == "interleaved") {
      config.jet_disjuncts = JET_DISJUNCTS_INTERLEAVED;
    }
    else if(jet_disjuncts == "sequential") {
      config.jet_disjuncts = JET_DISJUNCTS_SEQUENTIAL;
    }
    else {
      std::cerr << "Unknown -jet_disjuncts " << jet_disjuncts << " (expected interleaved or sequential)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string jet_leaf_check;
  if(input.read_string("-jet_leaf_check", jet_leaf_check)) {
    if(jet_leaf_check == "strict") {
      config.jet_leaf_check = JET_LEAF_STRICT;
    }
    else if(jet_leaf_check == "lenient") {
      config.jet_leaf_check = JET_LEAF_LENIENT;
    }
    else {
      std::cerr << "Unknown -jet_leaf_check " << jet_leaf_check << " (expected strict or lenient)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string pir_fdeduce;
  if(input.read_string("-pir_fdeduce", pir_fdeduce)) {
    if(pir_fdeduce == "v1") {
      config.pir_fdeduce = 1;
    }
    else if(pir_fdeduce == "v2") {
      config.pir_fdeduce = 2;
    }
    else {
      std::cerr << "Unknown -pir_fdeduce " << pir_fdeduce << " (expected v1 or v2)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string split_on;
  if(input.read_string("-split_on", split_on)) {
    if(split_on == "input") {
      config.split_on = SPLIT_ON_INPUT;
    }
    else if(split_on == "relu") {
      config.split_on = SPLIT_ON_RELU;
    }
    else if(split_on == "mixed") {
      config.split_on = SPLIT_ON_MIXED;
    }
    else {
      std::cerr << "Unknown -split_on " << split_on << " (expected input, relu or mixed)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string relu_order;
  if(input.read_string("-relu_order", relu_order)) {
    if(relu_order == "earliest") {
      config.relu_order = RELU_ORDER_EARLIEST;
    }
    else if(relu_order == "widest") {
      config.relu_order = RELU_ORDER_WIDEST;
    }
    else if(relu_order == "smear") {
      config.relu_order = RELU_ORDER_SMEAR;
    }
    else if(relu_order == "abs") {
      config.relu_order = RELU_ORDER_ABS;
    }
    else {
      std::cerr << "Unknown -relu_order " << relu_order << " (expected earliest, widest, smear or abs)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string relu_score;
  if(input.read_string("-relu_score", relu_score)) {
    if(relu_score == "smear") { config.relu_score = RELU_SCORE_SMEAR; }
    else if(relu_score == "gap") { config.relu_score = RELU_SCORE_GAP; }
    else {
      std::cerr << "Unknown -relu_score " << relu_score << " (expected smear or gap)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string relu_leaf;
  if(input.read_string("-relu_leaf", relu_leaf)) {
    if(relu_leaf == "input") {
      config.relu_leaf = RELU_LEAF_INPUT;
    }
    else if(relu_leaf == "point") {
      config.relu_leaf = RELU_LEAF_POINT;
    }
    else {
      std::cerr << "Unknown -relu_leaf " << relu_leaf << " (expected input or point)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  input.read_bool("-jet_sym_clip_hidden", config.jet_sym_clip_hidden);
  input.read_bool("-jet_sym_root_only", config.jet_sym_root_only);
  input.read_bool("-jet_sym_skip_final_interval", config.jet_sym_skip_final_interval);
  input.read_bool("-jet_candidates", config.jet_candidates);
  input.read_int("-jet_cand_every", config.jet_cand_every);
  std::string cand_grad;
  if(input.read_string("-jet_cand_grad", cand_grad)) {
    if(cand_grad == "on") { config.jet_cand_grad = true; }
    else if(cand_grad == "off") { config.jet_cand_grad = false; }
    else {
      std::cerr << "Unknown -jet_cand_grad " << cand_grad << " (expected on or off)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string jet_backward;
  if(input.read_string("-jet_backward", jet_backward)) {
    if(jet_backward == "off") { config.jet_backward = JET_BACKWARD_OFF; }
    else if(jet_backward == "root") { config.jet_backward = JET_BACKWARD_ROOT; }
    else if(jet_backward == "spec") { config.jet_backward = JET_BACKWARD_SPEC; }
    else if(jet_backward == "all") { config.jet_backward = JET_BACKWARD_ALL; }
    else {
      std::cerr << "Unknown -jet_backward " << jet_backward << " (expected off, root, spec or all)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string jet_linear;
  if(input.read_string("-jet_linear", jet_linear)) {
    if(jet_linear == "both") { config.jet_linear = JET_LINEAR_BOTH; }
    else if(jet_linear == "forward") { config.jet_linear = JET_LINEAR_FORWARD; }
    else if(jet_linear == "backward") { config.jet_linear = JET_LINEAR_BACKWARD; }
    else if(jet_linear == "auto") { config.jet_linear = JET_LINEAR_AUTO; }
    else {
      std::cerr << "Unknown -jet_linear " << jet_linear << " (expected both, forward, backward or auto)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  input.read_int("-jet_linear_n0", config.jet_linear_n0);
  input.read_int("-jet_bwd_rows", config.jet_bwd_rows);
  input.read_int("-jet_bwd_every", config.jet_bwd_every);
  input.read_int("-jet_bwd_rounds", config.jet_bwd_rounds);
  input.read_bool("-jet_bwd_clip", config.jet_bwd_clip);
  input.read_bool("-jet_bwd_et", config.jet_bwd_et);
  if(config.jet_bwd_rows < 2 || config.jet_bwd_rows % 2 != 0) {
    const int r = config.jet_bwd_rows < 2 ? 2 : config.jet_bwd_rows + 1;
    std::cout << "% WARNING: -jet_bwd_rows must be even and at least 2, it is set to " << r << "." << std::endl;
    config.jet_bwd_rows = r;
  }
  if(config.jet_bwd_rounds < 1) {
    std::cout << "% WARNING: -jet_bwd_rounds must be at least 1, it is set to 1." << std::endl;
    config.jet_bwd_rounds = 1;
  }
  if(config.jet_bwd_every < 0) { config.jet_bwd_every = 0; }
  std::string under;
  if(input.read_string("-under", under)) {
    if(under == "jet") { config.under = UNDER_JET; }
    else if(under == "scip") { config.under = UNDER_SCIP; }
    else {
      std::cerr << "Unknown -under " << under << " (expected jet or scip)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  input.read_int("-scip_threads", config.scip_threads);
  input.read_int("-scip_max_unstable", config.scip_max_unstable);
  input.read_double("-scip_time", config.scip_time);
  input.read_double("-scip_margin", config.scip_margin);
  input.read_int("-scip_queue", config.scip_queue);
  std::string scip_mode;
  if(input.read_string("-scip_mode", scip_mode)) {
    if(scip_mode == "observe") { config.scip_mode = SCIP_MODE_OBSERVE; }
    else if(scip_mode == "race") { config.scip_mode = SCIP_MODE_RACE; }
    else {
      std::cerr << "Unknown -scip_mode " << scip_mode << " (expected observe or race)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string scip_cex;
  if(input.read_string("-scip_cex", scip_cex)) {
    if(scip_cex == "exact") { config.scip_cex_exact = true; }
    else if(scip_cex == "float") { config.scip_cex_exact = false; }
    else {
      std::cerr << "Unknown -scip_cex " << scip_cex << " (expected exact or float)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  input.read_double("-scip_time_ratio", config.scip_time_ratio);
  input.read_double("-scip_time_min", config.scip_time_min);
  input.read_double("-scip_time_max", config.scip_time_max);
  if(config.scip_time_ratio < 0) { config.scip_time_ratio = 0; }
  if(config.scip_time_min < 0) { config.scip_time_min = 0; }
  if(config.scip_time_max < config.scip_time_min) { config.scip_time_max = config.scip_time_min; }
  std::string var_select;
  if(input.read_string("-jet_var_select", var_select)) {
    if(var_select == "default") { config.jet_var_select = JET_VAR_DEFAULT; }
    else if(var_select == "maxdom") { config.jet_var_select = JET_VAR_MAXDOM; }
    else if(var_select == "rr") { config.jet_var_select = JET_VAR_RR; }
    else if(var_select == "smear") { config.jet_var_select = JET_VAR_SMEAR; }
    else if(var_select == "abs") { config.jet_var_select = JET_VAR_ABS; }
    else {
      std::cerr << "Unknown -jet_var_select " << var_select << " (expected default, maxdom, rr, smear or abs)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  input.read_double("-abs_beta", config.abs_beta);
  if(!(config.abs_beta >= 0.0)) {
    std::cout << "% WARNING: -abs_beta must be nonnegative, it is set to 0.5." << std::endl;
    config.abs_beta = 0.5;
  }
  std::string abs_psi;
  if(input.read_string("-abs_psi", abs_psi)) {
    if(abs_psi == "std") { config.abs_psi = ABS_PSI_STD; }
    else if(abs_psi == "cv") { config.abs_psi = ABS_PSI_CV; }
    else {
      std::cerr << "Unknown -abs_psi " << abs_psi << " (expected std or cv)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string abs_grad;
  if(input.read_string("-abs_grad", abs_grad)) {
    if(abs_grad == "interval") { config.abs_grad = ABS_GRAD_INTERVAL; }
    else if(abs_grad == "relax") { config.abs_grad = ABS_GRAD_RELAX; }
    else {
      std::cerr << "Unknown -abs_grad " << abs_grad << " (expected interval or relax)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::string abs_dive_count;
  if(input.read_string("-abs_dive_count", abs_dive_count)) {
    if(abs_dive_count == "on") { config.abs_dive_count = true; }
    else if(abs_dive_count == "off") { config.abs_dive_count = false; }
    else {
      std::cerr << "Unknown -abs_dive_count " << abs_dive_count << " (expected on or off)" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  if(config.scip_threads < 1) {
    std::cout << "% WARNING: -scip_threads must be at least 1, it is set to 1." << std::endl;
    config.scip_threads = 1;
  }
  if(config.scip_max_unstable < 0) { config.scip_max_unstable = 0; }
  if(!(config.scip_time > 0.0)) {
    std::cout << "% WARNING: -scip_time must be positive, it is set to 10." << std::endl;
    config.scip_time = 10.0;
  }
  if(!(config.scip_margin >= 0.0)) {
    std::cout << "% WARNING: -scip_margin must be nonnegative, it is set to 1e-6." << std::endl;
    config.scip_margin = 1e-6;
  }
  if(config.scip_queue < 1) { config.scip_queue = 1; }
#ifndef JET_WITH_SCIP
  if(config.under == UNDER_SCIP) {
    std::cerr << "-under scip requires Turbo built with SCIP (cmake -DWITH_SCIP=ON)." << std::endl;
    exit(EXIT_FAILURE);
  }
#endif
  input.read_bool("-forward_only", config.forward_only);
  /** With JET, `-forward_only` is `-jet_interval_prop forward` (and no clipping): the statistics show it. */
  if(config.forward_only && config.arch == Arch::JET) {
    if(config.jet_interval_prop == JET_INTERVAL_FORWARD_BACKWARD) {
      std::cout << "% WARNING: -forward_only overrides -jet_interval_prop forward_backward (it is -jet_interval_prop forward)." << std::endl;
    }
    config.jet_interval_prop = JET_INTERVAL_FORWARD;
  }
  input.read_double("-epsilon", config.epsilon);
  std::string var_order;
  if(input.read_string("-var_order", var_order)) {
    config.var_order = battery::string<battery::standard_allocator>(var_order.data());
  }
  std::string value_order;
  if(input.read_string("-value_order", value_order)) {
    config.value_order = battery::string<battery::standard_allocator>(value_order.data());
  }
  std::string eps_var_order;
  if(input.read_string("-eps_var_order", eps_var_order)) {
    config.eps_var_order = battery::string<battery::standard_allocator>(eps_var_order.data());
  }
  std::string eps_value_order;
  if(input.read_string("-eps_value_order", eps_value_order)) {
    config.eps_value_order = battery::string<battery::standard_allocator>(eps_value_order.data());
  }
  if(eps_var_order.empty() != eps_value_order.empty()) {
    printf("-eps_var_order and -eps_value_order must be specified together.\n");
    exit(EXIT_FAILURE);
  }
  std::string version;
  if(input.read_string("-version", version)) {
    config.version = battery::string<battery::standard_allocator>(version.data());
  }
  std::string hardware;
  if(input.read_string("-hardware", hardware)) {
    config.hardware = battery::string<battery::standard_allocator>(hardware.data());
  }

  std::string vnnlib_path;
  if(input.read_string("-vnnlib_path", vnnlib_path)) {
    config.vnnlib_path = battery::string<battery::standard_allocator>(vnnlib_path.data());
    std::string onnx_path;
    if(input.read_string("-onnx_path", onnx_path)) {
      config.onnx_path = battery::string<battery::standard_allocator>(onnx_path.data());
      config.problem_path = config.onnx_path;
    }
    else {
      std::cerr << "Either -onnx_path or -vnnlib_path must be specified" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  else { 
    std::string problem_path;
    input.read_input_file(problem_path);
    config.problem_path = battery::string<battery::standard_allocator>(problem_path.data());
  }

  return config;
}
