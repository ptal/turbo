# TCN must-pass instances

The TCN format has no comment, so the purpose of each instance is given here (see `doc/tcn_parsing.md` in lala-parsing).
The expected optimum is the minimum of the objective variable; a maximization of `x` minimizes `m` with `0 = x + m`.
Turbo passes an instance marked **Known failure** only once the bug is fixed.

| File | Corner case | Optimum |
|---|---|---|
| `add.tcn` | x2 = x0 + x1 with positive domains. | 1 |
| `add_negative.tcn` | x2 = x0 + x1 with negative domains. | -8 |
| `add_aliased.tcn` | x1 = x0 + x0 (the same variable twice). | -6 |
| `add_self.tcn` | x0 = x0 + x1, which forces x1 = 0. | 0 |
| `add_as_sub.tcn` | a subtraction x3 = x1 - x0 written x1 = x0 + x3. | -6 |
| `mul_mixed_signs.tcn` | x2 = x0 * x1 with domains crossing zero. | -15 |
| `mul_square.tcn` | x1 = x0 * x0, maximized. | -9 |
| `mul_self_one.tcn` | x0 = x0 * x1 with x0 > 0, which forces x1 = 1. | 1 |
| `mul_self_zero.tcn` | x0 = x0 * x1 where x0 can be 0, which leaves x1 free. | -3 |
| `mul_divisors.tcn` | x0 * x1 = 6, minimizing x0 over negative divisors. | -6 |
| `tdiv_pos.tcn` | truncated division by a positive divisor. | -3 |
| `fdiv_pos.tcn` | floor division by a positive divisor (bounded quotient). | -4 |
| `fdiv_unbounded_pos.tcn` | floor division by a positive divisor with an unbounded quotient. **Known failure:** the floor division propagator does nothing when a bound is infinite, so the quotient is never fixed and turbo answers UNKNOWN. | -4 |
| `cdiv_pos.tcn` | ceil division by a positive divisor. | -3 |
| `ediv_pos.tcn` | Euclidean division by a positive divisor. | -4 |
| `tdiv_neg.tcn` | truncated division by a negative divisor. | -3 |
| `fdiv_neg.tcn` | floor division by a negative divisor (bounded quotient). | -3 |
| `fdiv_unbounded_neg.tcn` | floor division by a negative divisor with an unbounded quotient. **Known failure:** the floor division propagator does nothing when a bound is infinite, so the quotient is never fixed and turbo answers UNKNOWN. | -3 |
| `cdiv_neg.tcn` | ceil division by a negative divisor. | -4 |
| `ediv_neg.tcn` | Euclidean division by a negative divisor. | -4 |
| `tmod_pos.tcn` | truncated modulus by a positive divisor (sign of the dividend). **Known failure:** PIR has no modulus, so turbo cannot load the network. | -2 |
| `tmod_neg.tcn` | truncated modulus by a negative divisor (sign of the dividend). **Known failure:** PIR has no modulus, so turbo cannot load the network. | -2 |
| `fmod_pos.tcn` | floor modulus by a positive divisor (sign of the divisor). **Known failure:** PIR has no modulus, so turbo cannot load the network. | -2 |
| `fmod_neg.tcn` | floor modulus by a negative divisor (sign of the divisor). **Known failure:** PIR has no modulus, so turbo cannot load the network. | -2 |
| `cmod_pos.tcn` | ceil modulus by a positive divisor (opposite sign of the divisor). **Known failure:** PIR has no modulus, so turbo cannot load the network. | -2 |
| `cmod_neg.tcn` | ceil modulus by a negative divisor (opposite sign of the divisor). **Known failure:** PIR has no modulus, so turbo cannot load the network. | -2 |
| `emod_pos.tcn` | Euclidean modulus by a positive divisor (always nonnegative). **Known failure:** PIR has no modulus, so turbo cannot load the network. | -2 |
| `emod_neg.tcn` | Euclidean modulus by a negative divisor (always nonnegative). **Known failure:** PIR has no modulus, so turbo cannot load the network. | -2 |
| `div_by_zero.tcn` | the divisor domain contains 0, which has no quotient. | 1 |
| `max.tcn` | x2 = max(x0, x1). | -3 |
| `min.tcn` | x2 = min(x0, x1), maximized. | 2 |
| `min_max_sum.tcn` | max(x0,x1) + min(x0,x1) = 7. | 4 |
| `leq_true.tcn` | x2 = (x0 <= x1) with x2 = 1. | 2 |
| `leq_false.tcn` | x2 = (x0 <= x1) with x2 = 0. | -3 |
| `leq_free.tcn` | x2 = (x0 <= x1) with x2 free, used in the objective. | 1 |
| `eq_true.tcn` | x2 = (x0 == x1) with x2 = 1. | 4 |
| `eq_false.tcn` | x2 = (x0 == x1) with x2 = 0. | 4 |
| `reified_same_var.tcn` | x1 = (x0 == x0) and x2 = (x0 <= x0) are both true. | 2 |
| `reified_count.tcn` | exactly two of x0, x1, x2 are equal to 2 (a count with reified equalities). | 4 |
| `inf_bounds.tcn` | a variable with both bounds infinite, determined by the other ones. | -5 |
| `inf_objective.tcn` | an objective variable with infinite bounds. | -12 |
| `half_inf.tcn` | domains with one infinite bound. | 2 |
| `constant_objective.tcn` | the objective is a constant variable and there is no constraint. | 7 |
| `no_constraint.tcn` | no constraint: the optimum is the lower bound of the objective. | -3 |
| `large_values.tcn` | large values in additions and multiplications. | -600000 |
| `weighted_sum.tcn` | a weighted sum 1*x0 + 2*x1 + 3*x2 with constant coefficients and x0 + x1 + x2 = 6. | 8 |
| `cycle.tcn` | a cycle of constraints: x0 = x1 + x2 and x1 = x0 + x3, hence x3 = -x2. | -4 |
| `layout_one_line.tcn` | the whole file on a single line. | -12 |
| `layout_crlf_tabs.tcn` | Windows line endings, tabulations and blank lines. | -12 |
| `search_input_order_max.tcn` | a search strategy with the value order max. | -12 |
| `search_first_fail_split.tcn` | a search strategy first_fail / split. | -12 |
| `search_anti_first_fail_reverse_split.tcn` | a search strategy anti_first_fail / reverse_split. | -12 |
| `search_smallest_largest.tcn` | two search strategies (smallest, largest). | -12 |
| `search_random.tcn` | a search strategy random. | -12 |
| `search_empty.tcn` | a search strategy with no variable. | -12 |
| `dump_bug1.tcn` | a network dumped by turbo (`-dump_preprocessed_tcn`) from `test-must-pass/fzn/bug1.fzn`. | -10 |
| `dump_reified_in.tcn` | a network dumped by turbo (`-dump_preprocessed_tcn`) from `test-must-pass/fzn/reified_in.fzn`. | 1 |
| `dump_sudoku_opt2.tcn` | a network dumped by turbo (`-dump_preprocessed_tcn`) from `test-must-pass/fzn/sudoku_opt2.fzn (with a search strategy)`. | -2 |
| `dump_sudoku_opt2_max.tcn` | a network dumped by turbo (`-dump_preprocessed_tcn`) from `test-must-pass/fzn/sudoku_opt2_max.fzn (a maximization)`. | -2 |
| `dump_logic_xor.tcn` | a network dumped by turbo (`-dump_preprocessed_tcn`) from `test-must-pass/xcsp3/logic_xor.xml (a maximization)`. | -7 |
