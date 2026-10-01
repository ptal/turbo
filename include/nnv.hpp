// Copyright 2025 Yi-Nung Tsao 

#ifndef TURBO_NNV_HPP 
#define TURBO_NNV_HPP

#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <utility>

#include "lala/onnx_parser.hpp"
#include "lala/smt_parser.hpp" 
#include "lala/solver_output.hpp"

namespace lala { 

namespace impl {

/** The bounds `[lb, ub]` that a formula puts on each logical variable, when the formula is a unary bound
 * (`x <= c`, `x >= c`, `c <= x`, `x = c`, ...) or a conjunction of such bounds. Any other atom bounds
 * nothing, and a strict inequality is taken as the non-strict one, which only weakens the bound.
 * The constants are intervals `[c_lb, c_ub]` (see `string_to_real`), so we keep the end that widens the
 * bound: `c_ub` for an upper bound and `c_lb` for a lower bound. */
template <class F>
void collect_unary_bounds(const F& f, std::map<std::string, std::pair<double, double>>& bounds) {
	if(!f.is(F::Seq)) { return; }
	if(f.sig() == AND) {
		for(int i = 0; i < f.seq().size(); ++i) {
			collect_unary_bounds(f.seq(i), bounds);
		}
		return;
	}
	if(f.seq().size() != 2) { return; }
	const Sig sig = f.sig();
	if(sig != LEQ && sig != LT && sig != GEQ && sig != GT && sig != EQ) { return; }
	const F& lhs = f.seq(0);
	const F& rhs = f.seq(1);
	auto constant = [](const F& c, double& lb, double& ub) {
		if(c.is(F::R)) { lb = battery::get<0>(c.r()); ub = battery::get<1>(c.r()); return true; }
		if(c.is(F::Z)) { lb = ub = static_cast<double>(c.z()); return true; }
		return false;
	};
	double c_lb, c_ub;
	std::string name;
	bool var_on_left;
	if(lhs.is(F::LV) && constant(rhs, c_lb, c_ub)) { name = std::string(lhs.lv().data()); var_on_left = true; }
	else if(rhs.is(F::LV) && constant(lhs, c_lb, c_ub)) { name = std::string(rhs.lv().data()); var_on_left = false; }
	else { return; }
	/** `x <= c`, `x < c`, `c >= x` and `c > x` bound `x` from above; the mirrored forms from below. */
	const bool upper = (sig == LEQ || sig == LT) == var_on_left;
	auto it = bounds.emplace(name, std::make_pair(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())).first;
	if(sig == EQ || !upper) { it->second.first = std::max(it->second.first, c_lb); }
	if(sig == EQ || upper) { it->second.second = std::min(it->second.second, c_ub); }
}

/** A disjunction `(or D_1 ... D_n)` whose disjuncts all bound a variable `x` implies that `x` lies in the
 * hull of these bounds, `[min_i lb_i, max_i ub_i]`. We add the hull of every such variable as extra unary
 * bounds, for every disjunction at the top level of `vnnlib`, and return how many bounds were added.
 * This is what gives the input neurons a finite domain when the input region is a union of boxes, as in
 * ACAS Xu's `prop_6` (`(assert (or (and (<= X_0 ...) ...) (and (<= X_0 ...) ...)))`): otherwise the
 * inputs stay unbounded and the search cannot split them. The disjunction itself is kept, so it still
 * prunes the part of the hull that lies outside every box. */
template <class F>
int add_disjunctive_hulls(const F& vnnlib, typename F::Sequence& seq) {
	using LB = std::pair<double, double>;
	const double inf = std::numeric_limits<double>::infinity();
	int added = 0;
	auto visit = [&](const F& c) {
		if(!c.is(F::Seq) || c.sig() != OR || c.seq().size() == 0) { return; }
		std::map<std::string, LB> hull;
		for(int d = 0; d < c.seq().size(); ++d) {
			std::map<std::string, LB> bounds;
			collect_unary_bounds(c.seq(d), bounds);
			if(d == 0) { hull = std::move(bounds); continue; }
			/** A variable not bounded by this disjunct is unbounded in the hull (on that side). */
			for(auto& [name, b] : hull) {
				auto it = bounds.find(name);
				b.first = (it == bounds.end()) ? -inf : std::min(b.first, it->second.first);
				b.second = (it == bounds.end()) ? inf : std::max(b.second, it->second.second);
			}
		}
		for(const auto& [name, b] : hull) {
			if(b.first != -inf) {
				seq.push_back(F::make_binary(F::make_lvar(UNTYPED, LVar<typename F::allocator_type>(name.data())), GEQ, F::make_real(b.first, b.first)));
				++added;
			}
			if(b.second != inf) {
				seq.push_back(F::make_binary(F::make_lvar(UNTYPED, LVar<typename F::allocator_type>(name.data())), LEQ, F::make_real(b.second, b.second)));
				++added;
			}
		}
	};
	if(vnnlib.is(F::Seq) && vnnlib.sig() == AND) {
		for(int i = 0; i < vnnlib.seq().size(); ++i) { visit(vnnlib.seq(i)); }
	}
	else {
		visit(vnnlib);
	}
	return added;
}

template<class Allocator> 
class NNV {
	using allocator_type = Allocator;
	using F = TFormula<allocator_type>;
	using FSeq = typename F::Sequence;

	bool is_nnv;
	battery::vector<std::string, Allocator>& input_neurons;
	battery::vector<std::string, Allocator>& hidden_neurons;
	SolverOutput<Allocator>& output;

public:
	NNV(battery::vector<std::string, Allocator>& input_neurons, battery::vector<std::string, Allocator>& hidden_neurons, SolverOutput<Allocator>& output, bool is_nnv): input_neurons(input_neurons), hidden_neurons(hidden_neurons), output(output), is_nnv(is_nnv) {}

	battery::shared_ptr<F, allocator_type> make_nnv_formulas(const std::string& onnx_path, const std::string& vnnlib_path) {
		FSeq seq;
		seq.push_back(std::move(parse_onnx<allocator_type>(onnx_path, input_neurons, hidden_neurons, output)));
		F vnnlib = parse_smt<allocator_type>(vnnlib_path, output, is_nnv);
		/** The bounds implied by a disjunctive property, see `add_disjunctive_hulls`. */
		int added = add_disjunctive_hulls(vnnlib, seq);
		if(added > 0) {
			printf("%% Added %d bound(s) implied by the disjunctions of the property (hull of their boxes).\n", added);
		}
		seq.push_back(std::move(vnnlib));
		return battery::make_shared<F, allocator_type>(std::move(F::make_nary(AND, std::move(seq))));
	}
};

template<class Allocator>
class SMT2 {
	using allocator_type = Allocator;
	using F = TFormula<allocator_type>;
	using FSeq = typename F::Sequence;

	bool is_nnv;
	SolverOutput<Allocator>& output;

public:
	SMT2(SolverOutput<Allocator>& output, bool is_nnv): output(output), is_nnv(is_nnv) {}

	battery::shared_ptr<F, allocator_type> make_smt2_formulas(const std::string& smt2_path) {
		return battery::make_shared<F, allocator_type>(std::move(parse_smt<allocator_type>(smt2_path, output, is_nnv)));
	}
};
} // namespace impl

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_nnv(const std::string& onnx_path, const std::string& vnnlib_path) {
	impl::NNV<Allocator> nnv;
	return nnv.make_nnv_formulas(onnx_path, vnnlib_path);
}

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_nnv(const std::string& onnx_path, const std::string& vnnlib_path, battery::vector<std::string, Allocator>& input_neurons, battery::vector<std::string, Allocator>& hidden_neurons, SolverOutput<Allocator>& output, bool is_nnv) {
	impl::NNV<Allocator> nnv(input_neurons, hidden_neurons, output, is_nnv);
	return nnv.make_nnv_formulas(onnx_path, vnnlib_path);
}

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_smt2(const std::string& smt2_path) {
	impl::SMT2<Allocator> smt2;
	return smt2.make_smt2_formulas(smt2_path);
}

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_smt2(const std::string& smt2_path, SolverOutput<Allocator>& output, bool is_nnv) {
	impl::SMT2<Allocator> smt2(output, is_nnv);
	return smt2.make_smt2_formulas(smt2_path);
}

} // namespace lala 

#endif