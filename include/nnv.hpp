// Copyright 2025 Yi-Nung Tsao 

#ifndef TURBO_NNV_HPP 
#define TURBO_NNV_HPP

#include "lala/onnx_parser.hpp"
#include "lala/smt_parser.hpp" 
#include "lala/solver_output.hpp"

namespace lala { 

namespace impl {

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
		seq.push_back(std::move(parse_smt<allocator_type>(vnnlib_path, output, is_nnv)));
		return battery::make_shared<F, allocator_type>(std::move(F::make_nary(AND, std::move(seq))));
	}

	/** Like `make_nnv_formulas`, but keeps the network equations and the (negated)
	 * postcondition as two SEPARATE formulas instead of pre-conjoining them.
	 * This is required to build a meet-free forward-only verification oracle
	 * (see AbstractDomains::setup_verification_oracle in common_solving.hpp):
	 * telling the postcondition into the SAME propagator set as the network
	 * allows a sound-but-not-exact forward enclosure of an auxiliary/output
	 * variable to be met against the goal region and collapse to a spurious
	 * non-bottom fixed point that does not correspond to any real solution.
	 */
	struct SplitFormulas {
		battery::shared_ptr<F, allocator_type> network;
		battery::shared_ptr<F, allocator_type> postcondition;
	};

	SplitFormulas make_nnv_formulas_split(const std::string& onnx_path, const std::string& vnnlib_path) {
		auto network = parse_onnx<allocator_type>(onnx_path, input_neurons, hidden_neurons, output);
		auto postcondition = parse_smt<allocator_type>(vnnlib_path, output, is_nnv);
		return SplitFormulas{
			battery::make_shared<F, allocator_type>(std::move(network)),
			battery::make_shared<F, allocator_type>(std::move(postcondition))
		};
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

/** Same as `parse_nnv`, but returns the network equations and the (negated)
 * postcondition as two separate formulas. See `impl::NNV::make_nnv_formulas_split`. */
template <class Allocator>
typename impl::NNV<Allocator>::SplitFormulas parse_nnv_split(const std::string& onnx_path, const std::string& vnnlib_path, battery::vector<std::string, Allocator>& input_neurons, battery::vector<std::string, Allocator>& hidden_neurons, SolverOutput<Allocator>& output, bool is_nnv) {
	impl::NNV<Allocator> nnv(input_neurons, hidden_neurons, output, is_nnv);
	return nnv.make_nnv_formulas_split(onnx_path, vnnlib_path);
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