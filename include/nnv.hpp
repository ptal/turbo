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
	SolverOutput<Allocator>& output;

public:
	NNV(battery::vector<std::string, Allocator>& input_neurons, SolverOutput<Allocator>& output, bool is_nnv): input_neurons(input_neurons), output(output), is_nnv(is_nnv) {}

	battery::shared_ptr<F, allocator_type> make_nnv_formulas(const std::string& onnx_path, const std::string& vnnlib_path) {
		FSeq seq; 
		seq.push_back(std::move(parse_onnx<allocator_type>(onnx_path, input_neurons, output)));
		seq.push_back(std::move(parse_smt<allocator_type>(vnnlib_path, output, is_nnv)));
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
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_nnv(const std::string& onnx_path, const std::string& vnnlib_path, battery::vector<std::string, Allocator>& input_neurons, SolverOutput<Allocator>& output, bool is_nnv) {
	impl::NNV<Allocator> nnv(input_neurons, output, is_nnv);
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