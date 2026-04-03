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

	SolverOutput<Allocator>& output;

public:
	NNV(SolverOutput<Allocator>& output): output(output) {}

	battery::shared_ptr<F, allocator_type> make_nnv_formulas(const std::string& onnx_path, const std::string& vnnlib_path) {
		FSeq seq; 
		seq.push_back(std::move(parse_onnx<allocator_type>(onnx_path, output)));
		seq.push_back(std::move(parse_smt<allocator_type>(vnnlib_path)));
		return battery::make_shared<F, allocator_type>(std::move(F::make_nary(AND, std::move(seq))));
	} 
};

template<class Allocator>
class SMT2 {
	using allocator_type = Allocator;
	using F = TFormula<allocator_type>;
	using FSeq = typename F::Sequence;

	SolverOutput<Allocator>& output;

public:
	SMT2(SolverOutput<Allocator>& output): output(output) {}

	battery::shared_ptr<F, allocator_type> make_smt2_formulas(const std::string& smt2_path) {
		return battery::make_shared<F, allocator_type>(std::move(parse_smt<allocator_type>(smt2_path)));
	}
};
} // namespace impl

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_nnv(const std::string& onnx_path, const std::string& vnnlib_path) {
	impl::NNV<Allocator> nnv;
	return nnv.make_nnv_formulas(onnx_path, vnnlib_path);
}

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_nnv(const std::string& onnx_path, const std::string& vnnlib_path, SolverOutput<Allocator>& output) {
	impl::NNV<Allocator> nnv(output);
	return nnv.make_nnv_formulas(onnx_path, vnnlib_path); 
}

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_smt2(const std::string& smt2_path) {
	impl::SMT2<Allocator> smt2;
	return smt2.make_smt2_formulas(smt2_path);
}

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_smt2(const std::string& smt2_path, SolverOutput<Allocator>& output) {
	impl::SMT2<Allocator> smt2(output);
	return smt2.make_smt2_formulas(smt2_path);
}

} // namespace lala 

#endif