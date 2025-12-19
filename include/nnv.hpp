// Copyright 2025 Yi-Nung Tsao 

#ifndef TURBO_NNV_HPP 
#define TURBO_NNV_HPP

#include "lala/onnx_parser.hpp"
#include "lala/smt_parser.hpp" 

namespace lala { 

namespace impl {

template<class Allocator> 
class NNV {
	using allocator_type = Allocator;
	using F = TFormula<allocator_type>;
	using FSeq = typename F::Sequence;

public:
	battery::shared_ptr<F, allocator_type> make_nnv_formulas(const std::string& onnx_path, const std::string& vnnlib_path) {
		FSeq seq; 
		seq.push_back(std::move(parse_onnx<allocator_type>(onnx_path)));
		seq.push_back(std::move(parse_smt<allocator_type>(vnnlib_path)));
		return battery::make_shared<F, allocator_type>(std::move(F::make_nary(AND, std::move(seq))));
	} 
};
} // namespace impl

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_nnv(const std::string& onnx_path, const std::string& vnnlib_path) {
	impl::NNV<Allocator> nnv;
	return nnv.make_nnv_formulas(onnx_path, vnnlib_path);
}

} // namespace lala 

#endif