// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_PARSING_HPP
#define TURBO_JET_PARSING_HPP

/** The input files of `-arch jet`, read on the host before the search:
 *   * `parse_network`: the ONNX file (`-onnx_path`) into the layers of `FastNNReluNetwork`: `Gemm` and
 *     `MatMul` (dense affine layers), `Conv` (2D, lowered into sparse affine layers, `jet_conv.hpp`),
 *     `Add`/`Sub`/`Mul` (by a constant: scale layers; `Add` of two tensors: residual layers), `Relu`,
 *     `Sigmoid`, and the operations that do not change the values (`Flatten`, `Reshape`, `Identity`,
 *     `Dropout`, `Constant`); anything else is refused (e.g. `BatchNormalization`, `MaxPool`);
 *   * `load_property`: the vnnlib file (`-vnnlib_path`) into a DNF over the neurons, and
 *     `apply_disjunct`: one disjunct into the root store and the comparisons of `FastProperty`.
 * The solver (`jet_dive_and_solve.hpp`) only uses these three functions. */

#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cfenv>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

#include "common_solving.hpp"
#include "jet_network.hpp"
#include "jet_conv.hpp"
#include <climits>

namespace jet {

#ifdef __CUDACC__
#ifndef TURBO_IPC_ABSTRACT_DOMAIN

namespace bt = ::battery;

/** Read a float initializer, whether it is stored in `float_data` or in `raw_data`.
 * Returns false if the tensor is not a float tensor. */
static bool read_float_tensor(const onnx::TensorProto& tensor, battery::vector<float>& out) {
  if(tensor.data_type() != onnx::TensorProto::FLOAT) {
    return false;
  }
  if(tensor.float_data_size() > 0) {
    for(const auto& v : tensor.float_data()) {
      out.push_back(v);
    }
    return true;
  }
  /** Most exported models store the initializers in `raw_data` instead of `float_data`. */
  if(!tensor.raw_data().empty()) {
    const std::string& raw = tensor.raw_data();
    if(raw.size() % sizeof(float) != 0) {
      return false;
    }
    size_t n = raw.size() / sizeof(float);
    for(size_t i = 0; i < n; ++i) {
      float v;
      std::memcpy(&v, raw.data() + i * sizeof(float), sizeof(float));
      out.push_back(v);
    }
    return true;
  }
  return false;
}

/** Read an integer initializer (`INT64`, or `INT32`), from `int64_data`, `int32_data` or `raw_data`.
 * Returns false otherwise. */
static bool read_int_tensor(const onnx::TensorProto& tensor, std::vector<int64_t>& out) {
  out.clear();
  if(tensor.data_type() == onnx::TensorProto::INT64) {
    if(tensor.int64_data_size() > 0) {
      for(const auto& v : tensor.int64_data()) { out.push_back(v); }
      return true;
    }
    const std::string& raw = tensor.raw_data();
    if(raw.size() % sizeof(int64_t) != 0) { return false; }
    for(size_t i = 0; i < raw.size() / sizeof(int64_t); ++i) {
      int64_t v;
      std::memcpy(&v, raw.data() + i * sizeof(int64_t), sizeof(int64_t));
      out.push_back(v);
    }
    return true;
  }
  if(tensor.data_type() == onnx::TensorProto::INT32) {
    if(tensor.int32_data_size() > 0) {
      for(const auto& v : tensor.int32_data()) { out.push_back(v); }
      return true;
    }
    const std::string& raw = tensor.raw_data();
    if(raw.size() % sizeof(int32_t) != 0) { return false; }
    for(size_t i = 0; i < raw.size() / sizeof(int32_t); ++i) {
      int32_t v;
      std::memcpy(&v, raw.data() + i * sizeof(int32_t), sizeof(int32_t));
      out.push_back(v);
    }
    return true;
  }
  return false;
}

/** Read the network of the ONNX file `config.onnx_path`. */
inline Network parse_network(const Configuration<battery::standard_allocator>& config) {
  std::ifstream input(config.onnx_path.data(), std::ios::in | std::ios::binary);
  onnx::ModelProto network;

  if (!network.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse onnx file." << std::endl;
    return Network();
  }

  const onnx::GraphProto& graph = network.graph();
  std::unordered_map<std::string, onnx::TensorProto> tensor_map;
  for (const auto& tensor : graph.initializer()) {
    tensor_map[tensor.name()] = tensor;
  }

  battery::vector<LayerInfo> layers;
  battery::vector<float> weights;
  battery::vector<float> biases;
  /** The rows of the sparse layers (`LAYER_SPARSE`, the lowered `Conv`), see `jet_conv.hpp`. */
  battery::vector<int> sp_row;
  battery::vector<int> sp_col;
  battery::vector<float> sp_val;
  int total_neurons = 0;
  /** The shape of a computed tensor without its batch dimension (e.g. `[C, H, W]` for an image), when it
   * is known: the graph input, the output of a `Conv`, a `Reshape` to a constant shape, and what keeps
   * the shape of its operand (an activation, an elementwise operation). A `Conv` needs the shape
   * `[C, H, W]` of its operand; the other operations ignore the shapes. */
  std::unordered_map<std::string, std::vector<int64_t>> tensor_shape;

  /** Maps the name of a tensor of the graph to the layer producing it. This is what makes a
   * residual network representable: a node names the tensors it reads, so a layer can take its
   * operands from anywhere above it rather than from the layer just before. */
  std::unordered_map<std::string, int> tensor_layer;

  if(graph.input_size() == 0) {
    std::cerr << "The onnx graph has no input." << std::endl;
    return Network();
  }

  // number of input neurons
  /** The input of the network is the first input of the graph that is not an initializer: older
   * exporters also list the initializers among the inputs of the graph (the `acasxu` models list
   * `input_AvgImg` and all the weights before `input`). */
  int input_index = -1;
  for(int i = 0; i < graph.input_size(); ++i) {
    if(tensor_map.find(graph.input(i).name()) == tensor_map.end()) { input_index = i; break; }
  }
  if(input_index == -1) {
    std::cerr << "The onnx graph has no input that is not an initializer." << std::endl;
    return Network();
  }
  const onnx::ValueInfoProto& graph_input = graph.input(input_index);
  const auto& input_shape = graph_input.type().tensor_type().shape();  // <batch_size, num_input_channels, input_H, input_W>;
  int64_t batch_size = input_shape.dim(0).dim_value() != 0 ? input_shape.dim(0).dim_value() : 1;
  int64_t input_channels = input_shape.dim().size() > 1 ? input_shape.dim(1).dim_value() : 1;
  int64_t input_height = input_shape.dim().size() > 2 ? input_shape.dim(2).dim_value() : 1;
  int64_t input_width = input_shape.dim().size() > 3 ? input_shape.dim(3).dim_value() : 1;
  int64_t input_dimensions = batch_size * input_channels * input_height * input_width;  // number of input neurons.

  layers.push_back(LayerInfo{LAYER_INPUT, 0, static_cast<int>(input_dimensions), -1, -1, 0, 0, 0});
  tensor_layer[graph_input.name()] = 0;
  {
    std::vector<int64_t> dims;
    for(int k = 1; k < input_shape.dim().size(); ++k) {
      dims.push_back(input_shape.dim(k).dim_value() != 0 ? input_shape.dim(k).dim_value() : 1);
    }
    if(dims.empty()) { dims.push_back(input_dimensions); }
    tensor_shape[graph_input.name()] = dims;
  }
  total_neurons += static_cast<int>(input_dimensions);

  /** `Constant` nodes carry their tensor as the attribute `value`: we record it as an initializer,
   * so that an elementwise node can read its constant from either. */
  for(const auto& node : graph.node()) {
    if(node.op_type() != "Constant" || node.output_size() == 0) { continue; }
    for(const auto& attr : node.attribute()) {
      if(attr.name() == "value" && attr.has_t()) { tensor_map[node.output()[0]] = attr.t(); }
    }
  }

  /** How many nodes (and graph outputs) read each tensor. A layer is only modified in place (a
   * bias folded into it, a ReLU applied to it) when the tensor it modifies has no other reader,
   * since that reader would then see the modified values instead of the ones of the graph. */
  std::unordered_map<std::string, int> num_readers;
  for(const auto& node : graph.node()) {
    for(const auto& name : node.input()) { ++num_readers[name]; }
  }
  for(const auto& out : graph.output()) { ++num_readers[out.name()]; }

  /** Whether a tensor is the output of the activation (ReLU or sigmoid) of its layer. A layer holds
   * a single value per neuron: once its activation is applied, its pre-activation is no longer
   * available, and a node reading that pre-activation cannot be represented. */
  std::unordered_map<std::string, bool> post_relu;
  post_relu[graph_input.name()] = false;
  /** Whether the bias block of a layer (the shift of a `LAYER_SCALE`) was already written. */
  std::vector<bool> bias_set(1, false);

  auto unsupported = [&](const onnx::NodeProto& node, const std::string& why) {
    std::cerr << "ERROR: the node `" << (node.output_size() > 0 ? node.output()[0] : std::string("?"))
              << "` (" << node.op_type() << ") is not supported: " << why << ".\n";
    return Network();
  };
  auto is_initializer = [&](const std::string& name) { return tensor_map.find(name) != tensor_map.end(); };

  /** The layer producing the computed tensor `name` (not an initializer). Fails if it is unknown, or
   * if it is the pre-activation of a layer whose ReLU was applied. */
  auto layer_of = [&](const std::string& name, int& out_layer) -> bool {
    auto it = tensor_layer.find(name);
    if(it == tensor_layer.end()) { return false; }
    out_layer = it->second;
    return post_relu[name] == (activation_of(layers[out_layer]) != ACT_NONE);
  };
  auto set_output = [&](const onnx::NodeProto& node, int layer, bool relu) {
    if(node.output_size() > 0) {
      tensor_layer[node.output()[0]] = layer;
      post_relu[node.output()[0]] = relu;
    }
  };
  /** The output of `node` has the shape `dims` (or the shape of the tensor `from`, if it is known). */
  auto set_shape = [&](const onnx::NodeProto& node, const std::vector<int64_t>& dims) {
    if(node.output_size() > 0) { tensor_shape[node.output()[0]] = dims; }
  };
  auto copy_shape = [&](const onnx::NodeProto& node, const std::string& from) {
    auto it = tensor_shape.find(from);
    if(node.output_size() > 0 && it != tensor_shape.end()) { tensor_shape[node.output()[0]] = it->second; }
  };
  auto add_layer = [&](const LayerInfo& L) {
    layers.push_back(L);
    bias_set.push_back(false);
    total_neurons += L.size;
    return static_cast<int>(layers.size()) - 1;
  };
  /** `out` receives the constant `name` broadcast to `size` values: it must hold one value, or
   * `size` values (whatever its shape, e.g. `(1, 1)` or `(1, size)`). */
  auto broadcast = [&](const std::string& name, int size, battery::vector<float>& out) -> bool {
    battery::vector<float> raw;
    if(!read_float_tensor(tensor_map[name], raw)) { return false; }
    out.clear();
    if(raw.size() == 1) { for(int k = 0; k < size; ++k) { out.push_back(raw[0]); } return true; }
    if(static_cast<int>(raw.size()) != size) { return false; }
    out = std::move(raw);
    return true;
  };
  auto all_equal = [](const battery::vector<float>& v, float c) {
    for(size_t k = 0; k < v.size(); ++k) { if(v[k] != c) { return false; } }
    return true;
  };
  /** A new `LAYER_SCALE` \f$ s \cdot src + t \f$ (elementwise). */
  auto add_scale = [&](int src, const battery::vector<float>& s, const battery::vector<float>& t) {
    const int weight_base = static_cast<int>(weights.size());
    const int bias_base = static_cast<int>(biases.size());
    for(size_t k = 0; k < s.size(); ++k) { weights.push_back(s[k]); }
    for(size_t k = 0; k < t.size(); ++k) { biases.push_back(t[k]); }
    return add_layer(LayerInfo{LAYER_SCALE, total_neurons, layers[src].size, src, -1, weight_base, bias_base, 0});
  };

  for (const auto& node : graph.node()) {
    const std::string& op = node.op_type();
    if(config.verbose_solving >= 2) {
      std::cout << "%% Node: " << (node.output_size() > 0 ? node.output()[0] : std::string("?"))
                << " | OpType: " << op << std::endl;
    }

    if(op == "Constant") { continue; }  // recorded above.

    if(op == "Gemm" || op == "MatMul") {
      /** `Gemm(A, B, C)` computes \f$ \alpha A \cdot B + \beta C \f$; we only read the plain
       * `A \cdot B + C`. */
      bool transB = false;
      for(const auto& attr : node.attribute()) {
        if(attr.name() == "transB") { transB = attr.i(); }
        else if(attr.name() == "transA" && attr.i() != 0) { return unsupported(node, "transA is not 0"); }
        else if((attr.name() == "alpha" || attr.name() == "beta") && attr.f() != 1.0f) {
          return unsupported(node, attr.name() + " is not 1");
        }
      }
      if(node.input().size() < 2) { return unsupported(node, "it has less than two inputs"); }
      /** The weight matrix is the constant among the first two inputs. A `MatMul` exported as
       * `MatMul(W, X)` carries it at the index `0` and computes \f$ W \cdot X \f$, so a weight
       * matrix found there is transposed as well. */
      int weight_index = -1;
      if(is_initializer(node.input()[1]) && !is_initializer(node.input()[0])) { weight_index = 1; }
      else if(is_initializer(node.input()[0]) && !is_initializer(node.input()[1])) { weight_index = 0; }
      else { return unsupported(node, "it does not multiply a computed tensor by a constant matrix"); }
      const onnx::TensorProto& tensor = tensor_map[node.input()[weight_index]];
      if(tensor.dims().size() != 2) { return unsupported(node, "its weights are not a 2D matrix"); }
      transB = transB || weight_index == 0;

      battery::vector<float> tmp_weights;
      if(!read_float_tensor(tensor, tmp_weights)) {
        std::cerr << "ERROR: The weights of `" << tensor.name() << "` are not stored as floats.\n";
        return Network();
      }

      /** `out_features` is the number of neurons of the new layer, `in_features` must match
       * the number of neurons of the layer this node reads from. */
      int64_t out_features = transB ? tensor.dims(0) : tensor.dims(1);
      int64_t in_features = transB ? tensor.dims(1) : tensor.dims(0);

      int src = -1;
      if(!layer_of(node.input()[1 - weight_index], src)) {
        return unsupported(node, "its operand is unknown, or is the pre-activation of a ReLU");
      }

      if(in_features != static_cast<int64_t>(layers[src].size)) {
        std::cerr << "ERROR: The weight matrix of `" << tensor.name() << "` expects " << in_features
                  << " inputs but the layer " << src << " it reads has " << layers[src].size
                  << " neurons.\n";
        return Network();
      }
      if(static_cast<int64_t>(tmp_weights.size()) != out_features * in_features) {
        std::cerr << "ERROR: The weight matrix of `" << tensor.name() << "` has "
                  << tmp_weights.size() << " entries instead of " << out_features * in_features << ".\n";
        return Network();
      }

      const int weight_base = static_cast<int>(weights.size());
      /** We always store the weights column-major, all the output neurons of a given input
       * being contiguous (see the layout conventions of `FastNNReluNetwork`). The ONNX layout
       * `[in_features, out_features]` is already in that order, so only a matrix given
       * transposed, as `[out_features, in_features]`, has to be rearranged. */
      if(transB) {
        for(int64_t c = 0; c < in_features; ++c) {
          for(int64_t r = 0; r < out_features; ++r) {
            weights.push_back(tmp_weights[r * in_features + c]);
          }
        }
      }
      else {
        for(int64_t k = 0; k < out_features * in_features; ++k) {
          weights.push_back(tmp_weights[k]);
        }
      }

      /** Every affine layer owns a block of biases, filled with zeros when the graph has none, so
       * that `deduce_neuron` can read `biases[bias_base + j]` unconditionally. */
      const int bias_base = static_cast<int>(biases.size());
      for(int64_t k = 0; k < out_features; ++k) { biases.push_back(0.0f); }

      const int l = add_layer(LayerInfo{LAYER_AFFINE, total_neurons, static_cast<int>(out_features),
        src, -1, weight_base, bias_base, 0});
      /** The `C` of a `Gemm`, broadcast to the layer. */
      if(op == "Gemm" && node.input().size() >= 3 && !node.input()[2].empty()) {
        battery::vector<float> c;
        if(!is_initializer(node.input()[2]) || !broadcast(node.input()[2], layers[l].size, c)) {
          return unsupported(node, "its bias is not a float constant of 1 or " + std::to_string(layers[l].size) + " values");
        }
        for(int k = 0; k < layers[l].size; ++k) { biases[bias_base + k] = c[k]; }
        bias_set[l] = true;
      }
      set_output(node, l, false);
      set_shape(node, std::vector<int64_t>{out_features});
      continue;
    }

    if(op == "Conv") {
      /** A 2D convolution, lowered into a sparse affine layer (`LAYER_SPARSE`, `jet_conv.hpp`): its
       * operand must be a computed tensor of known shape `[C, H, W]`, its kernel a float constant of 4
       * dimensions, its bias (optional) a float constant of one value per output channel. */
      if(node.input().size() < 2) { return unsupported(node, "it has less than two inputs"); }
      int src = -1;
      if(!layer_of(node.input()[0], src)) {
        return unsupported(node, "its operand is unknown, or is the pre-activation of a ReLU");
      }
      if(!is_initializer(node.input()[1])) { return unsupported(node, "its kernel is not a constant"); }
      const onnx::TensorProto& kernel = tensor_map[node.input()[1]];
      if(kernel.dims().size() != 4) { return unsupported(node, "it is not a 2D convolution (its kernel does not have 4 dimensions)"); }
      auto shape_it = tensor_shape.find(node.input()[0]);
      if(shape_it == tensor_shape.end() || shape_it->second.size() != 3) {
        return unsupported(node, "the shape [C, H, W] of its operand is unknown");
      }
      const std::vector<int64_t>& in = shape_it->second;
      if(in[0] * in[1] * in[2] != static_cast<int64_t>(layers[src].size)) {
        return unsupported(node, "the shape of its operand does not match the layer it reads");
      }
      conv::Conv2d c;
      c.cin = static_cast<int>(in[0]); c.h = static_cast<int>(in[1]); c.w = static_cast<int>(in[2]);
      c.cout = static_cast<int>(kernel.dims(0));
      c.kh = static_cast<int>(kernel.dims(2)); c.kw = static_cast<int>(kernel.dims(3));
      std::string auto_pad = "NOTSET";
      for(const auto& attr : node.attribute()) {
        const std::string& a = attr.name();
        if(a == "strides" && attr.ints_size() == 2) { c.sh = static_cast<int>(attr.ints(0)); c.sw = static_cast<int>(attr.ints(1)); }
        else if(a == "dilations" && attr.ints_size() == 2) { c.dh = static_cast<int>(attr.ints(0)); c.dw = static_cast<int>(attr.ints(1)); }
        else if(a == "pads" && attr.ints_size() == 4) {
          c.pt = static_cast<int>(attr.ints(0)); c.pl = static_cast<int>(attr.ints(1));
          c.pb = static_cast<int>(attr.ints(2)); c.pr = static_cast<int>(attr.ints(3));
        }
        else if(a == "group") { c.group = static_cast<int>(attr.i()); }
        else if(a == "auto_pad") { auto_pad = attr.s(); }
        else if(a == "kernel_shape") {
          if(attr.ints_size() != 2 || attr.ints(0) != c.kh || attr.ints(1) != c.kw) {
            return unsupported(node, "its kernel_shape does not match its kernel");
          }
        }
        else { return unsupported(node, "its attribute `" + a + "` is not supported"); }
      }
      if(auto_pad != "NOTSET" && auto_pad != "VALID" && !auto_pad.empty()) {
        return unsupported(node, "auto_pad " + auto_pad + " is not supported (only explicit pads)");
      }
      if(auto_pad == "VALID") { c.pt = c.pl = c.pb = c.pr = 0; }
      if(kernel.dims(1) * c.group != c.cin) {
        return unsupported(node, "its kernel does not match the channels of its operand");
      }
      int ho = 0, wo = 0;
      if(!conv::output_shape(c, ho, wo)) { return unsupported(node, "its parameters are invalid"); }
      battery::vector<float> kw;
      if(!read_float_tensor(kernel, kw) || static_cast<int64_t>(kw.size()) != kernel.dims(0) * kernel.dims(1) * kernel.dims(2) * kernel.dims(3)) {
        return unsupported(node, "its kernel is not a float tensor");
      }
      const int64_t size = static_cast<int64_t>(c.cout) * ho * wo;
      const int64_t max_nnz = size * (c.cin / c.group) * c.kh * c.kw;
      if(size > INT_MAX / 2 || static_cast<int64_t>(sp_col.size()) + max_nnz > static_cast<int64_t>(INT_MAX)) {
        return unsupported(node, "the lowered layer has too many weights (more than 2^31 in total)");
      }
      const int weight_base = static_cast<int>(sp_row.size());
      conv::lower(c, kw, sp_row, sp_col, sp_val);
      const int bias_base = static_cast<int>(biases.size());
      battery::vector<float> b;
      const bool has_bias = node.input().size() >= 3 && !node.input()[2].empty();
      if(has_bias) {
        if(!is_initializer(node.input()[2]) || !read_float_tensor(tensor_map[node.input()[2]], b) || static_cast<int>(b.size()) != c.cout) {
          return unsupported(node, "its bias is not a float constant of one value per output channel");
        }
      }
      for(int o = 0; o < c.cout; ++o) {
        for(int k = 0; k < ho * wo; ++k) { biases.push_back(has_bias ? b[o] : 0.0f); }
      }
      const int l = add_layer(LayerInfo{LAYER_SPARSE, total_neurons, static_cast<int>(size), src, -1, weight_base, bias_base, 0});
      bias_set[l] = has_bias;
      set_output(node, l, false);
      set_shape(node, std::vector<int64_t>{c.cout, ho, wo});
      if(config.verbose_solving >= 1) {
        printf("%% INFO: the Conv `%s` [%d, %d, %d] -> [%d, %d, %d] is a sparse layer of %lld weights.\n",
          node.output()[0].c_str(), c.cin, c.h, c.w, c.cout, ho, wo,
          static_cast<long long>(sp_row[weight_base + static_cast<int>(size)] - sp_row[weight_base]));
      }
      continue;
    }

    if(op == "Dropout") {
      /** At inference, `Dropout` is the identity (its optional mask output must not be read). */
      int alias = -1;
      if(node.input().size() == 0 || !layer_of(node.input()[0], alias)) {
        return unsupported(node, "its operand is unknown, or is the pre-activation of a ReLU");
      }
      if(node.output_size() > 1 && !node.output()[1].empty() && num_readers[node.output()[1]] > 0) {
        return unsupported(node, "its mask is read by another node");
      }
      set_output(node, alias, post_relu[node.input()[0]]);
      copy_shape(node, node.input()[0]);
      continue;
    }

    if(op == "Add" || op == "Sub" || op == "Mul") {
      if(node.input().size() != 2) { return unsupported(node, "it does not have two inputs"); }
      const bool c0 = is_initializer(node.input()[0]), c1 = is_initializer(node.input()[1]);
      if(c0 && c1) { return unsupported(node, "both of its operands are constants"); }
      if(!c0 && !c1) {
        /** An `Add` of two computed tensors is a residual connection: the two branches are summed
         * elementwise into a new layer. This is the shape of the `cersyve` models. */
        if(op != "Add") { return unsupported(node, "it combines two computed tensors"); }
        int a = -1, b = -1;
        if(!layer_of(node.input()[0], a) || !layer_of(node.input()[1], b)) {
          return unsupported(node, "an operand is unknown, or is the pre-activation of a ReLU");
        }
        if(layers[a].size != layers[b].size) {
          std::cerr << "ERROR: the residual `Add` combines two layers of different sizes ("
                    << layers[a].size << " and " << layers[b].size << ").\n";
          return Network();
        }
        set_output(node, add_layer(LayerInfo{LAYER_ADD, total_neurons, layers[a].size, a, b, 0, 0, 0}), false);
        copy_shape(node, node.input()[0]);
        continue;
      }
      /** A computed tensor and a constant: a bias, or a normalization of the inputs
       * (`cora_2024`: `x * 3.97 - 1.88`). */
      const int var_index = c0 ? 1 : 0;
      const std::string& var_name = node.input()[var_index];
      int src = -1;
      if(!layer_of(var_name, src)) { return unsupported(node, "its operand is unknown, or is the pre-activation of a ReLU"); }
      battery::vector<float> c;
      if(!broadcast(node.input()[1 - var_index], layers[src].size, c)) {
        return unsupported(node, "its constant is not a float tensor of 1 or " + std::to_string(layers[src].size) + " values");
      }
      const int n = layers[src].size;
      copy_shape(node, var_name);
      if(op == "Mul") {
        if(all_equal(c, 1.0f)) { set_output(node, src, post_relu[var_name]); continue; }  // x * 1 is x.
        battery::vector<float> zero(n, 0.0f);
        set_output(node, add_scale(src, c, zero), false);
        continue;
      }
      /** `x + c`, `x - c` and `c - x` as \f$ s \cdot x + t \f$; the negation is exact. */
      battery::vector<float> s(n, 1.0f), t(n);
      for(int k = 0; k < n; ++k) { t[k] = (op == "Sub" && var_index == 0) ? -c[k] : c[k]; }
      if(op == "Sub" && var_index == 1) { for(int k = 0; k < n; ++k) { s[k] = -1.0f; } }
      else if(all_equal(t, 0.0f)) { set_output(node, src, post_relu[var_name]); continue; }  // x + 0, x - 0 (the `Sub` of `acasxu`).
      /** The bias of an affine layer (`MatMul` then `Add`), or the shift of a scale (`Mul` then
       * `Add`), is folded into its layer when nothing else reads the layer before it. */
      if(s[0] == 1.0f && (layers[src].kind == LAYER_AFFINE || layers[src].kind == LAYER_SCALE || layers[src].kind == LAYER_SPARSE)
        && activation_of(layers[src]) == ACT_NONE && !bias_set[src] && num_readers[var_name] == 1)
      {
        for(int k = 0; k < n; ++k) { biases[layers[src].bias_base + k] = t[k]; }
        bias_set[src] = true;
        set_output(node, src, false);
        continue;
      }
      set_output(node, add_scale(src, s, t), false);
      continue;
    }

    if(op == "Relu") {
      /** A `Relu` is the activation of the layer producing the tensor it reads. We do not assume a
       * ReLU on every layer: `tllverifybench` alternates affine and ReLU layers, and
       * `acasxu`/`safenlp`/`cersyve` have an affine output layer. */
      int target = -1;
      if(node.input().size() != 1 || !layer_of(node.input()[0], target)) {
        return unsupported(node, "its operand is unknown, or is the pre-activation of a ReLU");
      }
      if(layers[target].sigmoid != 0) {
        return unsupported(node, "it applies to the output of a sigmoid");
      }
      if(layers[target].has_relu == 0 && num_readers[node.input()[0]] != 1) {
        return unsupported(node, "its pre-activation is also read by another node");
      }
      if(target == 0) { return unsupported(node, "it applies to the inputs"); }
      layers[target].has_relu = 1;
      set_output(node, target, true);
      copy_shape(node, node.input()[0]);
      continue;
    }

    if(op == "Sigmoid") {
      /** A `Sigmoid` is the activation of the layer producing the tensor it reads, as a `Relu`
       * (`jet_sigmoid.hpp`). Unlike the ReLU, it is not idempotent: a sigmoid of an activated
       * layer is refused. */
      int target = -1;
      if(node.input().size() != 1 || !layer_of(node.input()[0], target)) {
        return unsupported(node, "its operand is unknown, or is the pre-activation of an activation");
      }
      if(activation_of(layers[target]) != ACT_NONE) {
        return unsupported(node, "it applies to the output of an activation");
      }
      if(num_readers[node.input()[0]] != 1) {
        return unsupported(node, "its pre-activation is also read by another node");
      }
      if(target == 0) { return unsupported(node, "it applies to the inputs"); }
      layers[target].sigmoid = 1;
      set_output(node, target, true);
      copy_shape(node, node.input()[0]);
      continue;
    }

    if(op == "Flatten" || op == "Reshape" || op == "Identity") {
      /** These only change the shape of a tensor, not its values: the output is the layer they
       * read (the shape operand of `Reshape` is ignored). */
      int alias = -1;
      if(node.input().size() == 0 || !layer_of(node.input()[0], alias)) {
        return unsupported(node, "its operand is unknown, or is the pre-activation of a ReLU");
      }
      set_output(node, alias, post_relu[node.input()[0]]);
      /** The shape of the output, for a `Conv` reading it: `Identity` keeps it, `Flatten` makes it a
       * vector, `Reshape` takes its constant shape operand (unknown when the operand is computed). */
      if(op == "Identity") { copy_shape(node, node.input()[0]); }
      else if(op == "Flatten") { set_shape(node, std::vector<int64_t>{static_cast<int64_t>(layers[alias].size)}); }
      else {
        std::vector<int64_t> target, out;
        auto it = tensor_shape.find(node.input()[0]);
        if(node.input().size() >= 2 && is_initializer(node.input()[1]) && read_int_tensor(tensor_map[node.input()[1]], target)) {
          /** The input with its batch dimension; a vector of the size of the layer if its shape is unknown. */
          std::vector<int64_t> in(1, 1);
          if(it != tensor_shape.end()) { in.insert(in.end(), it->second.begin(), it->second.end()); }
          else { in.push_back(static_cast<int64_t>(layers[alias].size)); }
          if(conv::reshape(in, target, out) && !out.empty()) {
            /** The first dimension of the target is the batch, of size 1. */
            if(out.size() >= 2 && out[0] == 1) { out.erase(out.begin()); }
            set_shape(node, out);
          }
        }
      }
      if(config.verbose_solving >= 1) {
        printf("%% INFO: the node `%s` is treated as a pass-through.\n", op.c_str());
      }
      continue;
    }

    /** Any other operation would change the values of the network: we refuse it rather than
     * ignoring it, which would make every answer meaningless. */
    return unsupported(node, "this operation is not implemented by -arch jet");
  }

  if(layers.size() < 2) {
    std::cerr << "ERROR: No layer could be read from the onnx graph.\n";
    return Network();
  }

  /** The `Y_j` of the vnnlib file designate the neurons of the layer producing the output of the
   * graph, which in a DAG is not necessarily the last one in topological order. */
  int output_layer = static_cast<int>(layers.size()) - 1;
  if(graph.output_size() > 0) {
    auto it = tensor_layer.find(graph.output(0).name());
    if(it != tensor_layer.end()) {
      output_layer = it->second;
      if(post_relu[graph.output(0).name()] != (activation_of(layers[output_layer]) != ACT_NONE)) {
        std::cerr << "ERROR: the output of the graph is the pre-activation of an activated layer, which is not representable.\n";
        return Network();
      }
    }
    else {
      std::cerr << "%% WARNING: the output `" << graph.output(0).name()
                << "` of the graph was not produced by any layer, using the last layer instead.\n";
    }
  }

  /** `neuron_layer` gives the layer owning a neuron in O(1), which is what `deduce_neuron` uses in
   * place of walking a chain. */
  battery::vector<int> neuron_layer;
  neuron_layer.resize(static_cast<size_t>(total_neurons));
  for(int l = 0; l < static_cast<int>(layers.size()); ++l) {
    for(int k = 0; k < layers[l].size; ++k) {
      neuron_layer[layers[l].base + k] = l;
    }
  }

  return Network(total_neurons, output_layer, layers, neuron_layer, weights, biases, sp_row, sp_col, sp_val);
}

/** A minimal s-expression: an atom when `children` is empty, a list otherwise.
 * It is enough for the subset of the vnnlib format we support (see `load_vnnlib`). */
struct SExpr {
  std::string atom;
  std::vector<SExpr> children;
  bool is_atom() const { return children.empty(); }
};

/** Split a vnnlib file into parentheses and atoms, dropping the `;` comments. */
static std::vector<std::string> vnnlib_tokenize(std::istream& in) {
  std::vector<std::string> tokens;
  std::string line;
  while(std::getline(in, line)) {
    size_t comment = line.find(';');
    if(comment != std::string::npos) { line.resize(comment); }
    std::string atom;
    for(char c : line) {
      if(c == '(' || c == ')') {
        if(!atom.empty()) { tokens.push_back(atom); atom.clear(); }
        tokens.push_back(std::string(1, c));
      }
      else if(std::isspace(static_cast<unsigned char>(c))) {
        if(!atom.empty()) { tokens.push_back(atom); atom.clear(); }
      }
      else { atom.push_back(c); }
    }
    if(!atom.empty()) { tokens.push_back(atom); }
  }
  return tokens;
}

/** Read one s-expression starting at `i`, which is advanced past it. */
static bool vnnlib_parse(const std::vector<std::string>& tokens, size_t& i, SExpr& out) {
  if(i >= tokens.size() || tokens[i] == ")") { return false; }
  if(tokens[i] != "(") { out.atom = tokens[i++]; return true; }
  ++i;  // consume '('
  while(i < tokens.size() && tokens[i] != ")") {
    SExpr child;
    if(!vnnlib_parse(tokens, i, child)) { return false; }
    out.children.push_back(std::move(child));
  }
  if(i >= tokens.size()) { return false; }  // missing ')'
  ++i;  // consume ')'
  /** An empty list would be indistinguishable from an atom, and we never expect one. */
  return !out.children.empty();
}

/** Convert a decimal literal into the tightest pair of doubles enclosing it, by reading it twice
 * with the rounding mode set outward. This is the same technique as `impl::string_to_real` of
 * `lala-parsing/flatzinc_parser.hpp`, and it is necessary because a decimal such as
 * `-0.303531156` is not representable: rounding it to the nearest double could move a bound inward
 * and cut off a part of the property. The weights and the biases need no such care, they are
 * already stored as floats in the onnx file. */
static void vnnlib_string_to_real(const std::string& str, double& lb, double& ub) {
  #if !defined(__GNUC__) && !defined(_MSC_VER)
    #pragma STDC FENV_ACCESS ON
  #endif
  int old_round = std::fegetround();
  std::fesetround(FE_DOWNWARD);
  lb = std::strtod(str.c_str(), nullptr);
  std::fesetround(FE_UPWARD);
  ub = std::strtod(str.c_str(), nullptr);
  std::fesetround(old_round);
}

/** A term of an assertion: either a neuron of the store, or a numeric literal.
 * A literal is kept as the enclosure \f$ [lb, ub] \f$ of its exact decimal value, so that each
 * bound can later be rounded in the direction that widens the property. */
struct VnnTerm {
  bool is_var;
  int var;    /**< index of the neuron in the store, when `is_var`. */
  double lb;  /**< the literal rounded downward, otherwise. */
  double ub;  /**< the literal rounded upward, otherwise. */
};

/** `X_i` is the input neuron `i` and `Y_j` the output neuron `j`, which lives at
 * `net.output_base() + j` in the store since all the neurons share a single numbering. */
static bool vnnlib_term(const SExpr& e, const Network& net, VnnTerm& out) {
  if(e.is_atom()) {
    const std::string& a = e.atom;
    if(a.size() > 2 && (a[0] == 'X' || a[0] == 'Y') && a[1] == '_') {
      char* endp = nullptr;
      long idx = std::strtol(a.c_str() + 2, &endp, 10);
      if(endp == a.c_str() + 2 || *endp != '\0' || idx < 0) {
        std::cerr << "ERROR: `" << a << "` is not a valid neuron name.\n";
        return false;
      }
      const int limit = (a[0] == 'X') ? net.num_inputs() : net.num_outputs();
      if(idx >= limit) {
        std::cerr << "ERROR: `" << a << "` is out of range, the network has " << limit
                  << (a[0] == 'X' ? " input" : " output") << " neurons.\n";
        return false;
      }
      const int base = (a[0] == 'X') ? 0 : net.output_base();
      out = VnnTerm{true, base + static_cast<int>(idx), 0.0, 0.0};
      return true;
    }
    char* endp = nullptr;
    std::strtod(a.c_str(), &endp);  /**< only to validate that the whole atom is a number. */
    if(endp == a.c_str() || *endp != '\0') {
      std::cerr << "ERROR: `" << a << "` is neither a neuron nor a number.\n";
      return false;
    }
    double lb, ub;
    vnnlib_string_to_real(a, lb, ub);
    out = VnnTerm{false, -1, lb, ub};
    return true;
  }
  /** Some vnnlib files write the negative literals as `(- 1.5)`. Negation is exact in floating
   * point, but it swaps the two ends of the enclosure. */
  if(e.children.size() == 2 && e.children[0].is_atom() && e.children[0].atom == "-") {
    VnnTerm t;
    if(!vnnlib_term(e.children[1], net, t) || t.is_var) { return false; }
    out = VnnTerm{false, -1, -t.ub, -t.lb};
    return true;
  }
  return false;
}

/** One atom of the property, once interpreted.
 *  * unary:  `neurons[x] <= k` when `upper`, `neurons[x] >= k` otherwise. It is embedded directly
 *    in the store of neurons at the root, and costs no propagator.
 *  * binary: `neurons[x] - neurons[y] <= k`, which becomes a `NeuronLeq` propagated by
 *    `FastNNRelu::deduce_property`.
 */
struct PropAtom {
  bool is_binary;
  int x;
  int y;
  /** The enclosure of the constant. `apply_disjunct` picks the end that widens the property:
   * `ub` for an upper bound, `lb` for a lower bound. */
  double k_lb;
  double k_ub;
  bool upper;
};

/** A property in disjunctive normal form: a disjunction of conjunctions of atoms.
 * `acasxu_2023` needs it, both for its output constraints
 * (`(assert (or (and (<= Y_1 Y_0)) (and (<= Y_2 Y_0)) ...))`) and, for `prop_6` and `prop_8`, for
 * its input box, which is a union of two boxes. Each disjunct is a box plus a conjunction of
 * comparisons, which is exactly what one run of the solver handles, so we solve the disjuncts one
 * after the other. */
using Conjunction = std::vector<PropAtom>;
using DNF = std::vector<Conjunction>;

/** Guard against the combinatorial explosion of the DNF expansion. The benchmarks we target stay
 * far below it (`acasxu`'s `prop_6` is the largest with 2 * 4 = 8 disjuncts). */
#define MAX_PROPERTY_DISJUNCTS 4096

/** `out = a /\ b` in DNF, i.e. the pairwise union of their conjunctions. */
static bool dnf_and(const DNF& a, const DNF& b, DNF& out) {
  if(a.size() * b.size() > MAX_PROPERTY_DISJUNCTS) {
    std::cerr << "ERROR: the property has more than " << MAX_PROPERTY_DISJUNCTS
              << " disjuncts once expanded into DNF.\n";
    return false;
  }
  out.clear();
  out.reserve(a.size() * b.size());
  for(const Conjunction& ca : a) {
    for(const Conjunction& cb : b) {
      Conjunction c(ca);
      c.insert(c.end(), cb.begin(), cb.end());
      out.push_back(std::move(c));
    }
  }
  return true;
}

/** Interpret one formula into a DNF over the neurons.
 * An empty DNF means `false` (no disjunct can be satisfied), while a DNF holding one empty
 * conjunction means `true` (no constraint at all).
 */
static bool vnnlib_dnf(const SExpr& e, const Network& net, DNF& out) {
  out.clear();
  if(e.is_atom()) {
    /** `true` and `false` are the only atoms that can stand as a formula by themselves. */
    if(e.atom == "true") { out.push_back(Conjunction{}); return true; }
    if(e.atom == "false") { return true; }  // empty DNF
    std::cerr << "ERROR: unexpected formula `" << e.atom << "`.\n";
    return false;
  }
  if(!e.children[0].is_atom()) {
    std::cerr << "ERROR: the head of a formula must be an operator.\n";
    return false;
  }
  const std::string& op = e.children[0].atom;

  if(op == "and") {
    out.push_back(Conjunction{});  // neutral element
    for(size_t k = 1; k < e.children.size(); ++k) {
      DNF child;
      if(!vnnlib_dnf(e.children[k], net, child)) { return false; }
      DNF merged;
      if(!dnf_and(out, child, merged)) { return false; }
      out = std::move(merged);
    }
    return true;
  }
  if(op == "or") {
    for(size_t k = 1; k < e.children.size(); ++k) {
      DNF child;
      if(!vnnlib_dnf(e.children[k], net, child)) { return false; }
      if(out.size() + child.size() > MAX_PROPERTY_DISJUNCTS) {
        std::cerr << "ERROR: the property has more than " << MAX_PROPERTY_DISJUNCTS << " disjuncts.\n";
        return false;
      }
      for(Conjunction& c : child) { out.push_back(std::move(c)); }
    }
    return true;
  }

  const bool is_le = (op == "<=" || op == "<");
  const bool is_ge = (op == ">=" || op == ">");
  const bool is_eq = (op == "=");
  if(!is_le && !is_ge && !is_eq) {
    std::cerr << "ERROR: unsupported operator `" << op << "` in the vnnlib file.\n";
    return false;
  }
  if(e.children.size() != 3) {
    std::cerr << "ERROR: `" << op << "` expects two arguments.\n";
    return false;
  }
  if(op == "<" || op == ">") {
    printf("%% WARNING: the strict comparison `%s` is relaxed into its non-strict version, which\n\
%% over-approximates the property.\n", op.c_str());
  }

  VnnTerm lhs, rhs;
  if(!vnnlib_term(e.children[1], net, lhs) || !vnnlib_term(e.children[2], net, rhs)) {
    return false;
  }

  Conjunction conj;
  if(lhs.is_var && rhs.is_var) {
    /** `x <= y` is `x - y <= 0`, and `x >= y` is `y - x <= 0`. This is the shape of the output
     * constraints of `acasxu_2023` and `safenlp_2024`. The constant `0` is exact. */
    if(is_le || is_eq) { conj.push_back(PropAtom{true, lhs.var, rhs.var, 0.0, 0.0, true}); }
    if(is_ge || is_eq) { conj.push_back(PropAtom{true, rhs.var, lhs.var, 0.0, 0.0, true}); }
  }
  else if(!lhs.is_var && !rhs.is_var) {
    /** A comparison between two constants is either vacuous or unsatisfiable. We keep it when it
     * *can* hold on the enclosures, so that a rounding never removes a disjunct. */
    bool holds = is_eq ? (lhs.lb <= rhs.ub && rhs.lb <= lhs.ub)
               : (is_le ? (lhs.lb <= rhs.ub) : (lhs.ub >= rhs.lb));
    if(!holds) { return true; }  // empty DNF, i.e. `false`
  }
  else {
    /** We normalize to `var op constant`, flipping the operator when the neuron is on the right. */
    const int var = lhs.is_var ? lhs.var : rhs.var;
    const VnnTerm& c = lhs.is_var ? rhs : lhs;
    const bool upper = lhs.is_var ? is_le : is_ge;
    if(is_eq) {
      conj.push_back(PropAtom{false, var, -1, c.lb, c.ub, true});
      conj.push_back(PropAtom{false, var, -1, c.lb, c.ub, false});
    }
    else {
      conj.push_back(PropAtom{false, var, -1, c.lb, c.ub, upper});
    }
  }
  out.push_back(std::move(conj));
  return true;
}

/** Load the property of the vnnlib file as a DNF over the neurons.
 *
 * `X_i` is the input neuron `i` and `Y_j` the output neuron `net.output_base() + j`. Following the
 * vnnlib convention, the assertions describe the region we are looking for: a point of that region
 * is a counterexample (`sat`), and an empty region is a proof that the property holds (`unsat`).
 * The assertions of a file are conjoined, and the result is expanded into DNF because a single run
 * of the solver explores one box with one conjunction of comparisons.
 *
 * Returns `false` on a malformed or unsupported file, in which case the search must not start.
 */
inline bool load_property(const Configuration<battery::standard_allocator>& config,
  const Network& net, DNF& dnf)
{
  dnf.clear();
  dnf.push_back(Conjunction{});  /**< `true`: a single disjunct with no constraint. */
  if(config.vnnlib_path.size() == 0) {
    printf("%% WARNING: no vnnlib file given (-vnnlib_path), every neuron is left unbounded, hence\n\
%% the input box is unbounded and the property is trivially satisfiable.\n");
    return true;
  }
  std::ifstream in(config.vnnlib_path.data());
  if(!in) {
    std::cerr << "ERROR: cannot open the vnnlib file `" << config.vnnlib_path.data() << "`.\n";
    return false;
  }
  std::vector<std::string> tokens = vnnlib_tokenize(in);
  size_t i = 0;
  int num_assertions = 0;
  while(i < tokens.size()) {
    SExpr e;
    if(!vnnlib_parse(tokens, i, e)) {
      std::cerr << "ERROR: syntax error in the vnnlib file `" << config.vnnlib_path.data() << "`.\n";
      return false;
    }
    /** `declare-const`, `set-logic`, ... carry no constraint, we skip them. */
    if(e.is_atom() || !e.children[0].is_atom() || e.children[0].atom != "assert") { continue; }
    if(e.children.size() != 2) {
      std::cerr << "ERROR: `assert` expects a single formula.\n";
      return false;
    }
    DNF child;
    if(!vnnlib_dnf(e.children[1], net, child)) { return false; }
    DNF merged;
    if(!dnf_and(dnf, child, merged)) { return false; }
    dnf = std::move(merged);
    ++num_assertions;
  }
  if(config.verbose_solving >= 1) {
    printf("%% Loaded %d assertions from `%s`, expanded into %d disjunct(s).\n",
      num_assertions, config.vnnlib_path.data(), (int)dnf.size());
  }
  return true;
}

/** Install one disjunct: its unary atoms become the bounds of the neurons at the root, and its
 * binary atoms become the comparisons propagated by `FastNNRelu::deduce_property`.
 * `neurons` must arrive with every neuron at top, i.e. `]-oo, +oo[`; any neuron left untouched
 * keeps that value.
 */
inline void apply_disjunct(const Conjunction& conj, NNStore& neurons,
  battery::vector<NeuronLeq>& leqs)
{
  using local_itv = typename NNStore::universe_type::local_type;
  using LB2 = typename local_itv::LB;
  using UB2 = typename local_itv::UB;
  using bound_t = typename LB2::value_type;
  leqs.clear();
  for(const PropAtom& a : conj) {
    /** The cast from `double` to `bound_t` (a `float` unless `TURBO_ITV_BITS == 64`) is a second
     * rounding, so it is directed too: `ru_cast` for an upper bound and `rd_cast` for a lower one.
     * Both widen the property, which never cuts off a counterexample. */
    if(a.is_binary) {
      /** `neurons[x] - neurons[y] <= k`: the largest `k` is the weakest constraint. */
      leqs.push_back(NeuronLeq{a.x, a.y, battery::ru_cast<float>(a.k_ub)});
    }
    else if(a.upper) {
      neurons.embed(a.x, local_itv(LB2::top(), UB2(battery::ru_cast<bound_t>(a.k_ub))));
    }
    else {
      neurons.embed(a.x, local_itv(LB2(battery::rd_cast<bound_t>(a.k_lb)), UB2::top()));
    }
  }
}

/** The disjunct `conj` rounded *inward*, for the strict test of the candidate points
 * (`jet_candidates.hpp`): its unary atoms become the bounds of `neurons` (which must arrive at top),
 * `o <= k` with the largest float below `k` (`rd_cast` of the lower end of the enclosure of `k`) and
 * `o >= k` with the smallest float above it (`ru_cast` of the upper end), and its comparisons
 * `x - y <= k` are kept with `k` rounded down. A point inside these bounds satisfies the atoms of the
 * real-valued property, unlike the outward-rounded ones of `apply_disjunct` (which may be a little
 * larger). An atom whose inner bound is empty makes the store bot: no candidate then. */
inline void apply_disjunct_inner(const Conjunction& conj, NNStore& neurons,
  battery::vector<NeuronLeq>& leqs)
{
  using local_itv = typename NNStore::universe_type::local_type;
  using LB2 = typename local_itv::LB;
  using UB2 = typename local_itv::UB;
  using bound_t = typename LB2::value_type;
  leqs.clear();
  for(const PropAtom& a : conj) {
    if(a.is_binary) {
      leqs.push_back(NeuronLeq{a.x, a.y, battery::rd_cast<float>(a.k_lb)});
    }
    else if(a.upper) {
      neurons.embed(a.x, local_itv(LB2::top(), UB2(battery::rd_cast<bound_t>(a.k_lb))));
    }
    else {
      neurons.embed(a.x, local_itv(LB2(battery::ru_cast<bound_t>(a.k_ub)), UB2::top()));
    }
  }
}

#endif // TURBO_IPC_ABSTRACT_DOMAIN
#endif // __CUDACC__

} // namespace jet

#endif // TURBO_JET_PARSING_HPP
