// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_SYMBOLIC_MAP_HPP
#define TURBO_JET_SYMBOLIC_MAP_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "lala/logic/ast.hpp"
#include "jet_symbolic.hpp"

/** Using the symbolic bounds of JET (design 2, `jet_symbolic.hpp`) in the ternarized model of
 * `-arch fbarebones` (`-jet_symbolic`). Host code only.
 *
 * `-arch fbarebones` solves the constraints produced by the ONNX and vnnlib parsers of lala-parsing,
 * ternarized into `x = y <op> z` and propagated by PIR: the network is a set of variables and
 * bytecodes, not a `FastNNReluNetwork`. We read the ONNX file a second time with
 * `jet::parse_network`, which gives the layers, weights and biases that `symbolic::pass` needs, and
 * map every neuron `i` of that network onto the variables of the ternarized model:
 *   * `post[i]`: the variable of the neuron (after its ReLU, if any), bounded by `pass`;
 *   * `pre[i]`: the variable of its pre-activation, which the ternarized model keeps as a separate
 *     variable (`post[i] = max(pre[i], 0)`), bounded by `pass` too. `pre[i] == post[i]` without ReLU.
 *
 * The variables are found by name. The ONNX parser of lala-parsing numbers the nodes of the graph
 * (the input layer is `0`, then every node but `Constant` gets `1, 2, ...` in the order of the
 * graph) and names the `j`-th neuron (from `0`) of the node `id`:
 *   * `X_j` for the input layer,
 *   * `Y_j` for the node producing the output of the graph,
 *   * `X_<id>_<j+1>` otherwise.
 * `parse_network` records in a `NetworkOrigin` which node computes the pre-activation and the
 * post-activation of every layer. This needs the variables of the parser to survive the
 * preprocessing, hence `-disable_simplify`.
 *
 * Two readings of the same file can still disagree (a transposed matrix, a normalization that one of
 * them ignores, ...). The symbolic bounds would then be deduced from another network than the one
 * being solved, which is unsound. So `check_network` evaluates the bytecodes as functions over the
 * reals on random points and checks that they compute the network of JET:
 *   * every layer, on its own: `pre = W post(src) + b` (`post(a) + post(b)` for a residual layer),
 *     with random values of both signs given to the inputs and to the neurons of all the other
 *     layers (twice per layer);
 *   * every ReLU: `post = max(pre, 0)`, with random negative, then positive, pre-activations.
 * The property is not read again either: the comparisons `x - y <= 0` used by the input clipping are
 * the top-level `x <= y` bytecodes between two neurons (`comparisons_of_tnf`), which hold in every
 * solution of the model.
 */
namespace jet {
namespace symbolic {

/** The nodes of the ONNX graph computing the layers of a `Network` (see above). */
struct NetworkOrigin {
  /** For every layer, the node computing its pre-activation (the affine or residual part); `0` for
   * the input layer. */
  std::vector<int> pre_node;
  /** For every layer, the node computing its post-activation: its ReLU, or `pre_node` without one. */
  std::vector<int> post_node;
  /** The node producing the output of the graph, whose neurons are named `Y_j`; `-1` if none. */
  int output_node = -1;
  /** Empty if every layer could be traced to the nodes of the graph, otherwise the reason. */
  std::string unsupported;

  void reject(const std::string& why) {
    if(unsupported.empty()) { unsupported = why; }
  }

  /** The name given by the ONNX parser of lala-parsing to the `j`-th neuron of `node`. */
  std::string name(int node, int j) const {
    if(node == 0) { return "X_" + std::to_string(j); }
    if(node == output_node) { return "Y_" + std::to_string(j); }
    return "X_" + std::to_string(node) + "_" + std::to_string(j + 1);
  }
};

/** Find `post` and `pre` (one variable index per neuron of `net`) in the environment `env` of the
 * ternarized model, whose store has the type `aty`. */
template <class Net, class Env, class AType>
bool map_variables(const Net& net, const NetworkOrigin& origin, const Env& env, AType aty,
  std::vector<int>& post, std::vector<int>& pre, std::string& why)
{
  if(!origin.unsupported.empty()) { why = origin.unsupported; return false; }
  if(static_cast<int>(origin.pre_node.size()) != net.num_layers()
   || static_cast<int>(origin.post_node.size()) != net.num_layers())
  {
    why = "the layers of the network could not be traced to the nodes of the ONNX graph";
    return false;
  }
  post.assign(static_cast<size_t>(net.num_neurons), -1);
  pre.assign(static_cast<size_t>(net.num_neurons), -1);
  auto find = [&](const std::string& name, int& vid) {
    auto var = env.variable_of(name.c_str());
    if(!var.has_value()) { return false; }
    auto av = var->get().avar_of(aty);
    if(!av.has_value()) { return false; }
    vid = av->vid();
    return true;
  };
  for(int l = 0; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    if((l > 0 && L.has_relu != 0) != (origin.post_node[l] != origin.pre_node[l])) {
      why = "the ReLU of the layer " + std::to_string(l) + " could not be traced to the ONNX graph";
      return false;
    }
    for(int j = 0; j < L.size; ++j) {
      const int t = L.base + j;
      const std::string post_name = origin.name(origin.post_node[l], j);
      const std::string pre_name = origin.name(origin.pre_node[l], j);
      if(!find(post_name, post[t])) {
        why = "no variable `" + post_name + "` in the model (use -disable_simplify)";
        return false;
      }
      if(!find(pre_name, pre[t])) {
        why = "no variable `" + pre_name + "` in the model (use -disable_simplify)";
        return false;
      }
    }
  }
  /** Two neurons (or a neuron and a pre-activation) on the same variable would make the forms of one
   * of them wrong. */
  std::vector<int> owner(post);
  for(int t = 0; t < net.num_neurons; ++t) {
    if(pre[t] != post[t]) { owner.push_back(pre[t]); }
  }
  std::sort(owner.begin(), owner.end());
  if(std::adjacent_find(owner.begin(), owner.end()) != owner.end()) {
    why = "two neurons are mapped onto the same variable";
    return false;
  }
  return true;
}

/** Evaluate the bytecodes `x = y <op> z` of `iprop` as functions over the reals (in `double`),
 * starting from the variables where `known` is set, until no bytecode gives a new value.
 * Only the operations of the networks are evaluated:
 *   * `ADD` forward, and backward when the other operand is a constant: `x - c` is ternarized as
 *     `x = t + c`, which gives `t = x - c` (a normalization `Sub`);
 *   * `MUL`, `MIN`, `MAX` forward;
 *   * `EQ` with `x = 1`, the equality of two variables (`Flatten`, `Identity`, ...), both ways.
 * A value is never overwritten, and a variable that no evaluation reaches stays unknown.
 * \return the number of passes over the bytecodes, or `-1` if they did not settle in `max_passes`. */
template <class IProp>
int evaluate_tnf(const IProp& iprop, const std::vector<char>& is_constant,
  std::vector<double>& val, std::vector<char>& known, int max_passes = 1000)
{
  const int n = static_cast<int>(iprop.num_deductions());
  for(int pass = 0; pass < max_passes; ++pass) {
    bool changed = false;
    auto set = [&](int v, double d) { val[v] = d; known[v] = 1; changed = true; };
    for(int i = 0; i < n; ++i) {
      const auto bc = iprop.load_deduce(i);
      const int x = bc.x.vid();
      const int y = bc.y.vid();
      const int z = bc.z.vid();
      const bool kx = known[x], ky = known[y], kz = known[z];
      switch(bc.op) {
        case lala::ADD:
          if(!kx && ky && kz) { set(x, val[y] + val[z]); }
          else if(kx && !ky && kz && (is_constant[z] || is_constant[x])) { set(y, val[x] - val[z]); }
          break;
        case lala::MUL:
          if(!kx && ky && kz) { set(x, val[y] * val[z]); }
          break;
        case lala::MIN:
          if(!kx && ky && kz) { set(x, std::min(val[y], val[z])); }
          break;
        case lala::MAX:
          if(!kx && ky && kz) { set(x, std::max(val[y], val[z])); }
          break;
        case lala::EQ:
          if(kx && val[x] == 1.0) {
            if(ky && !kz) { set(z, val[y]); }
            else if(!ky && kz) { set(y, val[z]); }
          }
          break;
        default: break;
      }
    }
    if(!changed) { return pass + 1; }
  }
  return -1;
}

/** Check on random points that the bytecodes of `iprop` compute the network `net` on the variables
 * `post` and `pre` (see the top of this file). `is_constant[v]` tells whether the variable `v` is
 * a constant of the model, of value `value[v]`.
 * \return `false` with the reason in `why` if they do not, or if a neuron cannot be evaluated. */
template <class Net, class IProp>
bool check_network(const Net& net, const IProp& iprop,
  const std::vector<char>& is_constant, const std::vector<double>& value,
  const std::vector<int>& post, const std::vector<int>& pre, std::string& why,
  uint32_t seed = 20260101u)
{
  const int n0 = net.num_inputs();
  const size_t nv = is_constant.size();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> any(-1.0, 1.0);
  std::uniform_real_distribution<double> away(0.125, 1.0);
  std::vector<double> val;
  std::vector<char> known;
  auto reset = [&]() {
    val.assign(nv, 0.0);
    known.assign(nv, 0);
    for(size_t v = 0; v < nv; ++v) {
      if(is_constant[v]) { val[v] = value[v]; known[v] = 1; }
    }
  };
  auto seed_var = [&](int v, double d) { val[v] = d; known[v] = 1; };
  auto evaluate = [&]() {
    if(evaluate_tnf(iprop, is_constant, val, known) < 0) {
      why = "the evaluation of the bytecodes did not settle";
      return false;
    }
    return true;
  };
  auto unknown = [&](int t, const char* what) {
    why = std::string("the ") + what + " of the neuron " + std::to_string(t)
        + " cannot be evaluated from the bytecodes (an operation of the network is not supported)";
    return false;
  };
  auto close = [](double a, double b, double scale) {
    return std::fabs(a - b) <= 1e-9 * scale + 1e-30;
  };

  /** A neuron has a variable for its pre-activation exactly when it has a ReLU. */
  for(int l = 0; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    for(int t = L.base; t < L.base + L.size; ++t) {
      if((l > 0 && L.has_relu != 0) != (post[t] != pre[t])) {
        why = "the neuron " + std::to_string(t) + " has a ReLU in one model only";
        return false;
      }
    }
  }

  /** A. Every affine or residual layer `l`, on its own: the inputs and the neurons of all the other
   * layers get random values of both signs, so that the operands of `l` are independent and any
   * transformation hidden between an operand and `l` (a ReLU, a shift, ...) shows. */
  for(int l = 1; l < net.num_layers(); ++l) {
    const auto& L = net.layers[l];
    for(int round = 0; round < 2; ++round) {
      reset();
      for(int k = 0; k < n0; ++k) { seed_var(post[k], any(rng)); }
      for(int t = n0; t < net.num_neurons; ++t) {
        if(t < L.base || t >= L.base + L.size) { seed_var(post[t], any(rng)); }
      }
      if(!evaluate()) { return false; }
      for(int j = 0; j < L.size; ++j) {
        const int t = L.base + j;
        if(!known[pre[t]]) { return unknown(t, "pre-activation"); }
        double e = 0.0;
        double scale = 0.0;
        if(L.kind == KIND_ADD) {
          const int a = net.layers[L.src0].base + j;
          const int b = net.layers[L.src1].base + j;
          if(!known[post[a]]) { return unknown(a, "value"); }
          if(!known[post[b]]) { return unknown(b, "value"); }
          e = val[post[a]] + val[post[b]];
          scale = std::fabs(val[post[a]]) + std::fabs(val[post[b]]);
        }
        else {
          const auto& S = net.layers[L.src0];
          e = static_cast<double>(net.biases[L.bias_base + j]);
          scale = std::fabs(e);
          for(int c = 0; c < S.size; ++c) {
            const int s = S.base + c;
            if(!known[post[s]]) { return unknown(s, "value"); }
            const double term = static_cast<double>(net.weights[L.weight_base + c * L.size + j]) * val[post[s]];
            e += term;
            scale += std::fabs(term);
          }
        }
        if(!close(val[pre[t]], e, scale)) {
          why = "the neuron " + std::to_string(t) + " (layer " + std::to_string(l)
              + ") is computed differently by the bytecodes (" + std::to_string(val[pre[t]])
              + ") and by the network read from the ONNX file (" + std::to_string(e) + ")";
          return false;
        }
      }
    }
  }

  /** B. The ReLUs, on negative then positive pre-activations. */
  for(int round = 0; round < 2; ++round) {
    reset();
    const double sign = (round == 0) ? -1.0 : 1.0;
    for(int t = n0; t < net.num_neurons; ++t) {
      if(post[t] != pre[t]) { seed_var(pre[t], sign * away(rng)); }
    }
    if(!evaluate()) { return false; }
    for(int t = n0; t < net.num_neurons; ++t) {
      if(post[t] == pre[t]) { continue; }
      if(!known[post[t]]) { return unknown(t, "ReLU"); }
      const double e = std::max(val[pre[t]], 0.0);
      if(!close(val[post[t]], e, std::fabs(e))) {
        why = "the ReLU of the neuron " + std::to_string(t) + " is not `max(pre, 0)` in the bytecodes";
        return false;
      }
    }
  }
  return true;
}

/** Give to the variables equal to a neuron (`1 = (y == z)`, as a `Flatten` or an `Identity` at the
 * end of the graph produces, whose variables are then the `Y_j` of the property) the neuron of the
 * variable they are equal to, in `neuron_of_var`. */
template <class IProp>
void extend_through_equalities(const IProp& iprop, const std::vector<char>& is_constant,
  const std::vector<double>& value, std::vector<int>& neuron_of_var, int max_passes = 64)
{
  const int n = static_cast<int>(iprop.num_deductions());
  for(int pass = 0; pass < max_passes; ++pass) {
    bool changed = false;
    for(int i = 0; i < n; ++i) {
      const auto bc = iprop.load_deduce(i);
      const int x = bc.x.vid();
      if(bc.op != lala::EQ || !is_constant[x] || value[x] != 1.0) { continue; }
      int& ny = neuron_of_var[bc.y.vid()];
      int& nz = neuron_of_var[bc.z.vid()];
      if(ny < 0 && nz >= 0) { ny = nz; changed = true; }
      else if(nz < 0 && ny >= 0) { nz = ny; changed = true; }
    }
    if(!changed) { return; }
  }
}

/** A comparison `x - y <= 0` between two neurons (indices of the network). */
struct NeuronComparison {
  int x;
  int y;
};

/** The comparisons between two neurons that hold in every solution of the model: the bytecodes
 * `c = (y <= z)` whose `c` is a constant, `y <= z` for `c = 1` and `z <= y` for `c = 0` (the
 * negation `y > z` is relaxed into `z <= y`), and `1 = (y == z)` (both ways).
 * `neuron_of_var[v]` is the neuron whose `post` is `v`, or `-1`. */
template <class IProp>
std::vector<NeuronComparison> comparisons_of_tnf(const IProp& iprop,
  const std::vector<char>& is_constant, const std::vector<double>& value,
  const std::vector<int>& neuron_of_var)
{
  std::vector<NeuronComparison> out;
  const int n = static_cast<int>(iprop.num_deductions());
  for(int i = 0; i < n; ++i) {
    const auto bc = iprop.load_deduce(i);
    const int x = bc.x.vid();
    if(!is_constant[x]) { continue; }
    const int ny = neuron_of_var[bc.y.vid()];
    const int nz = neuron_of_var[bc.z.vid()];
    if(ny < 0 || nz < 0 || ny == nz) { continue; }
    if(bc.op == lala::LEQ) {
      if(value[x] == 1.0) { out.push_back(NeuronComparison{ny, nz}); }
      else if(value[x] == 0.0) { out.push_back(NeuronComparison{nz, ny}); }
    }
    else if(bc.op == lala::EQ && value[x] == 1.0) {
      out.push_back(NeuronComparison{ny, nz});
      out.push_back(NeuronComparison{nz, ny});
    }
  }
  return out;
}

} // namespace symbolic
} // namespace jet

#endif
