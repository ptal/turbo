// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_NETWORK_HPP
#define TURBO_JET_NETWORK_HPP

/** The data of `-arch jet` that the parsing (`jet_parsing.hpp`) produces and the solver
 * (`jet_dive_and_solve.hpp`) reads: the layers of the network (`LayerInfo`, `FastNNReluNetwork`),
 * the comparisons of the property (`NeuronLeq`, `FastProperty`) and the store of neurons
 * (`NNStore`). Nothing here parses a file or propagates a constraint. */

#include "battery/allocator.hpp"
#include "battery/vector.hpp"
#include "common_solving.hpp"
#include "lala/vstore.hpp"
#include "jet_symbolic.hpp"
#include "jet_sigmoid.hpp"

namespace jet {

#ifdef __CUDACC__
#ifndef TURBO_IPC_ABSTRACT_DOMAIN

namespace bt = ::battery;

/** `ConcurrentAllocator` allocates memory available both on CPU and GPU. For non-Linux systems such as Windows pinned memory must be used (see PR #19). */
#ifdef NO_CONCURRENT_MANAGED_MEMORY
  using ConcurrentAllocator = bt::pinned_allocator;
#else
  using ConcurrentAllocator = bt::managed_allocator;
#endif

using ::FItv;
/** The kind of a layer of the network. */
enum LayerKind {
  LAYER_INPUT  = 0,  /**< the input neurons, deduced by nothing. */
  LAYER_AFFINE = 1,  /**< \f$ W \cdot src0 + b \f$, optionally followed by a ReLU. */
  LAYER_ADD    = 2,  /**< \f$ src0 + src1 \f$ elementwise: the residual connections of `cersyve`. */
  LAYER_SCALE  = 3,  /**< \f$ s_j \cdot src0_j + t_j \f$ elementwise, optionally followed by a ReLU: the
                      * `Mul`/`Add`/`Sub` by a constant, e.g. the normalization of the inputs of
                      * `cora_2024`. `s` is at `weight_base`, `t` at `bias_base`, one value per neuron. */
  LAYER_SPARSE = 4   /**< \f$ W \cdot src0 + b \f$ with only the nonzero weights stored (`symbolic::KIND_SPARSE`):
                      * a `Conv` lowered by the parsing. The rows are at `weight_base` in `sp_row`, the
                      * columns at `csc_base` in `sp_tcol`, the biases at `bias_base`. */
};
static_assert(LAYER_INPUT == symbolic::KIND_INPUT && LAYER_AFFINE == symbolic::KIND_AFFINE
  && LAYER_ADD == symbolic::KIND_ADD && LAYER_SCALE == symbolic::KIND_SCALE
  && LAYER_SPARSE == symbolic::KIND_SPARSE,
  "jet_symbolic.hpp must use the values of LayerKind.");

/** One layer of the network. Layers are numbered in topological order, the layer `0` being the
 * input, and each layer owning the contiguous range of neurons `[base, base + size)`.
 *
 * A layer names its operands explicitly (`src0`, `src1`) instead of implicitly taking the previous
 * layer, which is what makes a residual network representable: in `cersyve` a layer can read a
 * layer far above it (`Gemm(data_0)` appearing after ten other layers) and an `Add` combines two
 * separate branches.
 */
struct LayerInfo {
  int kind;
  int base;         /**< index of the first neuron of the layer. */
  int size;         /**< number of neurons of the layer. */
  int src0;         /**< operand layer; the first one for `LAYER_ADD`. -1 for the input layer. */
  int src1;         /**< second operand for `LAYER_ADD`, -1 otherwise. */
  int weight_base;  /**< offset of the layer's block in `weights`, for `LAYER_AFFINE` and `LAYER_SCALE`. */
  int bias_base;    /**< offset of the layer's block in `biases`, for `LAYER_AFFINE` and `LAYER_SCALE`. */
  int has_relu;     /**< whether a ReLU is applied to the result of the layer. */
  int sigmoid = 0;  /**< whether a sigmoid is applied to the result of the layer (`jet_sigmoid.hpp`);
                     * never together with `has_relu`. */
  int csc_base = 0; /**< `LAYER_SPARSE`: offset of the columns of the layer in `sp_tcol` (set by
                     * `symbolic::build_sparse_columns`). */
};


/** The read-only description of the neural network: its topology, weights and biases.
 * It is immutable during solving, so a single copy in managed memory is shared by every block;
 * only the store of neurons (`FastNNRelu::store`) is duplicated per block.
 *
 * The network is a DAG of layers, not a chain: `acasxu`, `safenlp` and `tllverifybench` happen to
 * be chains, but the `cersyve` models branch off the input several times and merge the branches
 * back with residual `Add` nodes.
 *
 * Layout conventions:
 *  * The neurons are numbered consecutively, layer by layer in topological order: the layer `l`
 *    owns `[layers[l].base, layers[l].base + layers[l].size)`, and the input layer comes first, so
 *    the neurons having a deduction are exactly `[num_inputs(), num_neurons)`.
 *  * `neuron_layer[n]` is the layer owning the neuron `n`, so locating it is O(1).
 *  * `weights` stores the affine layers consecutively and, within a layer, column-major: the weight
 *    of the connection `(c, j)` of the layer `l` sits at `layers[l].weight_base + c * size + j`.
 *  * `biases` likewise stores one block of `size` entries per affine layer, at `bias_base`; the
 *    layers without a bias in the graph get a block of zeros.
 *  * A sparse layer (`LAYER_SPARSE`, a lowered `Conv`) has no block in `weights`: its nonzero weights
 *    are in `sp_row`, `sp_col`, `sp_val` by rows (the row `j` is `[sp_row[weight_base + j],
 *    sp_row[weight_base + j + 1])`, sources in increasing order), and in `sp_tcol`, `sp_trow`, `sp_tval`
 *    by columns (built by the constructor, `symbolic::build_sparse_columns`), for the backward
 *    contractor. Its biases are a block at `bias_base`, as an affine layer.
 */
template <class Alloc>
struct FastNNReluNetwork {
  using allocator_type = Alloc;

  int num_neurons;
  /** The index of the layer producing the output of the graph, whose neurons the `Y_j` of the
   * vnnlib file refer to. It is not necessarily the last layer of the topological order. */
  int output_layer;

  bt::vector<LayerInfo, Alloc> layers;
  bt::vector<int, Alloc> neuron_layer;
  bt::vector<float, Alloc> weights;
  bt::vector<float, Alloc> biases;
  /** The sparse layers, by rows (CSR) and by columns (CSC), see the layout conventions above. */
  bt::vector<int, Alloc> sp_row;
  bt::vector<int, Alloc> sp_col;
  bt::vector<float, Alloc> sp_val;
  bt::vector<int, Alloc> sp_tcol;
  bt::vector<int, Alloc> sp_trow;
  bt::vector<float, Alloc> sp_tval;
  /** The rows of the symbolic forms of `-jet_symbolic`, by layer, and their number
   * (`symbolic::plan_form_rows`): the layers that are no longer read share their rows, and a scale of
   * the inputs is not stored. Computed by the constructor from the layers. */
  bt::vector<int, Alloc> form_row;
  int num_form_rows;

  FastNNReluNetwork(const Alloc& alloc = Alloc{})
   : num_neurons(0), output_layer(0), layers(alloc), neuron_layer(alloc)
   , weights(alloc), biases(alloc)
   , sp_row(alloc), sp_col(alloc), sp_val(alloc), sp_tcol(alloc), sp_trow(alloc), sp_tval(alloc)
   , form_row(alloc), num_form_rows(0)
  {}

  template <class Alloc2>
  FastNNReluNetwork(int num_neurons, int output_layer,
    const bt::vector<LayerInfo, Alloc2>& layers,
    const bt::vector<int, Alloc2>& neuron_layer,
    const bt::vector<float, Alloc2>& weights,
    const bt::vector<float, Alloc2>& biases,
    const bt::vector<int, Alloc2>& sp_row,
    const bt::vector<int, Alloc2>& sp_col,
    const bt::vector<float, Alloc2>& sp_val,
    const Alloc& alloc = Alloc{})
   : num_neurons(num_neurons)
   , output_layer(output_layer)
   , layers(layers, alloc)
   , neuron_layer(neuron_layer, alloc)
   , weights(weights, alloc)
   , biases(biases, alloc)
   , sp_row(sp_row, alloc)
   , sp_col(sp_col, alloc)
   , sp_val(sp_val, alloc)
   , sp_tcol(alloc)
   , sp_trow(alloc)
   , sp_tval(alloc)
   , form_row(alloc)
   , num_form_rows(0)
  {
    symbolic::build_sparse_columns(this->layers, this->sp_row, this->sp_col, this->sp_val,
      this->sp_tcol, this->sp_trow, this->sp_tval);
    std::vector<int> rows;
    num_form_rows = symbolic::plan_form_rows(this->layers, static_cast<int>(this->layers.size()), output_layer, rows);
    for(int r : rows) { form_row.push_back(r); }
  }

  template <class Alloc2>
  FastNNReluNetwork(const FastNNReluNetwork<Alloc2>& other, const Alloc& alloc = Alloc{})
   : FastNNReluNetwork(other.num_neurons, other.output_layer, other.layers, other.neuron_layer,
       other.weights, other.biases, other.sp_row, other.sp_col, other.sp_val, alloc)
  {}

  FastNNReluNetwork(const FastNNReluNetwork&) = default;
  FastNNReluNetwork(FastNNReluNetwork&&) = default;

  /** Whether the layer `l` is followed by a ReLU. */
  CUDA INLINE bool relu_at(int l) const {
    return l >= 0 && l < static_cast<int>(layers.size()) && layers[l].has_relu != 0;
  }

  /** The number of neurons of the input layer, the only ones we branch on. */
  CUDA INLINE int num_inputs() const {
    return layers.size() >= 1 ? layers[0].size : num_neurons;
  }

  /** The index of the first neuron of the output layer, that is the neuron `Y_0`. */
  CUDA INLINE int output_base() const {
    return layers.size() >= 1 ? layers[output_layer].base : 0;
  }

  /** The number of neurons of the output layer. */
  CUDA INLINE int num_outputs() const {
    return layers.size() >= 1 ? layers[output_layer].size : 0;
  }

  /** The number of neurons that have a deduction, that is every neuron but the input ones. */
  CUDA INLINE int num_deductions() const {
    return num_neurons - num_inputs();
  }

  CUDA INLINE int num_layers() const {
    return static_cast<int>(layers.size());
  }

  /** A network we failed to parse, or with no hidden layer, has nothing to propagate. */
  CUDA bool empty() const {
    return num_neurons == 0 || layers.size() < 2;
  }

  void print() const {
    int affine = 0;
    int residual = 0;
    int scale = 0;
    int sparse = 0;
    for(int l = 1; l < num_layers(); ++l) {
      affine += (layers[l].kind == LAYER_AFFINE) ? 1 : 0;
      residual += (layers[l].kind == LAYER_ADD) ? 1 : 0;
      scale += (layers[l].kind == LAYER_SCALE) ? 1 : 0;
      sparse += (layers[l].kind == LAYER_SPARSE) ? 1 : 0;
    }
    printf("%% In total, we have %d neurons in the network (%d layers: %d affine, %d sparse (conv), %d residual, %d scale; %d inputs, %d outputs, %d deductions, %d sparse weights)\n",
      num_neurons, num_layers(), affine, sparse, residual, scale, num_inputs(), num_outputs(), num_deductions(), static_cast<int>(sp_val.size()));
    printf("%% Layers (A = affine, C = sparse/conv, + = residual add, S = elementwise scale, R = with ReLU, Sg = with sigmoid): ");
    for(int l = 1; l < num_layers(); ++l) {
      printf("%c%s", layers[l].kind == LAYER_ADD ? '+' : (layers[l].kind == LAYER_SCALE ? 'S' : (layers[l].kind == LAYER_SPARSE ? 'C' : 'A')),
        relu_at(l) ? "R" : (layers[l].sigmoid != 0 ? "Sg" : ""));
      if(l + 1 < num_layers()) { printf(" "); }
    }
    printf("\n");
    /** A chain reads each layer from the one just above it; anything else is a DAG and worth
     * showing, since it is where the residual connections are. */
    bool is_chain = true;
    for(int l = 1; l < num_layers(); ++l) {
      if((layers[l].kind != LAYER_AFFINE && layers[l].kind != LAYER_SPARSE) || layers[l].src0 != l - 1) { is_chain = false; }
    }
    if(!is_chain) {
      printf("%% Topology: ");
      for(int l = 1; l < num_layers(); ++l) {
        if(layers[l].kind == LAYER_ADD) { printf("%d=(%d + %d) ", l, layers[l].src0, layers[l].src1); }
        else { printf("%d<-%d ", l, layers[l].src0); }
      }
      printf("\n");
    }
  }
};

/** A binary constraint of the property: \f$ neurons[x] - neurons[y] \leq k \f$.
 * The output constraints of `acasxu_2023` and `safenlp_2024` are comparisons between two output
 * neurons (`(assert (<= Y_0 Y_1))`), which is `x = Y_0, y = Y_1, k = 0`. A unary bound never
 * reaches here: it is embedded directly in the store of neurons at the root.
 */
struct NeuronLeq {
  int x;
  int y;
  float k;
};

/** The non-unary part of the property, shared read-only by all the blocks like the network.
 * It holds one conjunction of comparisons: a disjunctive property is expanded into DNF on the host
 * and each disjunct is solved as a separate run (see `load_property`).
 */
template <class Alloc>
struct FastProperty {
  using allocator_type = Alloc;

  bt::vector<NeuronLeq, Alloc> leqs;

  FastProperty(const Alloc& alloc = Alloc{}): leqs(alloc) {}

  template <class Alloc2>
  FastProperty(const bt::vector<NeuronLeq, Alloc2>& leqs, const Alloc& alloc = Alloc{})
   : leqs(leqs, alloc) {}

  template <class Alloc2>
  FastProperty(const FastProperty<Alloc2>& other, const Alloc& alloc = Alloc{})
   : leqs(other.leqs, alloc) {}

  FastProperty(const FastProperty&) = default;
  FastProperty(FastProperty&&) = default;
  FastProperty& operator=(const FastProperty&) = default;
  FastProperty& operator=(FastProperty&&) = default;

  CUDA INLINE int size() const { return static_cast<int>(leqs.size()); }
};

using NNStore = VStore<FItv, ConcurrentAllocator>;
using Network = FastNNReluNetwork<ConcurrentAllocator>;
using Property = FastProperty<ConcurrentAllocator>;

#endif // TURBO_IPC_ABSTRACT_DOMAIN
#endif // __CUDACC__

} // namespace jet

#endif // TURBO_JET_NETWORK_HPP
