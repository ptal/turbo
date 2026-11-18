// Copyright 2026 Yi-Nung Tsao

#ifndef TURBO_JET_CONV_HPP
#define TURBO_JET_CONV_HPP

#include <cstdint>
#include <vector>

/** The lowering of a `Conv` of the ONNX graph into a sparse affine layer of `-arch jet`
 * (`LAYER_SPARSE`, `symbolic::KIND_SPARSE`), used by the parsing (`jet_parsing.hpp`) and the host tests.
 *
 * A 2D convolution of an input `[Cin, H, W]` (the layout NCHW of ONNX, without the batch) into an output
 * `[Cout, Ho, Wo]` is linear: the output neuron `(o, y, x)` (index `(o * Ho + y) * Wo + x`, the order of a
 * `Flatten` of the output) is \f$ b_o + \sum w_{o,i,u,v} \cdot in_{g \cdot Cin_g + i,\ y s_h - p_t + u d_h,\ x s_w - p_l + v d_w} \f$
 * over the input channels `i` of its group `g` and the kernel positions `(u, v)` that fall inside the
 * input (the padding is zero). The lowering writes one row per output neuron with these terms; a zero
 * weight is not stored. The terms of a row are written in increasing order of their sources (channel,
 * then line, then column, the kernel positions increasing with them), which is the order of a dense
 * layer (`KIND_SPARSE`).
 */
namespace jet {
namespace conv {

/** The parameters of a 2D convolution (the attributes of the ONNX `Conv`, `auto_pad` resolved). */
struct Conv2d {
  int cin = 1, h = 1, w = 1;    /**< the input `[cin, h, w]`. */
  int cout = 1, kh = 1, kw = 1; /**< the kernel `[cout, cin / group, kh, kw]`. */
  int sh = 1, sw = 1;           /**< `strides`. */
  int pt = 0, pl = 0, pb = 0, pr = 0;  /**< `pads` `[top, left, bottom, right]`. */
  int dh = 1, dw = 1;           /**< `dilations`. */
  int group = 1;
};

/** The output size `[ho, wo]` of the convolution; false if the parameters are invalid or the output
 * is empty. */
inline bool output_shape(const Conv2d& c, int& ho, int& wo) {
  if(c.cin <= 0 || c.h <= 0 || c.w <= 0 || c.cout <= 0 || c.kh <= 0 || c.kw <= 0 || c.sh <= 0 || c.sw <= 0
    || c.dh <= 0 || c.dw <= 0 || c.group <= 0 || c.pt < 0 || c.pl < 0 || c.pb < 0 || c.pr < 0
    || c.cin % c.group != 0 || c.cout % c.group != 0)
  {
    return false;
  }
  const int64_t eh = static_cast<int64_t>(c.dh) * (c.kh - 1) + 1;
  const int64_t ew = static_cast<int64_t>(c.dw) * (c.kw - 1) + 1;
  const int64_t nh = static_cast<int64_t>(c.h) + c.pt + c.pb - eh;
  const int64_t nw = static_cast<int64_t>(c.w) + c.pl + c.pr - ew;
  if(nh < 0 || nw < 0) { return false; }
  ho = static_cast<int>(nh / c.sh + 1);
  wo = static_cast<int>(nw / c.sw + 1);
  return true;
}

/** Appends the rows of the convolution to the CSR `row`, `col`, `val`: `row` receives the offset in
 * `col` of the first entry of each of the `cout * ho * wo` rows, then the end of the last one (so a layer
 * contributes `size + 1` offsets, and the row `j` is `[row[base + j], row[base + j + 1])` for `base` the
 * size of `row` before the call); `col` the index of the source within the input (`(i * h + y) * w + x`),
 * `val` the weight. `weights` is the kernel in the ONNX layout `[cout][cin / group][kh][kw]`.
 * \return the number of entries appended. */
template <class Ints, class Floats, class Weights>
int64_t lower(const Conv2d& c, const Weights& weights, Ints& row, Ints& col, Floats& val) {
  int ho = 0, wo = 0;
  if(!output_shape(c, ho, wo)) { return -1; }
  const int cin_g = c.cin / c.group;
  const int cout_g = c.cout / c.group;
  const int64_t start = static_cast<int64_t>(col.size());
  for(int o = 0; o < c.cout; ++o) {
    const int g = o / cout_g;
    for(int y = 0; y < ho; ++y) {
      for(int x = 0; x < wo; ++x) {
        row.push_back(static_cast<int>(col.size()));
        for(int i = 0; i < cin_g; ++i) {
          const int ci = g * cin_g + i;
          for(int u = 0; u < c.kh; ++u) {
            const int iy = y * c.sh - c.pt + u * c.dh;
            if(iy < 0 || iy >= c.h) { continue; }
            for(int v = 0; v < c.kw; ++v) {
              const int ix = x * c.sw - c.pl + v * c.dw;
              if(ix < 0 || ix >= c.w) { continue; }
              const float wv = weights[((static_cast<int64_t>(o) * cin_g + i) * c.kh + u) * c.kw + v];
              if(wv == 0.0f) { continue; }
              col.push_back((ci * c.h + iy) * c.w + ix);
              val.push_back(wv);
            }
          }
        }
      }
    }
  }
  row.push_back(static_cast<int>(col.size()));
  return static_cast<int64_t>(col.size()) - start;
}

/** The shape of the output of an ONNX `Reshape` of an input of shape `in` (with its batch dimension
 * first) to the constant `target` (`0` copies the dimension of the input, `-1` is inferred). False if
 * the target is invalid or does not have as many elements as the input. */
inline bool reshape(const std::vector<int64_t>& in, const std::vector<int64_t>& target, std::vector<int64_t>& out) {
  int64_t total = 1;
  for(int64_t d : in) { total *= d; }
  out.clear();
  int infer = -1;
  int64_t known = 1;
  for(size_t k = 0; k < target.size(); ++k) {
    int64_t d = target[k];
    if(d == 0) {
      if(k >= in.size()) { return false; }
      d = in[k];
    }
    if(d == -1) {
      if(infer >= 0) { return false; }
      infer = static_cast<int>(k);
      out.push_back(1);
      continue;
    }
    if(d <= 0) { return false; }
    out.push_back(d);
    known *= d;
  }
  if(infer >= 0) {
    if(known == 0 || total % known != 0) { return false; }
    out[infer] = total / known;
    known *= out[infer];
  }
  return known == total;
}

} // namespace conv
} // namespace jet

#endif
