// Copyright 2022 Pierre Talbot

#ifndef TYPE_INFERENCE_HPP
#define TYPE_INFERENCE_HPP

#include "ast.hpp"

using namespace lala;

template <class F>
CUDA bool simple_form(const F& f) {
  return f.is(F::E) ||
    (f.is(F::Seq)
    && (f.seq(0).is(F::LV) || f.seq(0).is(F::V))
    && f.seq(1).is(F::Z));
}

/** A naive type inference algorithm.
    Formula with zero or one variable occurrence are treated in the `sty` abstract domain, others are typed in the `pty` abstract domain.
    For conjunction, if all sub-formulas are in `sty`, it inherits the `sty` type, otherwise it gets the `pty` type. */
template <class F>
CUDA void infer_type(F& f, AType sty, AType pty) {
  if(f.is(F::Seq) && f.sig() == AND) {
    auto& seq = f.seq();
    AType res = sty;
    for(int i = 0; i < seq.size(); ++i) {
      infer_type(seq[i], sty, pty);
      if(seq[i].type() != sty) {
        res = pty;
      }
    }
    f.type_as(res);
  }
  else {
    if(num_vars(f) <= 1 && simple_form(f)) {
      f.type_as(sty);
    }
    else {
      f.type_as(pty);
    }
  }
}

#endif
