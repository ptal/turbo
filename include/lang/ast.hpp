// Copyright 2021 Pierre Talbot, Frédéric Pinel

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AST_HPP
#define AST_HPP

#include "cuda_helper.hpp"

// Each abstract domain is uniquely identified by an UID.
typedef int AD_UID;
// This value means a formula is not typed in a particular abstract domain and its type should be inferred.
#define UNTYPED_AD (-1)

// A "logical variable" is just the name of the variable.
typedef char* LVar;

// We call an "abstract variable" the representation of this variable in an abstract domain.
// It is simply an integer containing the UID of the abstract element and an internal integer variable identifier proper to the abstract domain.
// The mapping between logical variables and abstract variables is maintained in `Environment` below.
// An abstract variable always has a single name (or no name if it is not explicitly represented in the initial formula).
// However, a logical variable can be represented by several abstract variables when the variable occurs in different domains.
typedef int AVar;
#define AD_UID(v) (v & 0xFF)
#define VAR_ID(v) (v >> 4)

// We represent everything at the same level (terms, formula, predicate, variable, constant).
// This is general convenient when modelling to avoid creating intermediate boolean variables when reifying.
// We can have `x + (x > y \/ y > x + 4)` and this expression is true if the value is != 0.
struct Formula {
  AD_UID ad_uid;

  enum {
    INT, FLOAT,                    // Constant in the domain of discourse that can be represented exactly.
    AVAR,                          // Abstract variable
    ADD, SUB, MUL, DIV, MOD, POW,  // Terms
    EQ, LEQ, GEQ, NEQ, GT, LT,     // Predicate
    AND, OR, IMPLY, EQUIV, NOT,    // Formula
    RAW                            // General tag for extension purposes.
  } tag;

  // The name of the variable, term, function or predicate is represented by a string.
  // This struct can also be used for representing constant such as real numbers.
  // NOTE: We sometimes cannot use a float type because it cannot represent all real numbers, it is up to the abstract domain to under- or over-approximate it, or choose an exact representation such as rational.
  struct Raw {
    char* name;
    Formula* children;
    size_t n;
  };

  union {
    int i;    // INT
    float f;  // FLOAT
    AVar v;   // AVAR
    struct {  // ADD, SUB, ..., EQ, ..., AND, .., NOT
      Formula* children;
      size_t n;
    };
    Raw raw;  // LVar, global constraints, predicates, real numbers, ...
  };
};

struct SolveMode {
  enum {
    MINIMIZE,
    MAXIMIZE,
    SATISFY
  } tag;

  union {
    LVar lv;  // The logical variable to optimize.
    AVar av;  // The abstract variable to optimize. (We use this one, whenever the variable has been added to an abstract element).
    int num_sols; // How many solutions should we compute (SATISFY mode).
  };
};

// An environment is a formula with an optimization mode and the mapping between logical variables and abstract variables.
struct Environment {
  SolveMode mode;
  Formula formula;
  // Given an abstract variable `v`, `avar2lvar[AD_UID(v)][VAR_ID(v)]` is the name of the variable.
  struct VarArray {
    LVar* data;
    size_t n;
  };
  VarArray* avar2lvar;
  size_t n;
};

#endif
