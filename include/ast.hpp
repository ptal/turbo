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

#ifndef TURBO_AST_HPP
#define TURBO_AST_HPP

#include <variant>
#include <vector>
#include <string>
#include <boost/spirit/home/x3.hpp>

typedef Variable = std::string

struct Unary {
  v: Variable
};

enum ArithOp {
  ADD, SUB, MUL, DIV, POW
};

enum CmpOp {
  EQ, LEQ, GEQ, NEQ, GT, LT
};

struct Binary {
  Expr left;
  Expr right;
  ArithOp op;
}

typedef Expr =
  std::variant<
    Variable,
    int,
    Unary,
    Binary>

struct Constraint {
  Expr left;
  Expr right;
  CmpOp op;
};

typedef Model = std::vector<Constraint> constraints;

#endif
