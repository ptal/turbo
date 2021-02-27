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

#ifndef MODEL_BUILDER_HPP
#define MODEL_BUILDER_HPP

#include <vector>
#include <map>
#include <tuple>
#include <exception>

#include "XCSP3Tree.h"
#include "XCSP3TreeNode.h"
#include "XCSP3Variable.h"
#include "XCSP3Constants.h"

#include "vstore.cuh"
#include "propagators.cuh"

using namespace XCSP3Core;
using namespace std::placeholders;

class ModelBuilder {
  private:
    std::map<std::string, std::tuple<Var, Interval>> var2idx;
    std::vector<std::string> idx2var;
    Constraints constraints;
    Var minimize_obj;

  public:
    ModelBuilder() {
      // Negative variable's indices are used for negative view of variable, e.g., `-x`.
      // However the index `0` can't be negated, so we occupy this position with a dumb value.
      add_var("zero_var(fake)", 0, 0);
    }

    VStore* build_store() {
      alignas(VStore) unsigned char *v;
      malloc2_managed(v, sizeof(VStore));
      VStore* vstore = new(v) VStore(idx2var.size());
      for (const auto& x : var2idx) {
        vstore->dom(std::get<0>(x.second), std::get<1>(x.second));
      }
      vstore->init_names(idx2var);
      return vstore;
    }

    Constraints build_constraints() {
      constraints.init_uids();
      return constraints;
    }

    Var build_minimize_obj() {
      return minimize_obj;
    }

    void add_var(std::string name, int min, int max) {
      Var idx = idx2var.size();
      idx2var.push_back(name);
      var2idx[name] = std::make_tuple(idx, Interval(min,max));
    }

    // x <op> k
    void domain_constraint(XVariable *x, OrderType op, int k) {
      strengthen_domain(x->id, op, k);
    }

    void push(Propagator* p) {
      if(p != nullptr) {
        constraints.propagators.push_back(p);
      }
    }

    // x + k <op> y
    Propagator* temporal_constraint(XVariable *x, int k, OrderType op, XVariable *y) {
      return temporal_constraint(x->id, k, op, y->id);
    }

    Propagator* intensional_constraint(Node *node) {
      Propagator* p = nullptr;
      if(!tautology(node)) {
        // Linear inequalities are treated by the global constraint sum.
        if (is_sum_constraint(node)) {
          return sum_constraint(node);
        }
        else {
          if (node->parameters.size() != 2) {
            error(node, "Expected binary primitive constraints.");
          }
          else if(isRelationalOperator(node->type)) {
            move_expr_to_lhs(node);
            Node* lhs = node->parameters[0];
            switch (lhs->type) {
              case OADD: p = add_sub_expr_constraint(node, std::bind(&ModelBuilder::add_lhs, this, _1, _2, _3)); break;
              case OSUB: p = add_sub_expr_constraint(node, std::bind(&ModelBuilder::sub_lhs, this, _1, _2, _3)); break;
              case OMUL: p = mul_expr_constraint(node); break;
              case ODIV: p = div_expr_constraint(node); break;
              case OMOD: p = mod_expr_constraint(node); break;
              default:
                error(node, "This arithmetic operator is not supported.");
            }
          }
          // b <=> C
          else if(node->type == OIFF) {
            p = reified_constraint(node);
          }
          else {
            error(node, "primitive constraint");
          }
        }
      }
      return p;
    }

    Propagator* reified_constraint(Node* node) {
      if (node->parameters[0]->type == OVAR &&
          node->parameters[1]->type == OAND)
      {
        std::string b = node->parameters[0]->toString();
        NodeAnd* and_node = dynamic_cast<NodeAnd*>(node->parameters[1]);
        assert(and_node->parameters.size() == 2);
        Propagator* p1 = intensional_constraint(and_node->parameters[0]);
        Propagator* p2 = intensional_constraint(and_node->parameters[1]);
        if(p1 == nullptr || p2 == nullptr) {
          error(node, "non-reifiable constraints");
        }
        Propagator* rhs = new LogicalAnd(p1, p2);
        return new ReifiedProp(std::get<0>(var2idx[b]), rhs);
      }
      else if (node->parameters[0]->type == OAND &&
               node->parameters[1]->type == OVAR)
      {
        std::swap(node->parameters[0], node->parameters[1]);
        return reified_constraint(node);
      }
      else {
        error(node, "Expected reified constraint of the form  b <=> (c1 /\\ c2)");
        return nullptr;
      }
    }

    bool is_sum_constraint(Node* node) {
      bool res =
        node->parameters[0]->type == OADD &&
        node->parameters[1]->type == ODECIMAL &&
        isRelationalOperator(node->type);
      for (Node* n : node->parameters[0]->parameters) {
        if (n->type != OMUL || n->parameters[0]->type != OVAR || n->parameters[1]->type != ODECIMAL) {
          res = false;
        }
      }
      return res;
    }

    // Must be guarded with `is_sum_constraint`.
    Propagator* sum_constraint(Node* node) {
      int c = dynamic_cast<NodeConstant*>(node->parameters[1])->val;
      std::vector<Var> vars;
      std::vector<int> constants;
      for (Node* n : node->parameters[0]->parameters) {
        vars.push_back(std::get<0>(var2idx[n->parameters[0]->toString()]));
        constants.push_back(dynamic_cast<NodeConstant*>(n->parameters[1])->val);
      }
      return new LinearIneq(vars, constants, c);
    }

    void add_objective_minimize(XVariable *x) {
      minimize_obj = std::get<0>(var2idx[x->id]);
    }

  private:
    void error(Node* node, std::string msg) {
      std::cout << node->toString() << std::endl;
      throw std::runtime_error("unsupported: " + msg);
    }

    Propagator* temporal_constraint(std::string x, int k, OrderType op, std::string y) {
      Var xi = std::get<0>(var2idx[x]);
      Var yi = std::get<0>(var2idx[y]);
      assert(xi != 0 && yi != 0);

      auto p = std::make_pair(xi,-yi);
      return le_canonical_form<std::pair<int,int>>(p, op, -k,
        [](std::pair<int, int>& left, int k) -> Propagator* { return new TemporalProp(left.first, left.second, k); },
        [](std::pair<int, int>& left) {left.first = -left.first; left.second = abs(left.second); });
    }

    // Let a constraint of the form `left <op> k` where `k` is a constant.
    // Create one or more propagators in canonical form `left <= k`.
    template <typename T>
    Propagator* le_canonical_form(T& left, OrderType op, int k,
     std::function<Propagator* (T&, int)> make,
     std::function<void (T&)> neg) {
      // Turn > and < into <= and >=
      if (op == LT) {
        op = LE;
        k = k - 1;
      }
      else if (op == GT) {
        op = GE;
        k = k + 1;
      }

      if (op == GE) {
        neg(left);
        k = -k;
        op = LE;
      }
      else if (op == EQ) {
        Propagator* p1 = le_canonical_form(left, LE, k, make, neg);
        Propagator* p2 = le_canonical_form(left, GE, k, make, neg);
        return new LogicalAnd(p1, p2);
      }
      else if(op == NE) {
        Propagator* p1 = le_canonical_form(left, LT, k, make, neg);
        Propagator* p2 = le_canonical_form(left, GT, k, make, neg);
        return new LogicalOr(p1, p2);
      }
      else if (op == IN) {
        throw std::runtime_error("Operators IN and NE are not supported in unary constraints.");
      }
      return make(left, k);
    }

    int val(Node* node) {
      return dynamic_cast<NodeConstant*>(node)->val;
    }

    // Treat X or Y in X + Y <= K.
    void add_lhs(Node* x, int& k, bool left) {
      if(x->type == ODECIMAL) {
        k -= val(x);
      }
    }

    // Treat X or Y in X - Y <= K. It is X if left == true, Y otherwise.
    void sub_lhs(Node* x, int& k, bool left) {
      if(x->type == ODECIMAL) {
        if(left) {
          k -= val(x);
        } else {
          k += val(x);
        }
      }
    }

    // Precondition: constraint of the form `x + y <op> z` or `x - y <op> z` where x,y,z can be variables or integers.
    Propagator* add_sub_expr_constraint(Node* node, std::function<void(Node* node,int&,bool)> treat_lhs) {
      int xi = 0;
      int yi = 0;
      int k = 0;
      OrderType op = to_order_type(node->type);
      Node* lhs = node->parameters[0];
      Node* x = lhs->parameters[0];
      Node* y = lhs->parameters[1];
      Node* z = node->parameters[1];
      // Base situation
      if(x->type == OVAR) {
        xi = std::get<0>(var2idx[x->toString()]);
      }
      if(y->type == OVAR) {
        yi = std::get<0>(var2idx[y->toString()]);
      }
      if(z->type == ODECIMAL) {
        k = val(z);
      }
      // If X is an integer, move it to the right.
      treat_lhs(x, k, true);
      // If Y is an integer, move it to the right.
      treat_lhs(y, k, false);
      // If Z is a variable, move it to the left, either in `X` or `Y` (at least one slot must be free).
      if(z->type == OVAR && xi == 0) {
        xi = -std::get<0>(var2idx[z->toString()]);
      }
      else if(z->type == OVAR && yi == 0) {
        yi = -std::get<0>(var2idx[z->toString()]);
      }
      else { assert(false); }
      // Case where x <op> k
      auto inv = [](OrderType o) {
        switch(o) {
          case LE: return GE;
          case LT: return GT;
          case GE: return LE;
          case GT: return LT;
          default: return o;
        }
      };
      if(xi == 0 && yi != 0) {
        if(yi < 0) { yi = -yi; k = -k; op = inv(op); }
        strengthen_domain(idx2var[yi], op, k);
      }
      else if(yi == 0 && xi != 0) {
        if(xi < 0) { xi = -xi; k = -k; op = inv(op); }
        strengthen_domain(idx2var[xi], op, k);
      }
      else if(xi == 0 && yi == 0) {
        tautology(0, op, k);
      }
      else {
        auto p = std::make_pair(xi,yi);
        return le_canonical_form<std::pair<int,int>>(p, op, k,
          [](std::pair<int, int>& left, int k) -> Propagator* { return new TemporalProp(left.first, left.second, k); },
          [](std::pair<int, int>& left) {left.first = -left.first; left.second = abs(left.second); });
      }
      // No propagator was created (the constraint was handled directly).
      return nullptr;
    }

    // Precondition: constraint of the form `x * y <op> z` where x,y,z can be variables or integers.
    Propagator* mul_expr_constraint(Node* node) {
      int xi = 0;
      int yi = 0;
      int k = 0;
      OrderType op = to_order_type(node->type);
      Node* lhs = node->parameters[0];
      Node* x = lhs->parameters[0];
      Node* y = lhs->parameters[1];
      Node* z = node->parameters[1];
      // Base situation
      if(x->type == OVAR) {
        xi = std::get<0>(var2idx[x->toString()]);
      }
      if(y->type == OVAR) {
        yi = std::get<0>(var2idx[y->toString()]);
      }
      if(z->type == ODECIMAL) {
        k = val(z);
      }
      error(node, "mul not implemented");
      return nullptr;
    }

    Propagator* div_expr_constraint(Node* node) {
      error(node, "mul not implemented");
      return nullptr;
    }

    Propagator* mod_expr_constraint(Node* node) {
      error(node, "mul not implemented");
      return nullptr;
    }

    // Make sure a primitive constraint is of the form `expr <op> x` with `x` an integer or a variable.
    void move_expr_to_lhs(Node* node) {
      int ty1 = node->parameters[0]->type;
      int ty2 = node->parameters[1]->type;
      if(ty1 == OVAR || ty1 == ODECIMAL && ty2 != ODECIMAL) {
        std::swap(node->parameters[0], node->parameters[1]);
      }
      else if(ty2 != OVAR && ty2 != ODECIMAL) {
        error(node, "expected one side of intensional constraint to be either a constant or a variable.");
      }
    }

    // Transform X * 1 into X.
    void evaluate_constant(Node** node_ptr) {
      Node* node = *node_ptr;
      if(node->type == OMUL) {
        if(node->parameters[0]->type == OVAR && node->parameters[1]->type == ODECIMAL) {
          if(((NodeConstant*)node->parameters[1])->val == 1) {
            *node_ptr = node->parameters[0];
          }
        }
      }
    }

    // Treat constraint of the form X * a <= b.
    bool le_mul_domain(Node* node) {
      if (node->type == OLE &&
          node->parameters[0]->type == OMUL &&
          node->parameters[1]->type == ODECIMAL &&
          node->parameters[0]->parameters[0]->type == OVAR &&
          node->parameters[0]->parameters[1]->type == ODECIMAL)
      {
        std::string x = node->parameters[0]->parameters[0]->toString();
        int a = ((NodeConstant*)node->parameters[0]->parameters[1])->val;
        int b = ((NodeConstant*)node->parameters[1])->val;
        if(a == 0) {
          if(b < 0) {
            add_var("fake var (contradiction detected at root node)", 1, 0);
          }
        }
        else if (b == 0) {
          if(a > 0) {
            strengthen_domain(x, LE, 0);
          }
          else if (a < 0) {
            strengthen_domain(x, GE, 0);
          }
        }
        else {
          // At this point, a and b are different from 0.
          int res = b / a;
          if(a > 0 && b > 0) {
            strengthen_domain(x, LE, res);
          }
          else if(a > 0 && b < 0) {
            strengthen_domain(x, LE, res - (-b % a));
          }
          else if(a < 0 && b > 0) {
            strengthen_domain(x, GE, res);
          }
          else {
            strengthen_domain(x, GE, res + (-b % -a));
          }
        }
        return true;
      }
      return false;
    }

  private:
    void strengthen_domain(std::string x, OrderType op, int k) {
      Interval& v = std::get<1>(var2idx[x]);
      if(op == LT) {
        v.ub = k - 1;
      }
      if (op == GT) {
        v.lb = k + 1;
      }
      if(op == LE || op == EQ) {
        v.ub = k;
      }
      if (op == GE || op == EQ) {
        v.lb = k;
      }
      if (op == IN || op == NE) {
        throw std::runtime_error("Operators IN and NE are not supported in unary constraints.");
      }
    }

    OrderType to_order_type(ExpressionType t) {
        switch(t) {
          case OLE: return LE;
          case OLT: return LT;
          case OGE: return GE;
          case OGT: return GT;
          case OEQ: return EQ;
          case ONE: return NE;
          case OIN: return IN;
          default:
            throw std::runtime_error("Unsupported unary domain operator.");
        }
    }

    void unary_constraint(Node* node) {
      bool treated = le_mul_domain(node);
      if(!treated) {
        if (node->parameters[0]->type != OVAR) {
          evaluate_constant(&node->parameters[0]);
          if (node->parameters[0]->type != OVAR) {
            error(node, "expected variable on the lhs (in domain constraint).");
          }
        }
        if (node->parameters[1]->type != ODECIMAL) {
          error(node, "Expected value on the rhs.");
        }
        std::string x = node->parameters[0]->toString();
        int v = dynamic_cast<NodeConstant*>(node->parameters[1])->val;
        OrderType op = to_order_type(node->type);
        strengthen_domain(x, op, v);
      }
    }

    void tautology(int l, OrderType op, int k) {
      bool res;
      switch(op) {
        case LE: res = l <= k; break;
        case LT: res = l < k; break;
        case GE: res = l >= k; break;
        case GT: res = l > k; break;
        case EQ: res = l == k; break;
        case NE: res = l != k; break;
        default: assert(false);
      }
      if(!res) {
        add_var("fake var (contradiction detected at root node)", 1, 0);
      }
    }

    bool tautology(Node* node) {
      if(node->parameters[0]->type == ODECIMAL &&
         node->parameters[1]->type == ODECIMAL) {
        tautology(val(node->parameters[0]), to_order_type(node->type), val(node->parameters[1]));
        return true;
      }
      return false;
    }
};

#endif
