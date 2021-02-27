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

#ifndef XCSP3_TURBO_CALLBACKS_HPP
#define XCSP3_TURBO_CALLBACKS_HPP

#include <iostream>

#include "XCSP3Tree.h"
#include "XCSP3CoreCallbacks.h"
#include "XCSP3Variable.h"

#include "model_builder.hpp"

/**
 * This is an example that prints useful informations of a XCSP3 instance.
 * You need to create your own class and to override functions of the callback.
 * We suggest to make a map between XVariable and your own variables in order to
 * facilitate the constructions of constraints.
 *
 * see main.cc to show declaration of the parser
 *
 */

namespace XCSP3Core {
    using namespace std;

    class XCSP3_turbo_callbacks : public XCSP3CoreCallbacks {
    public:
        using XCSP3CoreCallbacks::buildConstraintMinimum;
        using XCSP3CoreCallbacks::buildConstraintMaximum;
        using XCSP3CoreCallbacks::buildConstraintElement;
        using XCSP3CoreCallbacks::buildObjectiveMinimize;
        using XCSP3CoreCallbacks::buildObjectiveMaximize;

        XCSP3_turbo_callbacks(ModelBuilder* model_builder);

        virtual void beginInstance(InstanceType type) override;

        virtual void endInstance() override;

        virtual void beginVariables() override;

        virtual void endVariables() override;

        virtual void beginVariableArray(string id) override;

        virtual void endVariableArray() override;

        virtual void beginConstraints() override;

        virtual void endConstraints() override;

        virtual void beginGroup(string id) override;

        virtual void endGroup() override;

        virtual void beginBlock(string classes) override;

        virtual void endBlock() override;

        virtual void beginSlide(string id, bool circular) override;

        virtual void endSlide() override;

        virtual void beginObjectives() override;

        virtual void endObjectives() override;

        virtual void beginAnnotations() override;

        virtual void endAnnotations() override;

        virtual void buildVariableInteger(string id, int minValue, int maxValue) override;

        virtual void buildVariableInteger(string id, vector<int> &values) override;

        virtual void buildConstraintExtension(string id, vector<XVariable *> list, vector<vector<int>> &tuples, bool support, bool hasStar) override;

        virtual void buildConstraintExtension(string id, XVariable *variable, vector<int> &tuples, bool support, bool hasStar) override;

        virtual void buildConstraintExtensionAs(string id, vector<XVariable *> list, bool support, bool hasStar) override;

        virtual void buildConstraintIntension(string id, string expr) override;

        virtual void buildConstraintIntension(string id, Tree *tree) override;

        virtual void buildConstraintPrimitive(string id, OrderType op, XVariable *x, int k, XVariable *y) override;

        virtual void buildConstraintPrimitive(string id, OrderType op, XVariable *x, int k) override;

        virtual void buildConstraintPrimitive(string id, XVariable *x,  bool in, int min, int max) override;

        virtual void buildConstraintRegular(string id, vector<XVariable *> &list, string st, vector<string> &final, vector<XTransition> &transitions) override;

        virtual void buildConstraintMDD(string id, vector<XVariable *> &list, vector<XTransition> &transitions) override;

        virtual void buildConstraintAlldifferent(string id, vector<XVariable *> &list) override;

        virtual void buildConstraintAlldifferentExcept(string id, vector<XVariable *> &list, vector<int> &except) override;

        virtual void buildConstraintAlldifferent(string id, vector<Tree *> &list) override;

        virtual void buildConstraintAlldifferentList(string id, vector<vector<XVariable *>> &lists) override;

        virtual void buildConstraintAlldifferentMatrix(string id, vector<vector<XVariable *>> &matrix) override;

        virtual void buildConstraintAllEqual(string id, vector<XVariable *> &list) override;

        virtual void buildConstraintNotAllEqual(string id, vector<XVariable *> &list) override;

        virtual void buildConstraintOrdered(string id, vector<XVariable *> &list, OrderType order) override;

        virtual void buildConstraintOrdered(string id, vector<XVariable *> &list, vector<int> &lengths, OrderType order) override;

        virtual void buildConstraintLex(string id, vector<vector<XVariable *>> &lists, OrderType order) override;

        virtual void buildConstraintLexMatrix(string id, vector<vector<XVariable *>> &matrix, OrderType order) override;

        virtual void buildConstraintSum(string id, vector<XVariable *> &list, vector<int> &coeffs, XCondition &cond) override;

        virtual void buildConstraintSum(string id, vector<XVariable *> &list, XCondition &cond) override;

        virtual void buildConstraintSum(string id, vector<XVariable *> &list, vector<XVariable *> &coeffs, XCondition &cond) override;

        virtual void buildConstraintSum(string id, vector<Tree *> &list, vector<int> &coeffs, XCondition &cond) override;

        virtual void buildConstraintSum(string id, vector<Tree *> &list, XCondition &cond) override;

        virtual void buildConstraintAtMost(string id, vector<XVariable *> &list, int value, int k) override;

        virtual void buildConstraintAtLeast(string id, vector<XVariable *> &list, int value, int k) override;

        virtual void buildConstraintExactlyK(string id, vector<XVariable *> &list, int value, int k) override;

        virtual void buildConstraintAmong(string id, vector<XVariable *> &list, vector<int> &values, int k) override;

        virtual void buildConstraintExactlyVariable(string id, vector<XVariable *> &list, int value, XVariable *x) override;

        virtual void buildConstraintCount(string id, vector<XVariable *> &list, vector<int> &values, XCondition &xc) override;

        virtual void buildConstraintCount(string id, vector<XVariable *> &list, vector<XVariable *> &values, XCondition &xc) override;

        virtual void buildConstraintNValues(string id, vector<XVariable *> &list, vector<int> &except, XCondition &xc) override;

        virtual void buildConstraintNValues(string id, vector<XVariable *> &list, XCondition &xc) override;

        virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<int> values, vector<int> &occurs, bool closed) override;

        virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<int> values, vector<XVariable *> &occurs,
                                                bool closed) override;

        virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<int> values, vector<XInterval> &occurs,
                                                bool closed) override;

        virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<XVariable *> values, vector<int> &occurs,
                                                bool closed) override;

        virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<XVariable *> values, vector<XVariable *> &occurs,
                                                bool closed) override;

        virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<XVariable *> values, vector<XInterval> &occurs,
                                                bool closed) override;

        virtual void buildConstraintMinimum(string id, vector<XVariable *> &list, XCondition &xc) override;

        virtual void buildConstraintMinimum(string id, vector<XVariable *> &list, XVariable *index, int startIndex, RankType rank,
                                            XCondition &xc) override;

        virtual void buildConstraintMaximum(string id, vector<XVariable *> &list, XCondition &xc) override;

        virtual void buildConstraintMaximum(string id, vector<XVariable *> &list, XVariable *index, int startIndex, RankType rank,
                                            XCondition &xc) override;

        virtual void buildConstraintElement(string id, vector<XVariable *> &list, int value) override;

        virtual void buildConstraintElement(string id, vector<XVariable *> &list, XVariable *value) override;

        virtual void buildConstraintElement(string id, vector<XVariable *> &list, int startIndex, XVariable *index, RankType rank, int value) override;

        virtual void buildConstraintElement(string id, vector<XVariable *> &list, int startIndex, XVariable *index, RankType rank, XVariable *value) override;

        virtual void buildConstraintElement(string id, vector<int> &list, int startIndex, XVariable *index, RankType rank, XVariable *value) override;

        virtual void buildConstraintElement(string id, vector<vector<int> > &matrix, int startRowIndex, XVariable *rowIndex, int startColIndex, XVariable* colIndex, XVariable *value) override;

        virtual void buildConstraintChannel(string id, vector<XVariable *> &list, int startIndex) override;

        virtual void buildConstraintChannel(string id, vector<XVariable *> &list1, int startIndex1, vector<XVariable *> &list2, int startIndex2) override;

        virtual void buildConstraintChannel(string id, vector<XVariable *> &list, int startIndex, XVariable *value) override;

        virtual void buildConstraintStretch(string id, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths) override;

        virtual void
        buildConstraintStretch(string id, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths, vector<vector<int>> &patterns) override;

        virtual void buildConstraintNoOverlap(string id, vector<XVariable *> &origins, vector<int> &lengths, bool zeroIgnored) override;

        virtual void buildConstraintNoOverlap(string id, vector<XVariable *> &origins, vector<XVariable *> &lengths, bool zeroIgnored) override;

        virtual void buildConstraintNoOverlap(string id, vector<vector<XVariable *>> &origins, vector<vector<int>> &lengths, bool zeroIgnored) override;

        virtual void buildConstraintNoOverlap(string id, vector<vector<XVariable *>> &origins, vector<vector<XVariable *>> &lengths, bool zeroIgnored) override;

        virtual void buildConstraintInstantiation(string id, vector<XVariable *> &list, vector<int> &values) override;

        virtual void buildConstraintClause(string id, vector<XVariable *> &positive, vector<XVariable *> &negative) override ;

        virtual void buildConstraintCircuit(string id, vector<XVariable *> &list, int startIndex) override;


        virtual void buildConstraintCircuit(string id, vector<XVariable *> &list, int startIndex, int size) override;


        virtual void buildConstraintCircuit(string id, vector<XVariable *> &list, int startIndex, XVariable *size) override;


        virtual void buildObjectiveMinimizeExpression(string expr) override;

        virtual void buildObjectiveMaximizeExpression(string expr) override;


        virtual void buildObjectiveMinimizeVariable(XVariable *x) override;


        virtual void buildObjectiveMaximizeVariable(XVariable *x) override;


        virtual void buildObjectiveMinimize(ExpressionObjective type, vector<XVariable *> &list, vector<int> &coefs) override;


        virtual void buildObjectiveMaximize(ExpressionObjective type, vector<XVariable *> &list, vector<int> &coefs) override;


        virtual void buildObjectiveMinimize(ExpressionObjective type, vector<XVariable *> &list) override;


        virtual void buildObjectiveMaximize(ExpressionObjective type, vector<XVariable *> &list) override;


        virtual void buildAnnotationDecision(vector<XVariable*> &list) override;
        bool canonize;
        bool debug = false;
        ModelBuilder* model_builder;
    };


}

using namespace XCSP3Core;

XCSP3_turbo_callbacks::XCSP3_turbo_callbacks(ModelBuilder* model_builder)
  : XCSP3CoreCallbacks(), canonize(true), model_builder(model_builder) {}

template<class T>
void displayList(vector<T> &list, string separator = " ") {
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++)
            cout << list[i] << separator;
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++)
            cout << list[i] << separator;
        cout << endl;
        return;
    }
    for(unsigned int i = 0; i < list.size(); i++)
        cout << list[i] << separator;
    cout << endl;
}


void displayList(vector<XVariable *> &list, string separator = " ") {
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++)
            cout << list[i]->id << separator;
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++)
            cout << list[i]->id << separator;
        cout << endl;
        return;
    }
    for(unsigned int i = 0; i < list.size(); i++)
        cout << list[i]->id << separator;
    cout << endl;
}


void XCSP3_turbo_callbacks::beginInstance(InstanceType type) {
  if(debug) {
    cout << "Start Instance - type=" << type << endl;
  }
}


void XCSP3_turbo_callbacks::endInstance() {
  if(debug) {
    cout << "End SAX parsing " << endl;
  }
}


void XCSP3_turbo_callbacks::beginVariables() {
  if(debug) {
    cout << " start variables declaration" << endl;
  }
}


void XCSP3_turbo_callbacks::endVariables() {
  if(debug) {
    cout << " end variables declaration" << endl << endl;
  }
}


void XCSP3_turbo_callbacks::beginVariableArray(string id) {
  if(debug) {
    cout << "    array: " << id << endl;
  }
}


void XCSP3_turbo_callbacks::endVariableArray() {
}


void XCSP3_turbo_callbacks::beginConstraints() {
  if(debug) {
    cout << " start constraints declaration" << endl;
  }
}


void XCSP3_turbo_callbacks::endConstraints() {
  if(debug) {
    cout << "\n end constraints declaration" << endl << endl;
  }
}


void XCSP3_turbo_callbacks::beginGroup(string id) {
  if(debug) {
    cout << "   start group of constraint " << id << endl;
  }
}


void XCSP3_turbo_callbacks::endGroup() {
  if(debug) {
    cout << "   end group of constraint" << endl;
  }
}


void XCSP3_turbo_callbacks::beginBlock(string classes) {
  if(debug) {
    cout << "   start block of constraint classes = " << classes << endl;
  }
}


void XCSP3_turbo_callbacks::endBlock() {
  if(debug) {
    cout << "   end block of constraint" << endl;
  }
}


// string id, bool circular
void XCSP3_turbo_callbacks::beginSlide(string id, bool) {
  if(debug) {
    cout << "   start slide " << id << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::endSlide() {
  if(debug) {
    cout << "   end slide" << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::beginObjectives() {
  if(debug) {
    cout << "   start Objective " << endl;
  }
}


void XCSP3_turbo_callbacks::endObjectives() {
  if(debug) {
    cout << "   end Objective " << endl;
  }
}


void XCSP3_turbo_callbacks::beginAnnotations() {
  if(debug) {
    cout << "   begin Annotations " << endl;
  }
  throw std::runtime_error("annotations unsupported");
}


void XCSP3_turbo_callbacks::endAnnotations() {
  if(debug) {
    cout << "   end Annotations " << endl;
  }
}


void XCSP3_turbo_callbacks::buildVariableInteger(string id, int minValue, int maxValue) {
  if(debug) {
    cout << "    var " << id << " : " << minValue << "..." << maxValue << endl;
  }
  model_builder->add_var(id, minValue, maxValue);
}


void XCSP3_turbo_callbacks::buildVariableInteger(string id, vector<int> &values) {
  if(debug) {
    cout << "    var " << id << " : ";
    cout << "        ";
    displayList(values);
  }
  throw std::runtime_error("set variable unsupported");
}


void XCSP3_turbo_callbacks::buildConstraintExtension(string id, vector<XVariable *> list, vector<vector<int>> &tuples, bool support, bool hasStar) {
  if(debug) {
    cout << "\n    extension constraint : " << id << endl;
    cout << "        " << (support ? "support" : "conflict") << " arity: " << list.size() << " nb tuples: " << tuples.size() << " star: " << hasStar << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildConstraintExtension(string id, XVariable *variable, vector<int> &tuples, bool support, bool hasStar) {
  if(debug) {
    cout << "\n    extension constraint with one variable: " << id << endl;
    cout << "        " <<(*variable) << " "<< (support ? "support" : "conflict") << " nb tuples: " << tuples.size() << " star: " << hasStar << endl;
    cout << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> list, bool support, bool hasStar
void XCSP3_turbo_callbacks::buildConstraintExtensionAs(string id, vector<XVariable *>, bool, bool) {
  if(debug) {
    cout << "\n    extension constraint similar as previous one: " << id << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

void XCSP3_turbo_callbacks::buildConstraintIntension(string id, string expr) {
  if(debug) {
    cout << "\n    intension constraint (using string) : " << id << " : " << expr << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

void XCSP3_turbo_callbacks::buildConstraintIntension(string id, Tree *tree) {
  if(debug) {
    cout << "\n    intension constraint using canonized tree: " << id << " : ";
    tree->prefixe();
    std::cout << "\n";
  }
  Propagator* p = model_builder->intensional_constraint(tree->root);
  model_builder->push(p);
}

void XCSP3_turbo_callbacks::buildConstraintPrimitive(string id, OrderType op, XVariable *x, int k, XVariable *y) {
  if(debug) {
    cout << "\n   intension constraint " << id << ": " << x->id << (k >= 0 ? "+" : "") << k << " op " << y->id << endl;
  }
  Propagator* p = model_builder->temporal_constraint(x, k, op, y);
  model_builder->push(p);
}

void XCSP3_turbo_callbacks::buildConstraintPrimitive(string id, OrderType op, XVariable *x, int k) {
  if(debug) {
    cout << "\n   constraint  " << id << ":" << x->id << " op " << k << "\n";
  }
  model_builder->domain_constraint(x, op, k);
}

void XCSP3_turbo_callbacks::buildConstraintPrimitive(string id, XVariable *x, bool in, int min, int max) {
  if(debug) {
    cout << "\n   constraint  " << id << ":"<< x->id << (in ? " in " : " not in ") << min << ".." << max <<"\n";
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, string start, vector<string> &final, vector<XTransition> &transitions
void XCSP3_turbo_callbacks::buildConstraintRegular(string, vector<XVariable *> &list, string start, vector<string> &final, vector<XTransition> &transitions) {
  if(debug) {
    cout << "\n    regular constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        start: " << start << endl;
    cout << "        final: ";
    displayList(final, ",");
    cout << endl;
    cout << "        transitions: ";
    for(unsigned int i = 0; i < (transitions.size() > 4 ? 4 : transitions.size()); i++) {
        cout << "(" << transitions[i].from << "," << transitions[i].val << "," << transitions[i].to << ") ";
    }
    if(transitions.size() > 4) cout << "...";
    cout << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<XTransition> &transitions
void XCSP3_turbo_callbacks::buildConstraintMDD(string, vector<XVariable *> &list, vector<XTransition> &transitions) {
  if(debug) {
    cout << "\n    mdd constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        transitions: ";
    for(unsigned int i = 0; i < (transitions.size() > 4 ? 4 : transitions.size()); i++) {
        cout << "(" << transitions[i].from << "," << transitions[i].val << "," << transitions[i].to << ") ";
    }
    if(transitions.size() > 4) cout << "...";
    cout << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildConstraintAlldifferent(string id, vector<XVariable *> &list) {
  if(debug) {
    cout << "\n    allDiff constraint" << id << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildConstraintAlldifferentExcept(string id, vector<XVariable *> &list, vector<int> &except) {
  if(debug) {
    cout << "\n    allDiff constraint with exceptions" << id << endl;
    cout << "        ";
    displayList(list);
    cout << "        Exceptions:";
    displayList(except);
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildConstraintAlldifferent(string id, vector<Tree *> &list) {
  if(debug) {
    cout << "\n    allDiff constraint with expresions" << id << endl;
    cout << "        ";
    for(Tree *t : list) {
        t->prefixe();std::cout << " ";
    }
    std::cout << std::endl;
  }
  throw std::runtime_error("constraint unsupported");
}

void XCSP3_turbo_callbacks::buildConstraintAlldifferentList(string id, vector<vector<XVariable *>> &lists) {
  if(debug) {
    cout << "\n    allDiff list constraint" << id << endl;
    for(unsigned int i = 0; i < (lists.size() < 4 ? lists.size() : 3); i++) {
        cout << "        ";
        displayList(lists[i]);

    }
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildConstraintAlldifferentMatrix(string id, vector<vector<XVariable *>> &matrix) {
  if(debug) {
    cout << "\n    allDiff matrix constraint" << id << endl;
    for(unsigned int i = 0; i < matrix.size(); i++) {
        cout << "        ";
        displayList(matrix[i]);
    }
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildConstraintAllEqual(string id, vector<XVariable *> &list) {
  if(debug) {
    cout << "\n    allEqual constraint" << id << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildConstraintNotAllEqual(string id, vector<XVariable *> &list) {
  if(debug) {
    cout << "\n    not allEqual constraint" << id << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, OrderType order
void XCSP3_turbo_callbacks::buildConstraintOrdered(string, vector<XVariable *> &list, OrderType order) {
  if(debug) {
    cout << "\n    ordered constraint" << endl;
    string sep;
    if(order == LT) sep = " < ";
    if(order == LE) sep = " <= ";
    if(order == GT) sep = " > ";
    if(order == GE) sep = " >= ";
    cout << "        ";
    displayList(list, sep);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> &lengths, OrderType order
void XCSP3_turbo_callbacks::buildConstraintOrdered(string, vector<XVariable *> &list, vector<int> &lengths, OrderType order) {
  if(debug) {
    cout << "\n    ordered constraint with lengths" << endl;
    string sep;
    if(order == LT) sep = " < ";
    if(order == LE) sep = " <= ";
    if(order == GT) sep = " > ";
    if(order == GE) sep = " >= ";
    cout << "        ";
    displayList(lengths); cout << "      ";
    displayList(list, sep);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<vector<XVariable *>> &lists, OrderType order
void XCSP3_turbo_callbacks::buildConstraintLex(string, vector<vector<XVariable *>> &lists, OrderType order) {
  if(debug) {
    cout << "\n    lex constraint   nb lists: " << lists.size() << endl;
    string sep;
    if(order == LT) sep = " < ";
    if(order == LE) sep = " <= ";
    if(order == GT) sep = " > ";
    if(order == GE) sep = " >= ";
    cout << "        operator: " << sep << endl;
    for(unsigned int i = 0; i < lists.size(); i++) {
        cout << "        list " << i << ": ";
        cout << "        ";
        displayList(lists[i], " ");
    }
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<vector<XVariable *>> &matrix, OrderType order
void XCSP3_turbo_callbacks::buildConstraintLexMatrix(string, vector<vector<XVariable *>> &matrix, OrderType order) {
  if(debug) {
    cout << "\n    lex matrix constraint   matrix  " << endl;
    string sep;
    if(order == LT) sep = " < ";
    if(order == LE) sep = " <= ";
    if(order == GT) sep = " > ";
    if(order == GE) sep = " >= ";

    for(unsigned int i = 0; i < (matrix.size() < 4 ? matrix.size() : 3); i++) {
        cout << "        ";
        displayList(matrix[i]);
    }
    cout << "        Order " << sep << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> &coeffs, XCondition &cond
void XCSP3_turbo_callbacks::buildConstraintSum(string, vector<XVariable *> &list, vector<int> &coeffs, XCondition &cond) {
  if(debug) {
    cout << "\n        sum constraint:";
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++)
            cout << (coeffs.size() == 0 ? 1 : coeffs[i]) << "*" << *(list[i]) << " ";
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++)
            cout << (coeffs.size() == 0 ? 1 : coeffs[i]) << "*" << *(list[i]) << " ";
    } else {
        for(unsigned int i = 0; i < list.size(); i++)
            cout << (coeffs.size() == 0 ? 1 : coeffs[i]) << "*" << *(list[i]) << " ";
    }
    cout << cond << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, XCondition &cond
void XCSP3_turbo_callbacks::buildConstraintSum(string, vector<XVariable *> &list, XCondition &cond) {
  if(debug) {
    cout << "\n        unweighted sum constraint:";
    cout << "        ";
    displayList(list, "+");
    cout << cond << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<XVariable *> &coeffs, XCondition &cond
void XCSP3_turbo_callbacks::buildConstraintSum(string, vector<XVariable *> &list, vector<XVariable *> &coeffs, XCondition &cond) {
  if(debug) {
    cout << "\n        scalar sum constraint:";
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++)
            cout << coeffs[i]->id << "*" << *(list[i]) << " ";
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++)
            cout << coeffs[i]->id << "*" << *(list[i]) << " ";
    } else {
        for(unsigned int i = 0; i < list.size(); i++)
            cout << coeffs[i]->id << "*" << *(list[i]) << " ";
    }
    cout << cond << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildConstraintSum(string id, vector<Tree *> &list, vector<int> &coeffs, XCondition &cond) {
  if(debug) {
    std::cout << "\n        sum with expression constraint;";
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++) {
            cout << coeffs[i];
            list[i]->prefixe();
        }
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++) {
            cout << coeffs[i];
            list[i]->prefixe();
        }
    } else {
        for(unsigned int i = 0; i < list.size(); i++) {
            cout << coeffs[i];
            list[i]->prefixe();
        }
    }
    cout << cond << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

void XCSP3_turbo_callbacks::buildConstraintSum(string id, vector<Tree *> &list, XCondition &cond) {
  if(debug) {
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++) {
            list[i]->prefixe();
        }
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++) {
            list[i]->prefixe();
        }
    } else {
        for(unsigned int i = 0; i < list.size(); i++) {
            list[i]->prefixe();
        }
    }
    cout << cond << endl;
  }
  throw std::runtime_error("constraint unsupported");
}



// string id, vector<XVariable *> &list, int value, int k
void XCSP3_turbo_callbacks::buildConstraintAtMost(string, vector<XVariable *> &list, int value, int k) {
  if(debug) {
    cout << "\n    AtMost constraint: val=" << value << " k=" << k << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int value, int k
void XCSP3_turbo_callbacks::buildConstraintAtLeast(string, vector<XVariable *> &list, int value, int k) {
  if(debug) {
    cout << "\n    Atleast constraint: val=" << value << " k=" << k << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int value, int k
void XCSP3_turbo_callbacks::buildConstraintExactlyK(string, vector<XVariable *> &list, int value, int k) {
  if(debug) {
    cout << "\n    Exactly constraint: val=" << value << " k=" << k << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> &values, int k
void XCSP3_turbo_callbacks::buildConstraintAmong(string, vector<XVariable *> &list, vector<int> &values, int k) {
  if(debug) {
    cout << "\n    Among constraint: k=" << k << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int value, XVariable *x
void XCSP3_turbo_callbacks::buildConstraintExactlyVariable(string, vector<XVariable *> &list, int value, XVariable *x) {
  if(debug) {
    cout << "\n    Exactly Variable constraint: val=" << value << " variable=" << *x << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> &values, XCondition &xc
void XCSP3_turbo_callbacks::buildConstraintCount(string, vector<XVariable *> &list, vector<int> &values, XCondition &xc) {
  if(debug) {
    cout << "\n    count constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        values: ";
    cout << "        ";
    displayList(values);
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<XVariable *> &values, XCondition &xc
void XCSP3_turbo_callbacks::buildConstraintCount(string, vector<XVariable *> &list, vector<XVariable *> &values, XCondition &xc) {
  if(debug) {
    cout << "\n    count constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        values: ";
    displayList(values);
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> &except, XCondition &xc
void XCSP3_turbo_callbacks::buildConstraintNValues(string, vector<XVariable *> &list, vector<int> &except, XCondition &xc) {
  if(debug) {
    cout << "\n    NValues with exceptions constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        exceptions: ";
    displayList(except);
    cout << "        condition:" << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, XCondition &xc
void XCSP3_turbo_callbacks::buildConstraintNValues(string, vector<XVariable *> &list, XCondition &xc) {
  if(debug) {
    cout << "\n    NValues  constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        condition:" << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> values, vector<int> &occurs, bool closed
void XCSP3_turbo_callbacks::buildConstraintCardinality(string, vector<XVariable *> &list, vector<int> values, vector<int> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (int values, int occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> values, vector<XVariable *> &occurs, bool closed
void XCSP3_turbo_callbacks::buildConstraintCardinality(string, vector<XVariable *> &list, vector<int> values, vector<XVariable *> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (int values, var occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> values, vector<XInterval> &occurs, bool closed
void XCSP3_turbo_callbacks::buildConstraintCardinality(string, vector<XVariable *> &list, vector<int> values, vector<XInterval> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (int values, interval occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<XVariable *> values, vector<int> &occurs, bool closed
void XCSP3_turbo_callbacks::buildConstraintCardinality(string, vector<XVariable *> &list, vector<XVariable *> values, vector<int> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (var values, int occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<XVariable *> values, vector<XVariable *> &occurs, bool closed
void XCSP3_turbo_callbacks::buildConstraintCardinality(string, vector<XVariable *> &list, vector<XVariable *> values, vector<XVariable *> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (var values, var occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<XVariable *> values, vector<XInterval> &occurs, bool closed
void XCSP3_turbo_callbacks::buildConstraintCardinality(string, vector<XVariable *> &list, vector<XVariable *> values, vector<XInterval> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (var values, interval occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, XCondition &xc
void XCSP3_turbo_callbacks::buildConstraintMinimum(string, vector<XVariable *> &list, XCondition &xc) {
  if(debug) {
    cout << "\n    minimum  constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, XVariable *index, int startIndex, RankType rank, XCondition &xc
void XCSP3_turbo_callbacks::buildConstraintMinimum(string, vector<XVariable *> &list, XVariable *index, int startIndex, RankType, XCondition &xc) {
  if(debug) {
    cout << "\n    arg_minimum  constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        index:" << *index << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, XCondition &xc
void XCSP3_turbo_callbacks::buildConstraintMaximum(string, vector<XVariable *> &list, XCondition &xc) {
  if(debug) {
    cout << "\n    maximum  constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, XVariable *index, int startIndex, RankType rank, XCondition &xc
void XCSP3_turbo_callbacks::buildConstraintMaximum(string, vector<XVariable *> &list, XVariable *index, int startIndex, RankType, XCondition &xc) {
  if(debug) {
    cout << "\n    arg_maximum  constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        index:" << *index << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int value
void XCSP3_turbo_callbacks::buildConstraintElement(string, vector<XVariable *> &list, int value) {
  if(debug) {
    cout << "\n    element constant constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << value << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, XVariable *value
void XCSP3_turbo_callbacks::buildConstraintElement(string, vector<XVariable *> &list, XVariable *value) {
  if(debug) {
    cout << "\n    element variable constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << *value << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int startIndex, XVariable *index, RankType rank, int value
void XCSP3_turbo_callbacks::buildConstraintElement(string, vector<XVariable *> &list, int startIndex, XVariable *index, RankType, int value) {
  if(debug) {
    cout << "\n    element constant (with index) constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << value << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        index : " << *index << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

void XCSP3_turbo_callbacks::buildConstraintElement(string id, vector<vector<int> > &matrix, int startRowIndex, XVariable *rowIndex, int startColIndex, XVariable* colIndex, XVariable *value) {
  if(debug) {
    cout << "\n    element matrix with rowIndex, colIndex and Value variables\n";
    for(unsigned int i = 0; i < matrix.size(); i++) {
        for(int j = 0; j < matrix.size(); j++)
            cout << matrix[i][j] << " ";
        cout << endl;
    }
    cout << "        row index : " << *rowIndex << endl;
    cout << "        col index : " << *colIndex << endl;
    cout << "        value     : " << *value << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, int startIndex, XVariable *index, RankType rank, XVariable *value
void XCSP3_turbo_callbacks::buildConstraintElement(string, vector<XVariable *> &list, int startIndex, XVariable *index, RankType, XVariable *value) {
  if(debug) {
    cout << "\n    element variable (with index) constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << *value << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        index : " << *index << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string, vector<int> &list, int startIndex, XVariable *index, RankType rank, XVariable *value
void XCSP3_turbo_callbacks::buildConstraintElement(string, vector<int> &list, int startIndex, XVariable *index, RankType, XVariable *value) {
  if(debug) {
    cout << "\n    element variable with list of integers (with index) constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << *value << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        index : " << *index << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int startIndex
void XCSP3_turbo_callbacks::buildConstraintChannel(string, vector<XVariable *> &list, int startIndex) {
  if(debug) {
    cout << "\n    channel constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        Start index : " << startIndex << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list1, int startIndex1, vector<XVariable *> &list2, int startIndex2
void XCSP3_turbo_callbacks::buildConstraintChannel(string, vector<XVariable *> &list1, int, vector<XVariable *> &list2, int) {
  if(debug) {
    cout << "\n    channel constraint" << endl;
    cout << "        list1 ";
    displayList(list1);
    cout << "        list2 ";
    displayList(list2);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int startIndex, XVariable *value
void XCSP3_turbo_callbacks::buildConstraintChannel(string, vector<XVariable *> &list, int, XVariable *value) {
  if(debug) {
    cout << "\n    channel constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << *value << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths
void XCSP3_turbo_callbacks::buildConstraintStretch(string, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths) {
  if(debug) {
    cout << "\n    stretch constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        values :";
    displayList(values);
    cout << "        widths:";
    displayList(widths);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths, vector<vector<int>> &patterns
void XCSP3_turbo_callbacks::buildConstraintStretch(string, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths, vector<vector<int>> &patterns) {
  if(debug) {
    cout << "\n    stretch constraint (with patterns)" << endl;
    cout << "        ";
    displayList(list);
    cout << "        values :";
    displayList(values);
    cout << "        widths:";
    displayList(widths);
    cout << "        patterns";
    for(unsigned int i = 0; i < patterns.size(); i++)
        cout << "(" << patterns[i][0] << "," << patterns[i][1] << ") ";
    cout << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &origins, vector<int> &lengths, bool zeroIgnored
void XCSP3_turbo_callbacks::buildConstraintNoOverlap(string, vector<XVariable *> &origins, vector<int> &lengths, bool) {
  if(debug) {
    cout << "\n    nooverlap constraint" << endl;
    cout << "        origins";
    displayList(origins);
    cout << "        lengths";
    displayList(lengths);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &origins, vector<XVariable *> &lengths, bool zeroIgnored
void XCSP3_turbo_callbacks::buildConstraintNoOverlap(string, vector<XVariable *> &origins, vector<XVariable *> &lengths, bool) {
  if(debug) {
    cout << "\n    nooverlap constraint" << endl;
    cout << "        origins:";
    displayList(origins);
    cout << "        lengths";
    displayList(lengths);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<vector<XVariable *>> &origins, vector<vector<int>> &lengths, bool zeroIgnored
void XCSP3_turbo_callbacks::buildConstraintNoOverlap(string, vector<vector<XVariable *>> &origins, vector<vector<int>> &lengths, bool) {
  if(debug) {
    cout << "\n    kdim (int lengths) nooverlap constraint" << endl;
    cout << "origins: " << endl;
    for(unsigned int i = 0; i < origins.size(); i++) {
        cout << "        ";
        displayList(origins[i]);
    }
    cout << "lengths: " << endl;
    for(unsigned int i = 0; i < origins.size(); i++) {
        cout << "        ";
        displayList(lengths[i]);
    }
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<vector<XVariable *>> &origins, vector<vector<XVariable *>> &lengths, bool zeroIgnored
void XCSP3_turbo_callbacks::buildConstraintNoOverlap(string, vector<vector<XVariable *>> &origins, vector<vector<XVariable *>> &lengths, bool) {
  if(debug) {
    cout << "\n    kdim (lenghts vars nooverlap constraint" << endl;
    cout << "origins: " << endl;
    for(unsigned int i = 0; i < origins.size(); i++) {
        cout << "        ";
        displayList(origins[i]);
    }
    cout << "lengths: " << endl;
    for(unsigned int i = 0; i < origins.size(); i++) {
        cout << "        ";
        displayList(lengths[i]);
    }
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> &values
void XCSP3_turbo_callbacks::buildConstraintInstantiation(string, vector<XVariable *> &list, vector<int> &values) {
  if(debug) {
    cout << "\n    instantiation constraint" << endl;
    cout << "        list:";
    displayList(list);
    cout << "        values:";
    displayList(values);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, vector<int> &values
void XCSP3_turbo_callbacks::buildConstraintClause(string, vector<XVariable *> &positive, vector<XVariable *> &negative) {
  if(debug) {
    cout << "\n    Clause constraint" << endl;
    cout << "        positive lits size:" << positive.size() <<" ";
    displayList(positive);
    cout << "        negative lits size:" << negative.size() <<" ";
    displayList(negative);
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int startIndex
void XCSP3_turbo_callbacks::buildConstraintCircuit(string, vector<XVariable *> &list, int startIndex) {
  if(debug) {
    cout << "\n    circuit constraint" << endl;
    cout << "        list:";
    displayList(list);
    cout << "        startIndex:" << startIndex << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int startIndex, int size
void XCSP3_turbo_callbacks::buildConstraintCircuit(string, vector<XVariable *> &list, int startIndex, int size) {
  if(debug) {
    cout << "\n    circuit constraint" << endl;
    cout << "        list:";
    displayList(list);
    cout << "        startIndex:" << startIndex << endl;
    cout << "        size: " << size << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int startIndex, XVariable *size
void XCSP3_turbo_callbacks::buildConstraintCircuit(string, vector<XVariable *> &list, int startIndex, XVariable *size) {
  if(debug) {
    cout << "\n    circuit constraint" << endl;
    cout << "        list:";
    displayList(list);
    cout << "        startIndex:" << startIndex << endl;
    cout << "        size: " << size->id << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildObjectiveMinimizeExpression(string expr) {
  if(debug) {
    cout << "\n    objective: minimize" << expr << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildObjectiveMaximizeExpression(string expr) {
  if(debug) {
    cout << "\n    objective: maximize" << expr << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildObjectiveMinimizeVariable(XVariable *x) {
  if(debug) {
    cout << "\n    objective: minimize variable " << x << endl;
  }
  model_builder->add_objective_minimize(x);
}


void XCSP3_turbo_callbacks::buildObjectiveMaximizeVariable(XVariable *x) {
  if(debug) {
    cout << "\n    objective: maximize variable " << x << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildObjectiveMinimize(ExpressionObjective type, vector<XVariable *> &list, vector<int> &coefs) {
  if(debug) {
    XCSP3CoreCallbacks::buildObjectiveMinimize(type, list, coefs);
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildObjectiveMaximize(ExpressionObjective type, vector<XVariable *> &list, vector<int> &coefs) {
  if(debug) {
    XCSP3CoreCallbacks::buildObjectiveMaximize(type, list, coefs);
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildObjectiveMinimize(ExpressionObjective type, vector<XVariable *> &list) {
  if(debug) {
    XCSP3CoreCallbacks::buildObjectiveMinimize(type, list);
  }
  throw std::runtime_error("constraint unsupported");
}


void XCSP3_turbo_callbacks::buildObjectiveMaximize(ExpressionObjective type, vector<XVariable *> &list) {
  if(debug) {
    XCSP3CoreCallbacks::buildObjectiveMaximize(type, list);
  }
  throw std::runtime_error("constraint unsupported");
}

void XCSP3_turbo_callbacks::buildAnnotationDecision(vector<XVariable*> &list) {
  if(debug) {
    std::cout << "       decision variables" << std::endl<< "       ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}

#endif //XCSP3_TURBO_CALLBACKS_HPP
