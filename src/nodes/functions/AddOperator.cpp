//
// Created by aleks on 27.2.2025.
//

#include "AddOperator.h"

#include <stdexcept>

AddOperator::AddOperator(Node *parent) : FunctionNode("+") {
    this->parent = parent;
}

AddOperator::AddOperator(Node *parent, const std::vector<Node *> &children) : FunctionNode("+") {
    this->parent = parent;
    this->children = children;
}

double AddOperator::evaluate(const std::map<std::string, double> &variables) const {
    if (getNumChildren() != 2) {
        throw std::invalid_argument("AddOperator::evaluate() requires two children");
    }

    return children[0]->evaluate(variables) + children[1]->evaluate(variables);
}

std::string AddOperator::toString() const {
    return "(" + children[0]->toString() + " + " + children[1]->toString()+ ")";
}
