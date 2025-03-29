//
// Created by aleks on 27.2.2025.
//

#include "AddOperator.h"

#include <stdexcept>

AddOperator::AddOperator(): FunctionNode("+") {
    size_t lowerLimit = 2, upperLimit = 2;
    setLimits(&lowerLimit, &upperLimit);
}

Node * AddOperator::clone() const {
    auto clonedNode = new AddOperator();

    for (const auto& child: children) {
        clonedNode->addChild(child->clone());
    }

    return clonedNode;
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
