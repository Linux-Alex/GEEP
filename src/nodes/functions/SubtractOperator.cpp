//
// Created by aleks on 29.3.2025.
//

#include "SubtractOperator.h"

#include <stdexcept>

SubtractOperator::SubtractOperator(): FunctionNode("-") {
    size_t lowerLimit = 2, upperLimit = 2;
    setLimits(&lowerLimit, &upperLimit);
}

Node * SubtractOperator::clone() const {
    auto* clonedNode = new SubtractOperator();
    for (const auto& child: children) {
        clonedNode->addChild(child->clone());
    }
    return clonedNode;
}

double SubtractOperator::evaluate(const std::map<std::string, double> &variables) const {
    if (children.size() != 2) {
        throw std::runtime_error("SubtractOperator requires exactly two operands.");
    }
    double leftValue = children[0]->evaluate(variables);
    double rightValue = children[1]->evaluate(variables);

    return leftValue - rightValue;
}

std::string SubtractOperator::toString() const {
    return "(" + children[0]->toString() + " - " + children[1]->toString() + ")";
}
