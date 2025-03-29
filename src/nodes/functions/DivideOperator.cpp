//
// Created by aleks on 29.3.2025.
//

#include "DivideOperator.h"

DivideOperator::DivideOperator(): FunctionNode("/") {
    size_t lowerLimit = 2, upperLimit = 2;
    setLimits(&lowerLimit, &upperLimit);
}

Node * DivideOperator::clone() const {
    auto* clonedNode = new DivideOperator();
    for (const auto& child: children) {
        clonedNode->addChild(child->clone());
    }
    return clonedNode;
}

double DivideOperator::evaluate(const std::map<std::string, double> &variables) const {
    if (children.size() != 2) {
        throw std::runtime_error("DivideOperator requires exactly two operands.");
    }
    double leftValue = children[0]->evaluate(variables);
    double rightValue = children[1]->evaluate(variables);
    if (rightValue == 0) {
        // throw std::runtime_error("Division by zero.");
        return 0; // handle division by zero gracefully
    }
    return leftValue / rightValue;
}

std::string DivideOperator::toString() const {
    return "(" + children[0]->toString() + " / " + children[1]->toString() + ")";
}
