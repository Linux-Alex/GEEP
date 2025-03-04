//
// Created by aleks on 2.3.2025.
//

#include "MultiplyOperator.h"

#include <stdexcept>


double MultiplyOperator::evaluate(const std::map<std::string, double> &variables) const {
    if (getNumChildren() != 2) {
        throw std::invalid_argument("AddOperator::evaluate() requires two children");
    }

    return children[0]->evaluate(variables) * children[1]->evaluate(variables);
}

std::string MultiplyOperator::toString() const {
    return "(" + children[0]->toString() + " * " + children[1]->toString() + ")";
}
