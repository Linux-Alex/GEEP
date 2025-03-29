//
// Created by aleks on 2.3.2025.
//

#include "ConstNode.h"
#include "../../helpers/NumberConversions.h"

// Static range values
double ConstNode::MIN_VALUE = -10.0;
double ConstNode::MAX_VALUE = 10.0;

ConstNode::ConstNode() : TerminalNode(NumberConversions::doubleToString(value), nullptr) {
    // Random number between MIN_VALUE and MAX_VALUE
    value = MIN_VALUE + static_cast<double>(rand()) / RAND_MAX * (MAX_VALUE - MIN_VALUE);
}

double ConstNode::evaluate(const std::map<std::string, double> &variables) const {
    return value;
}

std::string ConstNode::toString() const {
    if (this->value < 0)
        return "(" + NumberConversions::doubleToString(value) + ")";
    return NumberConversions::doubleToString(value);
}
