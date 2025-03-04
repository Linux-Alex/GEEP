//
// Created by aleks on 2.3.2025.
//

#include "VariableNode.h"

VariableNode::VariableNode(const std::string &variableName) : TerminalNode(variableName, nullptr), variableName(variableName) { }

double VariableNode::evaluate(const std::map<std::string, double> &variables) const {
    return variables.at(variableName);
}

std::string VariableNode::toString() const {
    return variableName;
}
