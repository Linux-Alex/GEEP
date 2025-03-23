//
// Created by aleks on 2.3.2025.
//

#ifndef VARIABLENODE_H
#define VARIABLENODE_H

#include <string>

#include "../TerminalNode.h"

class VariableNode : public TerminalNode {
private:
    std::string variableName;

public:
    // Inherit constructors
    using TerminalNode::TerminalNode;
    VariableNode(const std::string &variableName);

    // Clone method
    Node* clone() const override {
        return new VariableNode(variableName);
        //return std::make_unique<VariableNode>(variableName);
    }

    // Evaluate the expression
    double evaluate(const std::map<std::string, double>& variables) const override;

    // Convert to string
    std::string toString() const override;
};



#endif //VARIABLENODE_H
