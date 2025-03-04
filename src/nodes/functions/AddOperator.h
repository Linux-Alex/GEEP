//
// Created by aleks on 27.2.2025.
//

#ifndef ADDOPERATOR_H
#define ADDOPERATOR_H

#include "../FunctionNode.h"

class AddOperator : public FunctionNode {
public:
    // Constructor
    AddOperator() : FunctionNode("+") {
        size_t lowerLimit = 2, upperLimit = 2;
        setLimits(&lowerLimit, &upperLimit);
    };

    // Constructor with parent
    AddOperator(Node* parent);

    // Constructor with parent and children
    AddOperator(Node* parent, const std::vector<Node*>& children);

    // Evaluate
    double evaluate(const std::map<std::string, double> &variables) const override;

    // String output
    std::string toString() const override;
};



#endif //ADDOPERATOR_H
