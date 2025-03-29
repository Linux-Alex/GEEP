//
// Created by aleks on 29.3.2025.
//

#ifndef SUBTRACTOPERATOR_H
#define SUBTRACTOPERATOR_H

#include "../FunctionNode.h"

class SubtractOperator : public FunctionNode {
public:
    // Constructor
    SubtractOperator();

    // Clone method
    Node* clone() const override;

    // Evaluate method
    double evaluate(const std::map<std::string, double> &variables) const override;

    // Returns the string representation of the node
    std::string toString() const override;
};



#endif //SUBTRACTOPERATOR_H
