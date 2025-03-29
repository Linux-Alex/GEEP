//
// Created by aleks on 29.3.2025.
//

#ifndef DIVIDEOPERATOR_H
#define DIVIDEOPERATOR_H
#include <stdexcept>

#include "../FunctionNode.h"


class DivideOperator : public FunctionNode {
public:
    // Constructor
    DivideOperator();;

    // Clone method
    Node* clone() const override;

    // Evaluate method
    double evaluate(const std::map<std::string, double> &variables) const override;

    // Returns the string representation of the node
    std::string toString() const override;
};



#endif //DIVIDEOPERATOR_H
