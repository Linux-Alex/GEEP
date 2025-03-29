//
// Created by aleks on 27.2.2025.
//

#ifndef ADDOPERATOR_H
#define ADDOPERATOR_H

#include "../FunctionNode.h"

class AddOperator : public FunctionNode {
public:
    // Constructor
    AddOperator();

    // Clone method
    Node* clone() const override;

    // Evaluate
    double evaluate(const std::map<std::string, double> &variables) const override;

    // String output
    std::string toString() const override;

    // Estimated number of children
    size_t getNumChildren() const override { return 2; }
};



#endif //ADDOPERATOR_H
