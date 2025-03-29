//
// Created by aleks on 2.3.2025.
//

#ifndef MULTIPLYOPERATOR_H
#define MULTIPLYOPERATOR_H

#include "../FunctionNode.h"


class MultiplyOperator : public FunctionNode {
public:
    // Constructor
    MultiplyOperator();

    // Clone method
    Node* clone() const override;

    // Evaluate
    double evaluate(const std::map<std::string, double> &variables) const override;

    // String output
    std::string toString() const override;

    // Estimated number of children
    size_t getNumChildren() const override { return 2; }

};



#endif //MULTIPLYOPERATOR_H
