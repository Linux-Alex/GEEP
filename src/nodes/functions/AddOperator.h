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

    // Clone method
    Node* clone() const override {
        auto clonedNode = std::make_unique<AddOperator>();
        for (const auto& child: children) {
            clonedNode->addChild(child->clone());
        }
        return clonedNode.release();
    }

    // Evaluate
    double evaluate(const std::map<std::string, double> &variables) const override;

    // String output
    std::string toString() const override;

    // Estimated number of children
    size_t getNumChildren() const override { return 2; }
};



#endif //ADDOPERATOR_H
