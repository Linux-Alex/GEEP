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
    std::unique_ptr<Node> clone() const override {
        auto clonedNode = std::make_unique<AddOperator>();
        for (const auto& child: children) {
            clonedNode->addChild(child->clone().release());
        }
        return clonedNode;
    }

    // Evaluate
    double evaluate(const std::map<std::string, double> &variables) const override;

    // String output
    std::string toString() const override;
};



#endif //ADDOPERATOR_H
