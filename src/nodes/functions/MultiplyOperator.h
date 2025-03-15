//
// Created by aleks on 2.3.2025.
//

#ifndef MULTIPLYOPERATOR_H
#define MULTIPLYOPERATOR_H

#include "../FunctionNode.h"


class MultiplyOperator : public FunctionNode {
public:
    // Constructor
    MultiplyOperator() : FunctionNode("*") {
        size_t lowerLimit = 2, upperLimit = 2;
        setLimits(&lowerLimit, &upperLimit);
    };

    // Constructor with parent
    MultiplyOperator(Node* parent);

    // Constructor with parent and children
    MultiplyOperator(Node* parent, const std::vector<Node*>& children);

    // Clone method
    std::unique_ptr<Node> clone() const override {
        auto clonedNode = std::make_unique<MultiplyOperator>();
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



#endif //MULTIPLYOPERATOR_H
