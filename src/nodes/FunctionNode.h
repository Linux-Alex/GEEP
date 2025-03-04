//
// Created by aleks on 27.2.2025.
//

#ifndef OPERATORNODE_H
#define OPERATORNODE_H

#include <functional>
#include <memory>

#include "Node.h"

class FunctionNode : public Node {
protected:
    std::vector<Node*> children;

    size_t* childUpperLimit;
    size_t* childLowerLimit;

public:
    // Inherit constructors
    using Node::Node;
    FunctionNode(std::string name) : Node(name, nullptr) { };
    FunctionNode(std::string name, Node *parent, const std::vector<Node*>& children);

    // Destructor
    ~FunctionNode();

    // Add a child to the node
    void addChild(Node *child);
    void addChildren(const std::vector<Node*>& children);

    // Set children
    void setChildren(const std::vector<Node*>& children);

    // Get number of children
    size_t getNumChildren() const { return children.size(); }

    // Set limits
    void setLimits(size_t* childLowerLimit, size_t* childUpperLimit);

};



#endif //OPERATORNODE_H
