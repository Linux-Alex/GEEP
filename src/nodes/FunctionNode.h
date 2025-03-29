//
// Created by aleks on 27.2.2025.
//

#ifndef FUNCTIONNODE_H
#define FUNCTIONNODE_H

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

    // Clone method
    virtual Node* clone() const override = 0;

    // Add a child to the node
    void addChild(Node *child);
    void addChildren(const std::vector<Node*>& children);

    // Set children
    void setChildren(const std::vector<Node*>& children);

    // Get children
    const std::vector<Node*>& getChildren() const { return children; }

    // Get number of children
    virtual size_t getNumChildren() const { return children.size(); }

    // Set limits
    void setLimits(size_t* childLowerLimit, size_t* childUpperLimit);

    // Evaluate the function
    double evaluate(const std::map<std::string, double>& variables) const override = 0;

    // To string
    std::string toString() const override = 0;

    // Get random node
    Node* getRandomNode() override;

    // Get all nodes
    std::vector<Node*> collectNodes() override;

    // Virtual estimated number of children
    virtual size_t getEstimatedNumberOfChildren() const { return 2; }

    // Get depth
    size_t getDepth() const override;

    // Get number of nodes
    size_t getNumOfNodes() const override;
};



#endif //OPERATORNODE_H
