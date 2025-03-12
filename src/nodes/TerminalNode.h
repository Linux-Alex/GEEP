//
// Created by aleks on 2.3.2025.
//

#ifndef TERMINALNODE_H
#define TERMINALNODE_H

#include "Node.h"


class TerminalNode : public Node {
public:
    // Inherit constructors
    using Node::Node;

    std::vector<Node *> children;

    // Add a child
    void addChild(Node *child) {
        children.push_back(child);
    }

    // Set children
    void setChildren(const std::vector<Node *> &children) {
        this->children = children;
    }

    // Get children
    const std::vector<Node *> &getChildren() const {
        return children;
    }
};



#endif //TERMINALNODE_H
