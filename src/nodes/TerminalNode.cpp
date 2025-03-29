//
// Created by aleks on 2.3.2025.
//

#include "TerminalNode.h"

Node * TerminalNode::getRandomNode() {
    return this;
}

std::vector<Node *> TerminalNode::collectNodes() {
    std::vector<Node*> nodes = { this };
    return nodes;
}

size_t TerminalNode::getDepth() const {
    return 1;
}

size_t TerminalNode::getNumOfNodes() const {
    return 1;
}
