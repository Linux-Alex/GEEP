//
// Created by aleks on 27.2.2025.
//

#include "FunctionNode.h"

#include <stdexcept>

FunctionNode::FunctionNode(std::string name, Node *parent, const std::vector<Node *> &children): Node(name, parent), children(children) { }

FunctionNode::~FunctionNode() {
    for (Node *child : children) {
        delete child;
    }
}


void FunctionNode::addChild(Node *child) {
    // If set limits, then check if the number of children (old and new) is within the limits
    if (childUpperLimit != nullptr && childLowerLimit != nullptr) {
        if (this->children.size() + 1 > *childUpperLimit) {
            throw std::invalid_argument("Number of children is not within the limits.");
        }
    }

    children.push_back(child);
    child->setParent(this);
}

void FunctionNode::addChildren(const std::vector<Node *> &children) {
    // If set limits, then check if the number of children (old and new) is within the limits
    if (childUpperLimit != nullptr && childLowerLimit != nullptr) {
        if (this->children.size() + children.size() > *childUpperLimit) {
            throw std::invalid_argument("Number of children is not within the limits.");
        }
    }

    for (Node *child : children) {
        addChild(child);
    }
}

void FunctionNode::setChildren(const std::vector<Node *> &children) {
    // If set limits, then check if the number of children is within the limits
    if (childUpperLimit != nullptr && childLowerLimit != nullptr) {
        if (children.size() < *childLowerLimit || children.size() > *childUpperLimit) {
            throw std::invalid_argument("Number of children is not within the limits.");
        }
    }

    // If there are children, delete them
    for (Node *child : this->children) {
        delete child;
    }

    this->children = children;
}

void FunctionNode::setLimits(size_t *childLowerLimit, size_t *childUpperLimit) {
    this->childLowerLimit = childLowerLimit;
    this->childUpperLimit = childUpperLimit;
}

Node * FunctionNode::getRandomNode() {
    std::vector<Node*> nodes = { this };

    // Travel through the tree and collect all nodes
    for (Node* child : children) {
        if (child) {
            std::vector<Node*> childNodes = child->collectNodes();
            nodes.insert(nodes.end(), childNodes.begin(), childNodes.end());
        }
    }

    // Randomly select a node
    if (!nodes.empty()) {
        size_t randomIndex = rand() % nodes.size();
        return nodes[randomIndex];
    }

    return nullptr;
}

std::vector<Node *> FunctionNode::collectNodes() {
    std::vector<Node*> nodes = { this };

    for (Node* child : children) {
        if (child) {
            std::vector<Node*> childNodes = child->collectNodes();
            nodes.insert(nodes.end(), childNodes.begin(), childNodes.end());
        }
    }

    return nodes;
}
