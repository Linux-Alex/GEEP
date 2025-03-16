//
// Created by aleks on 27.2.2025.
//

#include "Node.h"

#include <utility>
#include <algorithm>
#include <stdexcept>

std::atomic<size_t> Node::ID_COUNTER = 0; // Initialize static member

// Node::Node(std::string name) :
//         id(ID_COUNTER++), name(std::move(name)) { }

Node::Node(std::string name, Node *parent) : id(ID_COUNTER++), name(std::move(name)), parent(parent) { }

void Node::setParent(Node *parent) {
    this->parent = parent;
}

