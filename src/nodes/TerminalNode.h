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

    // Clone method
    virtual Node* clone() const override = 0;

    // Get random node
    Node* getRandomNode() override;

    // Get all nodes
    std::vector<Node*> collectNodes() override;

    // Get depth
    size_t getDepth() const override;

    // Get number of nodes
    size_t getNumOfNodes() const override;
};



#endif //TERMINALNODE_H
