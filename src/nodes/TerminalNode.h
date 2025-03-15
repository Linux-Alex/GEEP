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
    virtual std::unique_ptr<Node> clone() const override = 0;
};



#endif //TERMINALNODE_H
