//
// Created by aleks on 2.3.2025.
//

#ifndef CONSTNODE_H
#define CONSTNODE_H


#include <map>
#include "../TerminalNode.h"


class ConstNode : public TerminalNode {
private:
    double value;

    static double MIN_VALUE;
    static double MAX_VALUE;

public:
    // Inherit constructors
    using TerminalNode::TerminalNode;
    ConstNode();

    // Evaluate
    double evaluate(const std::map<std::string, double>& variables) const override;

    // To string
    std::string toString() const override;
};



#endif //CONSTNODE_H
