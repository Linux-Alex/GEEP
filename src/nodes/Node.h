//
// Created by aleks on 27.2.2025.
//

#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>
#include <atomic>
#include <map>
#include <memory>

// Abstract class representing a node in a tree
class Node {
protected:
    size_t id;
    std::string name;

    Node *parent{};

private:
    // Auto incrementing ID counter
    static std::atomic<size_t> ID_COUNTER;

public:
    // Constructor
    // Node(std::string name);
    Node(std::string name, Node *parent);

    // Destructor
    virtual ~Node() = default;

    // Clone method for deep copying
    virtual std::unique_ptr<Node> clone() const = 0;

    // Set parent
    void setParent(Node *parent);

    // Evaluate
    virtual double evaluate(const std::map<std::string, double>& variables) const = 0;

    // String representation
    virtual std::string toString() const = 0;
};



#endif //NODE_H
