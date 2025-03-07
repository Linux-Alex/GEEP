//
// Created by aleks on 28.2.2025.
//

#ifndef SOLUTION_H
#define SOLUTION_H

#include <atomic>
#include <vector>

#include "../nodes/FunctionFactory.h"
#include "../nodes/FunctionNode.h"
#include "../nodes/TerminalNode.h"

class Solution {
protected:
    size_t id;

    std::vector<double> values;
    std::vector<Solution*> ancestors;

    std::unique_ptr<Node> root; // Root of the expression tree

private:
    // Auto incrementing ID counter
    static std::atomic<size_t> ID_COUNTER;

public:
    // Constructor
    Solution();

    // Copy constructor
    Solution(const Solution& other);

    // Add ancestors
    void addAncestor(Solution* ancestor);
    void addAncestor(const std::vector<Solution*>& ancestors);

    // Set parents
    void setAncestor(Solution* ancestor);
    void setAncestors(const std::vector<Solution*>& ancestors);

    // Set values
    void setValue(double value);
    void setValues(const std::vector<double>& values);

    // Set root
    void setRoot(std::unique_ptr<Node> root);

    // Generate random solution
    Solution& generateRandomSolution(std::vector<FunctionFactory> *functionSet, std::vector<TerminalNode*> *terminalSet);

    // Getters
    size_t getId() const { return id; }
    const std::vector<double>& getValues() const { return values; }
    const std::vector<Solution*>& getAncestors() const { return ancestors; }

    // Setters
    void setId(size_t id) { this->id = id; }
};



#endif //SOLUTION_H
