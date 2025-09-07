//
// Created by aleks on 28.2.2025.
//

#ifndef SOLUTION_H
#define SOLUTION_H

#include <atomic>
#include <vector>

#include "../nodes/FunctionFactory.h"
#include "../nodes/FunctionNode.h"
#include "../nodes/TerminalFactory.h"
#include "../nodes/TerminalNode.h"

class Solution {
protected:
    size_t id;

    std::vector<double> values;
    std::vector<Solution*> ancestors;

    Node* root; // Root of the expression tree

    // Fitness
    double fitness;

private:
    // Auto incrementing ID counter
    static std::atomic<size_t> ID_COUNTER;

    // Collect all nodes (helper function)
    void collectNodes(Node* node, std::vector<Node*>& nodes);

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
    Solution& setRoot(Node* root);

    // Fitness getter and setter
    double getFitness() const { return fitness; }
    void setFitness(double fitness) { this->fitness = fitness; }

    // Generate random solution
    Solution& generateRandomSolution(std::vector<FunctionFactory> *functionSet, std::vector<TerminalFactory> *terminalSet);

    // Getters
    size_t getId() const { return id; }
    const std::vector<double>& getValues() const { return values; }
    const std::vector<Solution*>& getAncestors() const { return ancestors; }
    Node* getRoot() const { return root; }

    // Setters
    void setId(size_t id) { this->id = id; }

    // Get random node from the tree
    Node* getRandomNode();
};



#endif //SOLUTION_H
