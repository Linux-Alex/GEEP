//
// Created by aleks on 1.3.2025.
//

#include "Problem.h"

#include <stdexcept>
#include <random>

// Auto incrementing ID counter
std::atomic<size_t> Problem::ID_COUNTER = 0; // Initialize static member

Problem::Problem(std::string name) : id(Problem::ID_COUNTER++), name(std::move(name)) {
    shortName = std::to_string(Problem::ID_COUNTER);
}

Problem::Problem(std::string name, StopCriterion stopCrit, size_t dimensions, const std::vector<double> &upperLimits,
    const std::vector<double> &lowerLimits, ObjectiveType objectiveType) {
    Problem(std::move(name));
    this->stopCrit = stopCrit;
    this->objectiveType = objectiveType;
}

Problem::Problem(std::string name, std::string shortName, std::string description, StopCriterion stopCrit,
    size_t dimensions, const std::vector<double> &upperLimits, const std::vector<double> &lowerLimits,
    ObjectiveType objectiveType) {
    Problem give_me_a_name(std::move(name), stopCrit, dimensions, upperLimits, lowerLimits, objectiveType);
    this->shortName = std::move(shortName);
    this->description = std::move(description);
}

Solution Problem::generateRandomSolution(size_t maxDepth, size_t maxNodes) {
    // Initialize solution
    Solution solution;

    // Track the number of nodes in tree
    size_t nodeCount = 0;

    // Build the random expression tree
    auto root = buildRandomTree(functionSet, terminalSet, 0, maxDepth, nodeCount, maxNodes);

    // Set the root of the solution's expression tree
    solution.setRoot(std::move(root));

    return solution;
}

double Problem::evaluate(Solution *solution) {
    // Default implementation
    throw std::runtime_error("Problem::evaluate() not implemented.");
}


// Helper function to generate a random number within a range
int Problem::getRandomInt(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

// Helper function to generate a random terminal node
std::unique_ptr<TerminalNode> Problem::generateRandomTerminal(const std::vector<TerminalFactory> &terminalSet) {
    return terminalSet[getRandomInt(0, terminalSet.size() - 1)](); // Call the factory function to create a new TerminalNode
}

// Helper function to generate a random function node
std::unique_ptr<FunctionNode> Problem::generateRandomFunction(const std::vector<FunctionFactory>& functionSet) {
    return functionSet[getRandomInt(0, functionSet.size() - 1)](); // Call the factory function to create a new FunctionNode
}

// Recursive function to build expression tree
Node *Problem::buildRandomTree(const std::vector<FunctionFactory> &functionSet,
                               const std::vector<TerminalFactory> &terminalSet, size_t currentDepth,
                               size_t maxDepth, size_t &nodeCount, size_t maxNodes) {
    // Check if the maximum numbers of nodes has been reached
    if (nodeCount >= maxNodes) {
        return nullptr;
    }

    // Decide whether to create a terminal or function node
    if ((currentDepth >= maxDepth) || (getRandomInt(0, 1) == 0)) {
        // Create a terminal node
        nodeCount++;
        return generateRandomTerminal(terminalSet).release();
        // return std::unique_ptr<Node>(generateRandomTerminal(terminalSet));
    }
    else {
        // Create a function node
        auto functionNode = generateRandomFunction(functionSet);
        nodeCount++;

        // Recursively build children
        for (size_t i = 0; i < functionNode->getNumChildren(); i++) {
            auto child = buildRandomTree(functionSet, terminalSet, currentDepth + 1, maxDepth, nodeCount, maxNodes);

            if (child) {
                functionNode->addChild(child);
            }
            else {
                // Stop building children if the maximum number of nodes has been reached
                break;
            }
        }

        return functionNode.release();
    }
}
