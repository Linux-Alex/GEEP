//
// Created by aleks on 14. 11. 25.
//

#include "Mutation.h"

#include <c++/12/random>

#include "../problems/Problem.h"

void Mutation::mutation(Solution *&tree, Problem *problem) {
    if (tree == nullptr || problem == nullptr) return;

    float random = dist(gen);

    if (random < mutationRate) {
        // Make a clone of the solution
        Solution *s = new Solution();
        s->setRoot(tree->getRoot()->clone());

        Node* randomNode = s->getRandomNode();

        LogHelper::logMessage("Node before mutation: " + randomNode->toString());

        this->mutate(problem->getFunctionSet(),
                     problem->getTerminalSet(),
                     randomNode);

        LogHelper::logMessage("Node after mutation: " + randomNode->toString());

        delete tree;
        tree = s;
    }
}

void Mutation::mutate(std::vector<FunctionFactory> functionSet,
                      std::vector<TerminalFactory> terminalSet,
                      Node *&node) {
    // If node is null, return
    if (node == nullptr) return;

    // Determine the type of the node
    bool isFunctionNode = dynamic_cast<FunctionNode*>(node) != nullptr;
    bool isTerminalNode = dynamic_cast<TerminalNode*>(node) != nullptr;

    // Replace the node with a new random node of the same type
    if (isFunctionNode) {
        FunctionNode* newNode = Problem::generateRandomFunction(functionSet);
        if (newNode) {
            // Copy children from old node to new node
            FunctionNode* oldNode = dynamic_cast<FunctionNode*>(node);
            newNode->setChildren(oldNode->getChildren());

            // Replace the node
            node = newNode;
        }
    } else if (isTerminalNode) {
        TerminalNode* newNode = Problem::generateRandomTerminal(terminalSet);
        if (newNode) {
            // Replace the node
            node = newNode;
        }
    }
}