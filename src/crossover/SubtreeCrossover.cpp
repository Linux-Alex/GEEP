//
// Created by aleks on 16.3.2025.
//

#include "SubtreeCrossover.h"

std::vector<Solution *> SubtreeCrossover::crossover(Solution *parent1, Solution *parent2) {
    // Create children vector
    std::vector<Solution *> children;

    // Clone the nodes parents
    Node* clonedRoot1 = parent1->getRoot()->clone();
    Node* clonedRoot2 = parent2->getRoot()->clone();

    // Select random nodes from each cloned root
    Node* node1 = clonedRoot1->getRandomNode();
    Node* node2 = clonedRoot2->getRandomNode();

    // Swap the subtrees
    std::swap(node1, node2);

    // Set the new Solution and with the new root
    Solution* child1 = new Solution();
    child1->setRoot(clonedRoot1);

    // Add parents to ancestors
    child1->addAncestor(parent1);
    child1->addAncestor(parent2);

    // Add the first child to the children vector
    children.push_back(child1);

    // Create a second child by cloning the first child
    Solution* child2 = new Solution();
    child2->setRoot(clonedRoot2);

    // Add parents to ancestors
    child2->addAncestor(parent1);
    child2->addAncestor(parent2);

    // Add the second child to the children vector
    children.push_back(child2);

    // Return the new solution
    return children;
}