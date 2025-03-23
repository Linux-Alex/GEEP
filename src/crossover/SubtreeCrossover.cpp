//
// Created by aleks on 16.3.2025.
//

#include "SubtreeCrossover.h"

Solution * SubtreeCrossover::crossover(Solution *parent1, Solution *parent2) {
    // Create a new solution
    Solution* child = new Solution();

    // Clone the nodes parents
    Node* clonedRoot1 = parent1->getRoot()->clone();
    Node* clonedRoot2 = parent2->getRoot()->clone();

    // Select random nodes from each cloned root
    Node* node1 = clonedRoot1->getRandomNode();
    Node* node2 = clonedRoot2->getRandomNode();

    // Swap the subtrees
    std::swap(node1, node2);

    // Set the new Solution and with the new root
    Solution* newSolution = new Solution();
    newSolution->setRoot(clonedRoot1);

    // Add parents to ancestors
    newSolution->addAncestor(parent1);
    newSolution->addAncestor(parent2);

    // Return the new solution
    return newSolution;
}
