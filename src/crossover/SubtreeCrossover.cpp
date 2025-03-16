//
// Created by aleks on 16.3.2025.
//

#include "SubtreeCrossover.h"

Solution * SubtreeCrossover::crossover(Solution *parent1, Solution *parent2) {
    // Select two random nodes from each parent
    Node* node1 = parent1->getRandomNode();
    Node* node2 = parent2->getRandomNode();

    // Swap the subtrees
    std::swap(node1, node2);
}
