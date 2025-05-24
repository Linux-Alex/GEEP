//
// Created by aleks on 23.5.2025.
//

#include "SubtreeCrossover.h"


void SubtreeCrossover::crossoverGPU(GPUTree* population, size_t parent_idx1, size_t node_idx1,
    size_t parent_idx2, size_t node_idx2) {
    // Extract single trees from the GPU representation
    GPUTree tree1 = population->extractTree(parent_idx1);
    GPUTree tree2 = population->extractTree(parent_idx2);

    // TODO: Finish crossover logic


}
