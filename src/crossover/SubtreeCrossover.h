//
// Created by aleks on 16.3.2025.
//

#ifndef SUBTREECROSSOVER_H
#define SUBTREECROSSOVER_H

#include "Crossover.h"
#include "../cuda/GPUTree.h"

class SubtreeCrossover : public Crossover {
public:
    // Perform crossover
    std::vector<Solution*> crossover(Solution* parent1, Solution* parent2) override;

    // Perform crossover on GPU
    void crossoverGPU(GPUTree* population, size_t parent_idx1, size_t node_idx1, size_t parent_idx2, size_t node_idx2) override;
};



#endif //SUBTREECROSSOVER_H
