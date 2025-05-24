//
// Created by aleks on 16.3.2025.
//

#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "../cuda/GPUTree.h"
#include "../solutions/Solution.h"

class Crossover {
public:
    virtual ~Crossover() = default;

    // Method for performing crossover
    virtual std::vector<Solution*> crossover(Solution* parent1, Solution* parent2) = 0;

    // Method for performing crossover on GPU
    virtual void crossoverGPU(GPUTree *population, size_t parent_idx1, size_t node_idx1, size_t parent_idx2, size_t node_idx2) {};
};



#endif //CROSSOVER_H
