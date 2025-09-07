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
    void crossoverGPU(GPUTree* old_population, GPUTree* new_population) override;

    // Reproduction rate getter and setter
    SubtreeCrossover& setReproductionRate(float rate) { Crossover::setReproductionRate(rate); return *this; }
    float getReproductionRate() const { return Crossover::getReproductionRate(); }

private:
    void swapSubtrees(GPUTree *population, int pos1, int pos2);

    void updateParentReferencesSimple(GPUTree *population, int abs_pos);

    void updateParentReferences(GPUTree *population, int root_pos);
};



#endif //SUBTREECROSSOVER_H
