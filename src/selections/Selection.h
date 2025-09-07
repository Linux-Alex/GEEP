//
// Created by aleks on 15.3.2025.
//

#ifndef SELECTION_H
#define SELECTION_H

#include <vector>
#include "../solutions/Solution.h"
#include "../cuda/GPUTree.h"

class Selection {
public:
    virtual ~Selection() = default;

    // Method for selecting a solution from the population
    virtual Solution* select(const std::vector<Solution*>& population) = 0;

    // Get selected parents for crossover for CPU
    virtual std::pair<int, int> getSelectedParentsForCrossover(const std::vector<Solution*>& population, float reproductionRate) = 0;

    // Get selected parents for crossover for GPU
    virtual void getSelectedParentsForCrossoverGPU(GPUTree* population, float reproduction_rate) = 0;
};



#endif //SELECTION_H
