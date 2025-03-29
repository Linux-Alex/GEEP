//
// Created by aleks on 16.3.2025.
//

#ifndef SUBTREECROSSOVER_H
#define SUBTREECROSSOVER_H

#include "Crossover.h"

class SubtreeCrossover : public Crossover {
public:
    // Perform crossover
    std::vector<Solution*> crossover(Solution* parent1, Solution* parent2) override;
};



#endif //SUBTREECROSSOVER_H
