//
// Created by aleks on 16.3.2025.
//

#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "../solutions/Solution.h"

class Crossover {
public:
    virtual ~Crossover() = default;

    // Method for performing crossover
    virtual std::vector<Solution*> crossover(Solution* parent1, Solution* parent2) = 0;
};



#endif //CROSSOVER_H
