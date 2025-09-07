//
// Created by aleks on 16.3.2025.
//

#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "../cuda/GPUTree.h"
#include "../solutions/Solution.h"

class Crossover {
private:
    float reproductionRate = 0.9f;

public:
    virtual ~Crossover() = default;

    // Method for performing crossover
    virtual std::vector<Solution*> crossover(Solution* parent1, Solution* parent2) = 0;

    // Method for performing crossover on GPU
    virtual void crossoverGPU(GPUTree* old_population, GPUTree* new_population) {};

    // Chained setter and getter for reproduction rate
    Crossover& setReproductionRate(float rate) { reproductionRate = rate; return *this; }
    float getReproductionRate() const { return reproductionRate; }
};



#endif //CROSSOVER_H
