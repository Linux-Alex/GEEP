//
// Created by aleks on 15.3.2025.
//

#include "RouletteWheelSelection.h"

Solution * RouletteWheelSelection::select(const std::vector<Solution *> &population) {
    double totalFitness = 0.0;

    for (const auto& solution : population) {
        totalFitness += solution->getFitness();
    }

    double randomValue = static_cast<double>(rand()) / RAND_MAX * totalFitness;
    double cumulativeFitness = 0.0;

    for (const auto& solution : population) {
        cumulativeFitness += solution->getFitness();
        if (cumulativeFitness >= randomValue) {
            return solution;
        }
    }

    // Fallback
    return population.back();
}
