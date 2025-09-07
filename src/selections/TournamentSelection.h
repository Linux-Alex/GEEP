//
// Created by aleks on 15.3.2025.
//

#ifndef TOURNAMENTSELECTION_H
#define TOURNAMENTSELECTION_H

#include <stdexcept>

#include "Selection.h"

class TournamentSelection : public Selection {
private:
    size_t tournamentSize;
public:
    // Constructor
    TournamentSelection(size_t tournamentSize = 3);

    // Tournament selection
    Solution* select(const std::vector<Solution*>& population) override;

    // Get selected parents for crossover for CPU
    std::pair<int, int> getSelectedParentsForCrossover(const std::vector<Solution*>& population, float reproductionRate) override;

    // Get selected parents for crossover for GPU
    void getSelectedParentsForCrossoverGPU(GPUTree* population, float reproduction_rate) override;
};



#endif //TOURNAMENTSELECTION_H
