//
// Created by aleks on 15.3.2025.
//

#include "TournamentSelection.h"

TournamentSelection::TournamentSelection(size_t tournamentSize): tournamentSize(tournamentSize) { }

Solution* TournamentSelection::select(const std::vector<Solution *> &population) {
    Solution* best = nullptr;

    for (size_t i = 0; i < tournamentSize; i++) {
        size_t index = rand() % population.size();

        Solution* candidate = population[index];
        if (!best || candidate->getFitness() < best->getFitness()) {
            best = candidate;
        }
    }

    return best;
}
