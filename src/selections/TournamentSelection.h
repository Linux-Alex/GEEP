//
// Created by aleks on 15.3.2025.
//

#ifndef TOURNAMENTSELECTION_H
#define TOURNAMENTSELECTION_H

#include "Selection.h"

class TournamentSelection : public Selection {
private:
    size_t tournamentSize;
public:
    // Constructor
    TournamentSelection(size_t tournamentSize = 3);

    // Tournament selection
    Solution* select(const std::vector<Solution*>& population) override;
};



#endif //TOURNAMENTSELECTION_H
