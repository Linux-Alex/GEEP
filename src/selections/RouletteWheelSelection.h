//
// Created by aleks on 15.3.2025.
//

#ifndef ROULETTEWHEELSELECTION_H
#define ROULETTEWHEELSELECTION_H

#include "Selection.h"

class RouletteWheelSelection : public Selection {
public:
    // Roulette wheel selection
    Solution* select(const std::vector<Solution*>& population) override;
};



#endif //ROULETTEWHEELSELECTION_H
