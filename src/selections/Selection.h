//
// Created by aleks on 15.3.2025.
//

#ifndef SELECTION_H
#define SELECTION_H

#include <vector>

#include "../solutions/Solution.h"

class Selection {
public:
    virtual ~Selection() = default;

    // Method for selecting a solution from the population
    virtual Solution* select(const std::vector<Solution*>& population) = 0;
};



#endif //SELECTION_H
