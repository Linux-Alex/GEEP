//
// Created by aleks on 28.2.2025.
//

#ifndef STOPCRITERIONTYPE_H
#define STOPCRITERIONTYPE_H

enum StopCriterionType {
    EVALUATIONS,  // Stop after a certain number of evaluations
    GENERATIONS,  // Stop after a certain number of generations
    MSE,          // Stop if the mean squared error is below a threshold
    TIME,         // Stop after a certain amount of time has passed
};

#endif //STOPCRITERIONTYPE_H
