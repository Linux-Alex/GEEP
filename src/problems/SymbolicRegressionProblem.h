//
// Created by aleks on 2.3.2025.
//

#ifndef SYMBOLICREGRESSIONPROBLEM_H
#define SYMBOLICREGRESSIONPROBLEM_H
#include "Problem.h"


class SymbolicRegressionProblem : public Problem {
public:
    // Inherit constructors
    using Problem::Problem;

    // Evaluate the solution
    double evaluate(Solution *solution) override;
};



#endif //SYMBOLICREGRESSIONPROBLEM_H
