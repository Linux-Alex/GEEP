//
// Created by aleks on 2.3.2025.
//

#ifndef SYMBOLICREGRESSIONPROBLEM_H
#define SYMBOLICREGRESSIONPROBLEM_H

#include "Problem.h"
#include "../targets/Target.h"


class SymbolicRegressionProblem : public Problem {
private:
    std::vector<Target> targets;
public:
    // Inherit constructors
    using Problem::Problem;

    // Evaluate the solution
    double evaluate(Solution *solution) override;

    // Set targets
    void setTargets(const std::vector<Target> &targets);
};



#endif //SYMBOLICREGRESSIONPROBLEM_H
