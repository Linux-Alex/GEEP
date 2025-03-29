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

    // Setters
    SymbolicRegressionProblem& setStopCrit(StopCriterion stopCrit) {
        Problem::setStopCrit(stopCrit);
        return *this;
    }

    SymbolicRegressionProblem& setSelection(Selection* selection) {
        Problem::setSelection(selection);
        return *this;
    }

    SymbolicRegressionProblem& setCrossover(Crossover* crossover) {
        Problem::setCrossover(crossover);
        return *this;
    }

    SymbolicRegressionProblem& setPopulationSize(size_t populationSize) {
        Problem::setPopulationSize(populationSize);
        return *this;
    }

    SymbolicRegressionProblem& setElitism(size_t elitism) {
        Problem::setElitism(elitism);
        return *this;
    }

    SymbolicRegressionProblem& setMaxDepth(size_t maxDepth) {
        Problem::setMaxDepth(maxDepth);
        return *this;
    }

    SymbolicRegressionProblem& setMaxNodes(size_t maxNodes) {
        Problem::setMaxNodes(maxNodes);
        return *this;
    }
};



#endif //SYMBOLICREGRESSIONPROBLEM_H
