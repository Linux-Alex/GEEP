//
// Created by aleks on 2.3.2025.
//

#ifndef SYMBOLICREGRESSIONPROBLEM_H
#define SYMBOLICREGRESSIONPROBLEM_H

#include "Problem.h"
#include "../cuda/GPUTree.h"
#include "../targets/Target.h"
#include <set>


class SymbolicRegressionProblem : public Problem {
private:
    // Targets for CPU
    std::vector<Target> targets;

    // Targets for GPU
    std::vector<float> flattened_targets;
    std::vector<float> target_values;
    size_t num_variables; // Number of unique variables

public:
    // Inherit constructors
    using Problem::Problem;

    // Evaluate the solution with CPU
    double evaluate(Solution *solution) override;

    // Evaluate the solution with GPU
    void gpuEvaluate(GPUTree& trees, float* fitnesses);

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

    // Prepare targets for GPU
    void prepareTargetData();

    // Returns pointer to GPU-ready target data
    float* getTargetData() const { return const_cast<float*>(flattened_targets.data()); }

    // Returns pointer to target values
    float* getTargetValues() const { return const_cast<float*>(target_values.data()); }

    // Returns number of targets
    size_t getNumTargets() const { return targets.size(); }

    // Returns number of variables
    size_t getNumVariables() const { return num_variables; }

};



#endif //SYMBOLICREGRESSIONPROBLEM_H
