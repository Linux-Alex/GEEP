//
// Created by aleks on 2.3.2025.
//

#include "SymbolicRegressionProblem.h"

#include <stdexcept>

double SymbolicRegressionProblem::evaluate(Solution *solution) {
    const std::vector<double>& values = solution->getValues();

    double fitness = 0.0;

    // TODO: Implement the evaluation of the solution

    for (const Target& target : targets) {
        // Target input state
        const std::map<std::string, double>& state = target.getState();

        if (solution->getRoot() == nullptr) {
            continue;
        }

        // Evaluate the solution
        double result = solution->getRoot()->evaluate(state);

        // Calculate the difference (error)
        double error = result - target.getTargetValue();

        // Add the square of the error to the fitness
        fitness += error * error;
    }

    // Return the fitness (lower is better)
    return fitness;

    // Default implementation
    // throw std::runtime_error("SymbolicRegressionProblem::evaluate() not implemented.");
}

void SymbolicRegressionProblem::setTargets(const std::vector<Target> &targets) { this->targets = targets; }
