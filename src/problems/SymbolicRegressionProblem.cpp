//
// Created by aleks on 2.3.2025.
//

#include "SymbolicRegressionProblem.h"

#include <stdexcept>

double SymbolicRegressionProblem::evaluate(Solution *solution) {
    const std::vector<double>& values = solution->getValues();

    double fitness = 0.0;

    // TODO: Implement the evaluation of the solution

    // Default implementation
    throw std::runtime_error("SymbolicRegressionProblem::evaluate() not implemented.");
}

void SymbolicRegressionProblem::setTargets(const std::vector<Target> &targets) { this->targets = targets; }
