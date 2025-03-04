//
// Created by aleks on 1.3.2025.
//

#include "Problem.h"

#include <stdexcept>

// Auto incrementing ID counter
std::atomic<size_t> Problem::ID_COUNTER = 0; // Initialize static member

Problem::Problem(std::string name) : id(Problem::ID_COUNTER++), name(std::move(name)) {
    shortName = std::to_string(Problem::ID_COUNTER);
}

Problem::Problem(std::string name, StopCriterion stopCrit, size_t dimensions, const std::vector<double> &upperLimits,
    const std::vector<double> &lowerLimits, ObjectiveType objectiveType) {
    Problem(std::move(name));
    this->stopCrit = stopCrit;
    this->objectiveType = objectiveType;
}

Problem::Problem(std::string name, std::string shortName, std::string description, StopCriterion stopCrit,
    size_t dimensions, const std::vector<double> &upperLimits, const std::vector<double> &lowerLimits,
    ObjectiveType objectiveType) {
    Problem give_me_a_name(std::move(name), stopCrit, dimensions, upperLimits, lowerLimits, objectiveType);
    this->shortName = std::move(shortName);
    this->description = std::move(description);
}

double Problem::evaluate(Solution *solution) {
    // Default implementation
    throw std::runtime_error("Problem::evaluate() not implemented.");
}


