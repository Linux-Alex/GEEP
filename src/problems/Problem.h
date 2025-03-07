//
// Created by aleks on 1.3.2025.
//

#ifndef PROBLEM_H
#define PROBLEM_H

#include <string>
#include <atomic>
#include <vector>
#include <chrono>

#include "ObjectiveType.h"
#include "../criterions/StopCriterion.h"
#include "../nodes/FunctionFactory.h"
#include "../nodes/FunctionNode.h"
#include "../nodes/TerminalNode.h"
#include "../solutions/Solution.h"

class Problem {
private:
    size_t id;

    // Names and description
    std::string name;
    std::string shortName;
    std::string description;

protected:

    // Population size
    size_t populationSize;

    // Objective type
    ObjectiveType objectiveType;

    // Stop criterion
    StopCriterion stopCrit;

    // Max number of evaluations or iterations or MSE less than threshold or time
    size_t stopCritMaxIterations;
    size_t stopCritMaxEvaluations;
    double stopCritMinMse;
    std::chrono::seconds stopCritTime;

    // Function set
    std::vector<FunctionFactory> functionSet;

    // Terminal set
    std::vector<TerminalNode*> terminalSet;

    // Auto incrementing ID counter
    static std::atomic<size_t> ID_COUNTER;

public:
    // Constructor
    Problem(std::string name);
    Problem(std::string name, StopCriterion stopCrit, size_t dimensions, const std::vector<double>& upperLimits,
            const std::vector<double>& lowerLimits, ObjectiveType objectiveType);
    Problem(std::string name, std::string shortName, std::string description, StopCriterion stopCrit, size_t dimensions,
            const std::vector<double> &upperLimits, const std::vector<double> &lowerLimits, ObjectiveType objectiveType);

    // Getters
    size_t getId() const { return id; }
    const std::string& getName() const { return name; }
    const std::string& getShortName() const { return shortName; }
    const std::string& getDescription() const { return description; }
    StopCriterion getStopCrit() const { return stopCrit; }
    size_t getStopCritMaxEvaluations() const { return stopCritMaxEvaluations; }
    size_t getPopulationSize() const { return populationSize; }


    // Setters
    void setName(const std::string& name) { this->name = name; }
    void setShortName(const std::string& shortName) { this->shortName = shortName; }
    void setDescription(const std::string& description) { this->description = description; }
    void setStopCrit(StopCriterion stopCrit) { this->stopCrit = stopCrit; }
    void setStopCritMaxEvaluations(size_t stopCritMaxEvaluations) { this->stopCritMaxEvaluations = stopCritMaxEvaluations; }
    void setFunctionSet(std::vector<FunctionFactory> functionSet) { this->functionSet = functionSet; }
    void setTerminalSet(std::vector<TerminalNode*> terminalSet) { this->terminalSet = terminalSet; }
    void setPopulationSize(size_t populationSize) { this->populationSize = populationSize; }

    // Evaluate the solution
    virtual double evaluate(Solution* solution);
};



#endif //PROBLEM_H
