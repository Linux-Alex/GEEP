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
#include "../crossover/Crossover.h"
#include "../mutation/Mutation.h"
#include "../nodes/FunctionFactory.h"
#include "../nodes/FunctionNode.h"
#include "../nodes/TerminalFactory.h"
#include "../nodes/TerminalNode.h"
#include "../selections/Selection.h"
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

    // Function set
    std::vector<FunctionFactory> functionSet;

    // Terminal set
    std::vector<TerminalFactory> terminalSet;

    // Selection method
    Selection *selection;

    // Crossover method
    Crossover *crossover;
    Crossover *gpuCrossover;

    // Mutation method
    Mutation *mutation = nullptr;

    // Elitism
    size_t elitism;

    // Max depth and max nodes
    size_t maxDepth;
    size_t maxNodes;

    // Auto incrementing ID counter
    static std::atomic<size_t> ID_COUNTER;

    // Best solution
    std::string bestSolution;

    // Generation runned
    size_t generationsRunned;

    // Best fitness
    float bestFitness;

    // Time used
    std::chrono::duration<double> timeUsed;

private:
    // Helper Functions
    static int getRandomInt(int min, int max);
    static Node *buildRandomTree(const std::vector<FunctionFactory> &functionSet,
                                 const std::vector<TerminalFactory> &terminalSet,
                                 size_t currentDepth, size_t maxDepth,
                                 size_t &nodeCount, size_t maxNodes);

public:
    // Constructor
    Problem(std::string name);
    Problem(std::string name, StopCriterion stopCrit, size_t dimensions, const std::vector<double>& upperLimits,
            const std::vector<double>& lowerLimits, ObjectiveType objectiveType);
    Problem(std::string name, std::string shortName, std::string description, StopCriterion stopCrit, size_t dimensions,
            const std::vector<double> &upperLimits, const std::vector<double> &lowerLimits, ObjectiveType objectiveType);

    // Destructor
    virtual ~Problem();

    // Helper Functions
    static TerminalNode* generateRandomTerminal(const std::vector<TerminalFactory> &terminalSet);
    static FunctionNode* generateRandomFunction(const std::vector<FunctionFactory>& functionSet);

    // Getters
    size_t getId() const { return id; }
    const std::string& getName() const { return name; }
    const std::string& getShortName() const { return shortName; }
    const std::string& getDescription() const { return description; }
    const std::vector<FunctionFactory>& getFunctionSet() const { return functionSet; }
    const std::vector<TerminalFactory>& getTerminalSet() const { return terminalSet; }

    // Setters
    void setName(const std::string& name) { this->name = name; }
    void setShortName(const std::string& shortName) { this->shortName = shortName; }
    void setDescription(const std::string& description) { this->description = description; }
    void setFunctionSet(std::vector<FunctionFactory> functionSet) { this->functionSet = functionSet; }
    void setTerminalSet(std::vector<TerminalFactory> terminalSet) { this->terminalSet = terminalSet; }

    // Population size
    Problem& setPopulationSize(size_t populationSize) { this->populationSize = populationSize; return *this; }
    size_t getPopulationSize() const { return populationSize; }

    // Stop criterion
    Problem& setStopCrit(StopCriterion stopCrit) { this->stopCrit = stopCrit; return *this; }
    StopCriterion getStopCrit() const { return stopCrit; }

    // Selection method
    Problem& setSelection(Selection* selection) { this->selection = selection; return *this; }
    Selection* getSelection() { return selection; }

    // Crossover method
    Problem& setCrossover(Crossover* crossover) { this->crossover = crossover; return *this; }
    Crossover* getCrossover() { return crossover; }

    Problem& setGPUCrossover(Crossover* gpuCrossover) { this->gpuCrossover = gpuCrossover; return *this; }
    Crossover* getGPUCrossover() { return gpuCrossover; }

    // Mutation method
    Problem& setMutation(Mutation* mutation) { this->mutation = mutation; return *this; }
    Mutation* getMutation() { return mutation; }

    // Elitism
    Problem& setElitism(size_t elitism) { this->elitism = elitism; return *this; }
    size_t getElitism() const { return elitism; }

    // Max depth and max nodes
    Problem& setMaxDepth(size_t maxDepth) { this->maxDepth = maxDepth; return *this; }
    size_t getMaxDepth() const { return maxDepth; }
    Problem& setMaxNodes(size_t maxNodes) { this->maxNodes = maxNodes; return *this; }
    size_t getMaxNodes() const { return maxNodes; }

    // Best solution
    std::string getBestSolution() const { return bestSolution; }
    void setBestSolution(std::string bestSolution) { this->bestSolution = bestSolution; }

    // Generations runned
    size_t getGenerationsRunned() const { return generationsRunned; }
    void setGenerationsRunned(size_t generationsRunned) { this->generationsRunned = generationsRunned; }

    // Time used
    std::chrono::duration<double> getTimeUsed() const { return timeUsed; }
    void setTimeUsed(std::chrono::duration<double> timeUsed) { this->timeUsed = timeUsed; }

    // Best fitness
    float getBestFitness() const { return bestFitness; }
    void setBestFitness(float bestFitness) { this->bestFitness = bestFitness; }

    // Check if solution is in bounds (max depth and max nodes)
    bool isInBounds(Solution* solution);

    // Generate random solution
    Solution generateRandomSolution(size_t maxDepth, size_t maxNodes);

    // Evaluate the solution
    virtual float evaluate(Solution *solution);
};



#endif //PROBLEM_H
