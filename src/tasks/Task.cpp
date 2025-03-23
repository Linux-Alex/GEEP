//
// Created by aleks on 28.2.2025.
//

#include "Task.h"
#include <iostream>
#include <chrono>

#include "../../examples/LogHelper.h"

std::atomic<size_t> Task::ID_COUNTER = 0; // Initialize static member

Task::Task(const std::string &name): id(ID_COUNTER++), problem(nullptr), solution(nullptr) { }

Task::Task(const std::string &name, Problem *problem, Solution *solution, GEEPConfig* config) : id(ID_COUNTER++), problem(problem), solution(solution), config(config) { }

Task::~Task() {
    // Delete the solution
    delete solution;

    //
}

void Task::run() {
    LogHelper::logMessage("Starting task " + std::to_string(id) + " with problem " + problem->getName() + "...");

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Initialize evaluation counter
    size_t evaluations = 0;

    // Initialize generation counter
    size_t generations = 0;

    // Temporary solutions
    std::vector<Solution *> solutions;

    // Check if population size is set
    if (problem->getPopulationSize() == 0) {
        throw std::runtime_error("Population size not set.");
    }

    // Initialize the population
    for (size_t i = 0; i < problem->getPopulationSize(); i++) {
        Solution* solution = new Solution(problem->generateRandomSolution(5, 16));
        solutions.push_back(solution);
    }

    // Run the task
    while (!problem->getStopCrit().isMet(evaluations, generations, 0.0)) {
        // Evaluate solutions
        for (auto& solution : solutions) {
            double fitness = problem->evaluate(solution);
            evaluations++;
        }

        // Reproduction
        std::vector<Solution *> newSolutions;
        for (size_t i = 0; i < problem->getPopulationSize(); i++) {
            // Select parents
            Solution* parent1 = solutions[rand() % solutions.size()];
            Solution* parent2 = solutions[rand() % solutions.size()];

            // TODO: Crossover (maby make more than one child)
            Solution* child = problem->getCrossover()->crossover(parent1, parent2);

            // TODO: Mutate
            // child->mutate();

            // TODO: Add the child to the new solutions
            newSolutions.push_back(child);
        }
        generations++;
    }

    // Clean up dynamic memory
    for (auto& solution : solutions) {
        delete solution;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    LogHelper::logMessage("Task " + std::to_string(id) + " finished in " + std::to_string(duration) + " ms.");

    // throw std::runtime_error("Task::run() not implemented.");

}
