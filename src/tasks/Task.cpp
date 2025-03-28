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
            solution->setFitness(fitness);
            evaluations++;

            // TODO: Prfect solution found (finish breaking)
            if (fitness == 0) {
                LogHelper::logMessage("Perfect solution found in generation " + std::to_string(generations) + " with " + std::to_string(evaluations) + " evaluations.");
                break;
            }
        }

        // Reproduction
        std::vector<Solution *> newSolutions;

        while (newSolutions.size() < problem->getPopulationSize()) {
            // Select parents
            Solution* parent1 = problem->getSelection()->select(solutions);
            Solution* parent2 = problem->getSelection()->select(solutions);

            std::vector<Solution*> children = problem->getCrossover()->crossover(parent1, parent2);
            for (Solution* child : children) {
                // TODO: Mutate
                // child->mutate();

                if (newSolutions.size() < problem->getPopulationSize()) {
                    newSolutions.push_back(child);
                }
            }
        }

        // Replace the old solutions with the new ones
        for (auto& solution : solutions) {
            delete solution;
        }
        solutions = newSolutions;

        generations++;
    }

    // Evaluate the last generation
    for (auto& solution : solutions) {
        double fitness = problem->evaluate(solution);
        solution->setFitness(fitness);
        evaluations++;
    }

    // Find the best solution
    Solution* bestSolution = findBestSolution(solutions);
    LogHelper::logMessage("Best solution: " + std::to_string(bestSolution->getFitness()));

    // Print out solution
    LogHelper::logMessage(bestSolution->getRoot()->toString());


    // Clean up dynamic memory
    for (auto& solution : solutions) {
        delete solution;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    LogHelper::logMessage("Task " + std::to_string(id) + " finished in " + std::to_string(duration) + " ms.");

    // throw std::runtime_error("Task::run() not implemented.");

}

Solution * Task::findBestSolution(const std::vector<Solution *> &solutions) {
    if (solutions.empty()) {
        return nullptr;
    }

    Solution* bestSolution = solutions[0];
    for (size_t i = 1; i < solutions.size(); i++) {
        if (solutions[i]->getFitness() < bestSolution->getFitness()) {
            bestSolution = solutions[i];
        }
    }

    return bestSolution;
}
