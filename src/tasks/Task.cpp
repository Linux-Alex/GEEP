//
// Created by aleks on 28.2.2025.
//

#include "Task.h"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../../examples/LogHelper.h"
#include "../cuda/GPUTree.h"
#include "../problems/SymbolicRegressionProblem.h"

std::atomic<size_t> Task::ID_COUNTER = 0; // Initialize static member

Task::Task(const std::string &name): id(ID_COUNTER++), problem(nullptr), solution(nullptr) { }

Task::Task(const std::string &name, Problem *problem, Solution *solution, GEEPConfig* config) : id(ID_COUNTER++), problem(problem), solution(solution), config(config) { }

Task::~Task() {
    // Delete the solution
    delete solution;
}

void Task::run() {
    if (executionMode == ExecutionMode::CPU) {
        runOnCPU();
    } else if (executionMode == ExecutionMode::GPU) {
        runOnGPU();
    } else {
        throw std::runtime_error("Unknown execution mode.");
    }
}

void Task::runOnCPU() {
    LogHelper::logMessage("Starting task " + std::to_string(id) + " with problem " + problem->getName() + " (task running on CPU).");

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
        Solution* solution = new Solution(problem->generateRandomSolution(problem->getMaxDepth(), problem->getMaxNodes()));
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

        // Elitism
        size_t elitism = problem->getElitism();
        if (elitism > 0) {
            // Sort solutions by fitness
            std::sort(solutions.begin(), solutions.end(), [](Solution* a, Solution* b) {
                return a->getFitness() < b->getFitness();
            });

            // Add the best solutions to the new population
            for (size_t i = 0; i < elitism && i < solutions.size(); i++) {
                newSolutions.push_back(solutions[i]);
            }
        }

        while (newSolutions.size() < problem->getPopulationSize()) {
            // Select parents
            Solution* parent1 = problem->getSelection()->select(solutions);
            Solution* parent2 = problem->getSelection()->select(solutions);

            std::vector<Solution*> children = problem->getCrossover()->crossover(parent1, parent2);
            for (Solution* child : children) {
                // TODO: Mutate
                // child->mutate();

                if (newSolutions.size() < problem->getPopulationSize()) {
                    // Check if the solutions is in bounds with max depth and max nodes
                    if (problem->isInBounds(child)) {
                        // Add the child to the new population
                        newSolutions.push_back(child);
                    }
                    else {
                        // If not, delete the child
                        delete child;
                    }
                }
            }
        }

        // Replace the old solutions with the new ones
        for (auto& solution : solutions) {
            // Check if solution is not in newSolutions
            if (std::find(newSolutions.begin(), newSolutions.end(), solution) == newSolutions.end()) {
                // If not, delete the old solution
                delete solution;
            }
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
}

void Task::runOnGPU() {
    LogHelper::logMessage("Starting task " + std::to_string(id) + " with problem " + problem->getName() + " (task running on GPU).");

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Prepare target data for GPU
    dynamic_cast<SymbolicRegressionProblem*>(problem)->prepareTargetData();

    // Temporary solutions
    std::vector<Solution *> solutions;

    // Initialize GPU structures
    GPUTree gpu_trees;
    gpu_trees.allocate(problem->getMaxNodes(), problem->getPopulationSize());

    // Convert population
    for (size_t i = 0; i < problem->getPopulationSize(); i++) {
        Solution* solution = new Solution(problem->generateRandomSolution(problem->getMaxDepth(), problem->getMaxNodes()));
        solutions.push_back(solution);
        gpu_trees.addSolution(i, solution);
    }

    // Allocate fitness array
    float* gpu_fitness;
    cudaMallocManaged(&gpu_fitness, solutions.size() * sizeof(float));

    // Perform GPU evaluation
    dynamic_cast<SymbolicRegressionProblem*>(problem)->gpuEvaluate(gpu_trees, gpu_fitness);

    // TODO: Rest of the operations

    // Cleanup
    gpu_trees.free();
    cudaFree(gpu_fitness);

    throw std::runtime_error("GPU execution not implemented yet.");


    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    LogHelper::logMessage("Task " + std::to_string(id) + " finished in " + std::to_string(duration) + " ms.");
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
