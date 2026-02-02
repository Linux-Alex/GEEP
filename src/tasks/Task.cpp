//
// Created by aleks on 28.2.2025.
//

#include "Task.h"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "../../examples/LogHelper.h"
#include "../cuda/GPUTree.h"
#include "../problems/SymbolicRegressionProblem.cuh"
#include "../elitism/Elitism.h"
#include "../nodes/terminals/VariableNode.h"

std::atomic<size_t> Task::ID_COUNTER = 0; // Initialize static member

Task::Task(const std::string &name): id(ID_COUNTER++), problem(nullptr), solution(nullptr) { }

Task::Task(const std::string &name, Problem *problem, Solution *solution, GEEPConfig* config) : id(ID_COUNTER++), problem(problem), solution(solution), config(config) { }

Task::~Task() {
    LogHelper::logMessage("All things in task " + std::to_string(id) + " finished. This task will be closed.");
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

void Task::testGPUComputing() {
    // Run testGPUComputing on problem as SymbolicRegressionProblem
    try {
        // dynamic_cast<SymbolicRegressionProblem*>(problem)->testGPUComputing();
        dynamic_cast<SymbolicRegressionProblem*>(problem)->testGPUEvaluation();
    }
    catch (const std::bad_cast& e) {
        LogHelper::logMessage("Error: Problem is not a SymbolicRegressionProblem.", true);
    }
    catch (const std::exception& e) {
        LogHelper::logMessage("Error: " + std::string(e.what()), true);
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

    // Evaluate solutions
    for (auto& solution : solutions) {
        double fitness = problem->evaluate(solution);
        solution->setFitness(fitness);
        evaluations++;
    }


    // Run the task
    while (!problem->getStopCrit().isMet(evaluations, generations, findLowestMSEOnCPU(solutions))) {
        // Evaluate solutions
        // for (auto& solution : solutions) {
        //     double fitness = problem->evaluate(solution);
        //     solution->setFitness(fitness);
        //     evaluations++;
        //
        //     // TODO: Prfect solution found (finish breaking)
        //     if (fitness == 0) {
        //         LogHelper::logMessage("Perfect solution found in generation " + std::to_string(generations) + " with " + std::to_string(evaluations) + " evaluations.");
        //         break;
        //     }
        // }

        if (generations == 0) {
            // Print all trees
            for (Solution* s : solutions) {
                LogHelper::logMessage("Initial tree: " + s->getRoot()->toString() + ", Fitness: " + std::to_string(s->getFitness()));
            }
            LogHelper::logMessage("-------------------");
        }


        // Debug print first tree
        LogHelper::logMessage("Tree index 0: " + solutions[0]->getRoot()->toString() + ", Fitness: " + std::to_string(solutions[0]->getFitness()));
        // Debug print index 5
        LogHelper::logMessage("Tree index 5: " + solutions[5]->getRoot()->toString() + ", Fitness: " + std::to_string(solutions[5]->getFitness()));
        // Debug print last tree
        LogHelper::logMessage("Tree index " + std::to_string(solutions.size() - 1) + ": " + solutions[solutions.size() - 1]->getRoot()->toString() + ", Fitness: " + std::to_string(solutions[solutions.size() - 1]->getFitness()));


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
            // Generate random number between 0 and 1
            float randomValue = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

            // Select parents
            Solution* parent1 = problem->getSelection()->select(solutions);
            Solution* parent2 = problem->getSelection()->select(solutions);

            std::vector<Solution*> children;

            // Check for reproduction
            if (problem->getCrossover()->getReproductionRate() >= randomValue) {
                children = {
                    &(new Solution())->setRoot(parent1->getRoot()->clone()),
                    &(new Solution())->setRoot(parent2->getRoot()->clone())
                };
                LogHelper::logMessage("Reproduction occurred, reproduction rate: " + std::to_string(problem->getCrossover()->getReproductionRate()) + ", random value: " + std::to_string(randomValue));
            }
            else {
                children = problem->getCrossover()->crossover(parent1, parent2);
            }

            for (Solution* child : children) {
                if (problem->getMutation() != nullptr)
                    problem->getMutation()->mutation(child, problem);

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

        // Print first tree of newSolution
        LogHelper::logMessage("Tree index 0 of newSolutions: " + newSolutions[0]->getRoot()->toString());
        // Print index 5 tree of newSolution
        LogHelper::logMessage("Tree index 5 of newSolutions: " + newSolutions[5]->getRoot()->toString());
        // Print last tree of newSolution
        LogHelper::logMessage("Tree index " + std::to_string(newSolutions.size() - 1) + " of newSolutions: " + newSolutions[newSolutions.size() - 1]->getRoot()->toString());

        // Replace the old solutions with the new ones
        for (auto& solution : solutions) {
            // Check if solution is not in newSolutions
            if (std::find(newSolutions.begin(), newSolutions.end(), solution) == newSolutions.end()) {
                // If not, delete the old solution
                delete solution;
            }
        }
        solutions = newSolutions;

        for (auto& solution : solutions) {
            double fitness = problem->evaluate(solution);
            solution->setFitness(fitness);
            evaluations++;
        }

        generations++;
    }

    // // Evaluate the last generation
    // for (auto& solution : solutions) {
    //     double fitness = problem->evaluate(solution);
    //     solution->setFitness(fitness);
    //     evaluations++;
    // }

    // Find the best solution
    Solution* bestSolution = findBestSolution(solutions);
    LogHelper::logMessage("Best solution: " + std::to_string(bestSolution->getFitness()));
    LogHelper::logMessage("Found in generation: " + std::to_string(generations));

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

    // Set CUDA device limit for printf
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024);

    Elitism elitism;
    elitism.setElitismCount(problem->getElitism());

    // Initialize GPU structures
    GPUTree gpu_trees;
    gpu_trees.allocate(problem->getMaxNodes(), problem->getPopulationSize());

    // Prepare initial population
    std::vector<Solution *> solutions; // Use vector for easier management (deleting)
    for (size_t i = 0; i < problem->getPopulationSize(); i++) {
        Solution* solution = new Solution(problem->generateRandomSolution(problem->getMaxDepth(), problem->getMaxNodes()));
        solutions.push_back(solution);
        gpu_trees.addSolution(i, solution);
    }

    // Prepare target data
    dynamic_cast<SymbolicRegressionProblem*>(problem)->prepareTargetData(gpu_trees);

    // Main evolution loop
    size_t evaluations = 0;
    size_t generations = 0;

    // GPU evaluation
    dynamic_cast<SymbolicRegressionProblem*>(problem)->gpuEvaluate(gpu_trees);
    evaluations += problem->getPopulationSize();

    // Calculate number of variables
    size_t numberOfVariables = std::count_if(
        problem->getTerminalSet().begin(),
        problem->getTerminalSet().end(),
        [](const std::function<TerminalNode*()>& fn) {
            return dynamic_cast<VariableNode*>(fn()) != nullptr;
        }
    );


    while (!problem->getStopCrit().isMet(evaluations, generations, findLowestMSEOnGPU(gpu_trees))) {
        // Write current generation info
        // LogHelper::logMessage("Generation " + std::to_string(generations) + ", Evaluations: " + std::to_string(evaluations));

        // Debug print generation ever 100 generations
        if (generations % 1000 == 0) {
            LogHelper::logMessage("Generation " + std::to_string(generations) + ", Evaluations: " + std::to_string(evaluations) + ", Lowest MSE: " + std::to_string(findLowestMSEOnGPU(gpu_trees)));
        }

        // CPU evaluation
        // solutions = gpu_trees.getCPUSolutions();
        // for (size_t i = 0; i < solutions.size(); i++) {
        //     double fitness = problem->evaluate(solutions[i]);
        //     gpu_trees.fitness_values[i] = static_cast<float>(fitness);
        // }

        // Check if any solution is wierd
        // solutions = gpu_trees.getCPUSolutions();
        // for (int i = 0; i < problem->getPopulationSize(); i++) {
        //     gpu_trees.addSolution(i, solutions[i]);
        // }

        // Print first 2 GPU trees for debugging
        // for (size_t i = 0; i < 8 && i < problem->getPopulationSize(); i++) {
        //     int node_count = gpu_trees.node_counts[i];
        //     std::string tree_str = "GPU Tree " + std::to_string(i) + ": ";
        //     for (size_t j = 0; j < node_count; j++) {
        //         tree_str += std::to_string((int)gpu_trees.nodes[i * problem->getMaxNodes() + j]) + " ";
        //     }
        //     tree_str += "\nValues: ";
        //     for (size_t j = 0; j < node_count; j++) {
        //         tree_str += std::to_string(gpu_trees.values[i * problem->getMaxNodes() + j]) + " ";
        //     }
        //     tree_str += "\nChildren:";
        //     for (size_t j = 0; j < node_count * 2; j++) {
        //         tree_str += std::to_string(gpu_trees.children[i * problem->getMaxNodes() * 2 + j]) + " ";
        //     }
        //     tree_str += "\nNode count: " + std::to_string(node_count);
        //     LogHelper::logMessage(tree_str);
        // }
        // solutions = gpu_trees.getCPUSolutions();


        // Print first 5 solutions for debugging
        // for (size_t i = 0; i < 6 && i < solutions.size(); i++) {
        //     LogHelper::logMessage("Initial solution " + std::to_string(i) + ": " + solutions[i]->getRoot()->toString());
        // }


        // Update CPU fitness values
        // for (size_t i = 0; i < solutions.size(); i++) {
        //     solutions[i]->setFitness(gpu_trees.fitness_values[i]);
        //     if (gpu_trees.fitness_values[i] == 0.0f) {
        //         LogHelper::logMessage("Prfect solution found in generation " + std::to_string(generations));
        //         break;
        //     }
        // }

        // Selection (on CPU)
        // std::vector<Solution *> newSolutions;

        // Selection (on GPU)
        GPUTree new_gpu_trees;
        new_gpu_trees.allocate(problem->getMaxNodes(), problem->getPopulationSize());

        // TODO: Elitism
        // if (problem->getElitism() > 0) {
        //     std::sort(solutions.begin(), solutions.end(), [](Solution* a, Solution* b) {
        //         return a->getFitness() < b->getFitness();
        //     });
        //
        //     for (size_t i = 0; i < problem->getElitism(); i++) {
        //         newSolutions.push_back(solutions[i]);
        //     }
        // }

        // Check if crossover is set
        if (problem->getCrossover() == nullptr) {
            throw std::runtime_error("Crossover method is not set.");
        }


        // Print out selected how many parents2 have -1 (index not selected)
        // int not_selected_count = 0;
        // for (size_t i = 0; i < gpu_trees.population; i++) {
        //     if (gpu_trees.selection_parent2_idx[i] == -1) {
        //         not_selected_count++;
        //     }
        // }
        // LogHelper::logMessage("Selected parents with -1 index (not selected): " + std::to_string(not_selected_count));

        // Selection
        problem->getSelection()->getSelectedParentsForCrossoverGPU(&gpu_trees, problem->getCrossover()->getReproductionRate() / 2.0f);

        try {
            // Crossover with reproduction
            problem->getCrossover()->crossoverGPU(&gpu_trees, &new_gpu_trees);
        } catch (const std::exception& e) {
            LogHelper::logMessage("Problem during crossover: " + std::string(e.what()) + ". Retrying crossover...", true);
        }

        try {
            // Mutation on GPU
            problem->getMutation()->mutationGPU(&new_gpu_trees, numberOfVariables);
        } catch (const std::exception& e) {
            LogHelper::logMessage("Problem during mutation: " + std::string(e.what()) + ". Retrying mutation...", true);
        }

        try {
            // Elitism
            elitism.applyElitismGPU(&gpu_trees, &new_gpu_trees)           ;
        } catch (const std::exception& e) {
            LogHelper::logMessage("Problem during elitism: " + std::string(e.what()) + ". Retrying elitism...", true);
        }

        // while (newSolutions.size() < problem->getPopulationSize()) {
        //     // Selection on CPU
        //     Solution* parent1 = problem->getSelection()->select(solutions);
        //     Solution* parent2 = problem->getSelection()->select(solutions);
        //
        //     // Crossover on GPU
        //     // int parent1_idx = std::distance(solutions.begin(), std::find(solutions.begin(), solutions.end(), parent1));
        //     // int parent2_idx = std::distance(solutions.begin(), std::find(solutions.begin(), solutions.end(), parent2));
        //
        //     // TODO: Implement GPU crossover kernel call
        //     // gpuCrossover(gpu_trees, new_gpu_trees, parent1_idx, parent2_idx, newSolutions_idx);
        //
        //     // For now fall back to CPU crossover
        //     std::vector<Solution *> children = problem->getCrossover()->crossover(parent1, parent2);
        //
        //     for (Solution* child : children) {
        //         if (newSolutions.size() < problem->getPopulationSize()) {
        //             newSolutions.push_back(child);
        //         }
        //     }
        // }

        // Replace population
        // Free old GPU memory
        // gpu_trees.free();
        // gpu_trees.allocate(problem->getMaxNodes(), problem->getPopulationSize());

        // Convert new solution to GPU format
        // for (size_t i = 0; i < newSolutions.size(); i++) {
        //     gpu_trees.addSolution(i, newSolutions[i]);
        // }

        // solutions = newSolutions;
        // gpu_trees.nodes = new_gpu_trees.nodes;
        // gpu_trees.values = new_gpu_trees.values;
        // gpu_trees.children = new_gpu_trees.children;
        // gpu_trees.parent_indices = new_gpu_trees.parent_indices;
        // gpu_trees.node_counts = new_gpu_trees.node_counts;

        // gpu_trees = std::move(new_gpu_trees);
        gpu_trees.moveDataFrom(std::move(new_gpu_trees));

        new_gpu_trees.free();

        // GPU evaluation
        dynamic_cast<SymbolicRegressionProblem*>(problem)->gpuEvaluate(gpu_trees);
        evaluations += problem->getPopulationSize();

        generations++;
    }

    // Final evaluation and cleanup
    dynamic_cast<SymbolicRegressionProblem*>(problem)->gpuEvaluate(gpu_trees);
    // CPU evaluation
    solutions = gpu_trees.getCPUSolutions();
    for (size_t i = 0; i < solutions.size(); i++) {
        // solutions[i]->setFitness(gpu_trees.fitness_values[i]);
        // double fitness = problem->evaluate(solutions[i]);
        // //gpu_trees.fitness_values[i] = static_cast<float>(fitness);
        //
        // if (abs(fitness - gpu_trees.fitness_values[i]) > (fitness + gpu_trees.fitness_values[i]) / 2 * 0.01f) {
        //     throw std::runtime_error("Mismatch between CPU and GPU fitness evaluation.\nCPU: " + std::to_string(fitness) + ", GPU: " + std::to_string(gpu_trees.fitness_values[i]) + "\nSolution: " + solutions[i]->getRoot()->toString());
        // }
    }

    // Update CPU fitness values
    for (size_t i = 0; i < solutions.size(); i++) {
        solutions[i]->setFitness(gpu_trees.fitness_values[i]);
    }

    // Print fitness values
    // for (size_t i = 0; i < solutions.size(); i++) {
    //     LogHelper::logMessage("Solution " + std::to_string(i) + ": " + std::to_string(solutions[i]->getFitness()) + " → " + solutions[i]->getRoot()->toString());
    // }

    // Find best solution
    Solution* bestSolution = findBestSolution(solutions);
    LogHelper::logMessage("Best solution: " + std::to_string(bestSolution->getFitness()));
    LogHelper::logMessage("Found in generation: " + std::to_string(generations));
    LogHelper::logMessage(bestSolution->getRoot()->toString());
    problem->setBestSolution(std::to_string(bestSolution->getFitness()));
    problem->setBestFitness(static_cast<float>(bestSolution->getFitness()));

    gpu_trees.clearVariableMapping();

    // Cleanup
    gpu_trees.free();
    for (auto& solution : solutions) {
        delete solution;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    // Time used in milliseconds
    problem->setTimeUsed(std::chrono::duration<double, std::milli>(duration));

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

float Task::findLowestMSEOnCPU(const std::vector<Solution *> &solutions) {
    float lowestMSE = std::numeric_limits<float>::max();
    for (const auto& solution : solutions) {
        if (solution->getFitness() < lowestMSE) {
            lowestMSE = static_cast<float>(solution->getFitness());
        }
    }
    // std::cout << "Lowest MSE on CPU: " << lowestMSE << std::endl;
    return lowestMSE;
}

