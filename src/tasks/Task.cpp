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

    // Temporary solutions
    std::vector<Solution *> solutions;

    // Initialize the population
    for (size_t i = 0; i < problem->getPopulationSize(); i++) {
        solutions.push_back(new Solution());
    }

    // Run the task
    while (evaluations < problem->getStopCritMaxEvaluations()) {


        evaluations++;
    }

    throw std::runtime_error("Task::run() not implemented.");

}
