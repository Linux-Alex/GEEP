//
// Created by aleks on 28.2.2025.
//

#include "Task.h"
#include <iostream>
#include <chrono>

std::atomic<size_t> Task::ID_COUNTER = 0; // Initialize static member

Task::Task(const std::string &name): id(ID_COUNTER++), problem(nullptr), solution(nullptr) { }

Task::Task(const std::string &name, Problem *problem, Solution *solution, GEEPConfig* config) : id(ID_COUNTER++), problem(problem), solution(solution), config(config) { }

Task::~Task() {
    // Delete the solution
    delete solution;

    //
}

void Task::run() {
    throw std::runtime_error("Task::run() not implemented.");

    // Initialize evaluation counter
    size_t evaluations = 0;

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // Main loop
    while (evaluations < problem->getStopCritMaxEvaluations()) {
        // Evaluate the solution
        double fitness = problem->evaluate(solution);
    }
}
