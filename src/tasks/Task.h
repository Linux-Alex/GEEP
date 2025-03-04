//
// Created by aleks on 28.2.2025.
//

#ifndef TASK_H
#define TASK_H
#include "../GEEPConfig.h"
#include "../problems/Problem.h"
#include "../solutions/Solution.h"


class Task {
protected:
    size_t id;

    Problem* problem;
    Solution* solution;

    GEEPConfig* config;

private:
    // Auto incrementing ID counter
    static std::atomic<size_t> ID_COUNTER;

public:
    // Constructor
    Task(const std::string& name);
    Task(const std::string& name, Problem* problem, Solution* solution, GEEPConfig* config);

    // Destructor
    virtual ~Task();

    // Getters
    size_t getId() const { return id; }
    Problem* getProblem() const { return problem; }
    Solution* getSolution() const { return solution; }

    // Setters
    void setProblem(Problem* problem) { this->problem = problem; }
    void setSolution(Solution* solution) { this->solution = solution; }

    // Run the task
    void run();
};



#endif //TASK_H
