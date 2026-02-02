//
// Created by aleks on 28.2.2025.
//

#ifndef TASK_H
#define TASK_H
#include "../GEEPConfig.h"
#include "../problems/Problem.h"
#include "../solutions/Solution.h"


class Task {
public:
    enum class ExecutionMode {
        CPU,
        GPU,
    };

protected:
    size_t id;

    Problem* problem;
    Solution* solution;

    GEEPConfig* config;

    ExecutionMode executionMode = ExecutionMode::GPU;
    int selectedCudaDevice = 0;

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
    Task& setProblem(Problem* problem) { this->problem = problem; return *this; }
    Task& setSolution(Solution* solution) { this->solution = solution; return *this; }
    Task& setExecutionMode(ExecutionMode mode) { executionMode = mode; return *this; }
    Task& setCudaDevice(int device) { selectedCudaDevice = device; return *this; }

    // Run the task
    void run();

    // Test GPU computing
    void testGPUComputing();

    // Specific implementations
    void runOnCPU();
    void runOnGPU();

    // Find best solution
    Solution* findBestSolution(const std::vector<Solution*>& solutions);

    // Find lowest MSE on GPU
    float findLowestMSEOnGPU(GPUTree& gpu_trees) { return gpu_trees.findLowestMSE(); };

    // Find lowest MSE on GPU
    float findLowestMSEOnCPU(const std::vector<Solution*>& solutions);
};



#endif //TASK_H
