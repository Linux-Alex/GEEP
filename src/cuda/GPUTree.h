//
// Created by aleks on 5.4.2025.
//

#ifndef GPUTREE_H
#define GPUTREE_H

#include <cuda_runtime.h>
#include <stack>
#include "../solutions/Solution.h"
#include <unordered_map>

struct GPUTree {
    int* nodes;          // Node types (operators/terminals)
    float* values;       // Constant values for terminals
    int* children;       // Child indices (for operators)
    int* parent_indices; // Parent pointers
    size_t* node_counts; // Per-tree node counts
    size_t capacity;     // Max nodes per tree
    size_t population;   // Number of trees

    // Variable name to index mapping
    std::unordered_map<std::string, int> variable_indices;
    int nextVariableId = 0; // For assigning unique IDs to variables

    // Allocate unified memory
    void allocate(size_t max_nodes, size_t population_size);

    // Free memory
    void free();

    // Convert CPU solution to GPU representation
    void addSolution(int index, Solution* solution);

    // Convert GPU representation to std::vector<Solution*> and return
    std::vector<Solution*> getCPUSolutions();

    // Extract tree (possible solution) from population
    GPUTree extractTree(int index);

    // Get or create ID for a variable name
    int getVariableId(const std::string& name);

    // Clears the variable mapping
    void clearVariableMapping();
};

#endif //GPUTREE_H
