//
// Created by aleks on 5.4.2025.
//

#ifndef GPUTREE_H
#define GPUTREE_H

#include <cuda_runtime.h>
#include <stack>

struct GPUTree {
    int* nodes;          // Node types (operators/terminals)
    float* values;       // Constant values for terminals
    int* children;       // Child indices (for operators)
    int* parent_indices; // Parent pointers
    size_t* node_counts; // Per-tree node counts
    size_t capacity;     // Max nodes per tree
    size_t population;   // Number of trees

    // Allocate unified memory
    void allocate(size_t max_nodes, size_t population_size);

    // Free memory
    void free();

    // Convert CPU solution to GPU representation
    void addSolution(int index, class Solution* solution);
};

#endif //GPUTREE_H
