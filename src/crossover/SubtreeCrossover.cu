//
// Created by aleks on 23.5.2025.
//

#include "SubtreeCrossover.h"
#include <curand_kernel.h>
#include <iostream>

__device__ int getRandomNode(
    int tree_idx,
    const int* nodes,
    const size_t* node_counts,
    curandState* state
) {
    size_t node_count = node_counts[tree_idx];
    return curand(state) % node_count;
}

__global__ void subtreeCrossoverKernel(
    // Input: Parent population
    const int* old_nodes,
    const float* old_values,
    const int* old_children,
    const size_t* old_node_counts,
    size_t old_capacity,
    const int* parent1_idx,
    const int* parent2_idx,
    size_t population_size,

    // Output: Child population
    int* new_nodes,
    float* new_values,
    int* new_children,
    size_t* new_node_counts,

    unsigned long seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= population_size) return;

    curandState state;
    curand_init(seed + i, 0, 0, &state);

    int p1_idx = parent1_idx[i];
    int p2_idx = parent2_idx[i];

    // Case 1: No crossover (only reproduction)
    if (p2_idx == -1) {
        // Copy parent 1 to child
        size_t node_count = old_node_counts[p1_idx];
        new_node_counts[i] = node_count;

        for (size_t j = 0; j < node_count; j++) {
            int old_offset = p1_idx * old_capacity + j;
            int new_offset = i * old_capacity + j;

            new_nodes[new_offset] = old_nodes[old_offset];
            new_values[new_offset] = old_values[old_offset];
            new_children[new_offset] = new_offset;
        }
        return;
    }

    // Case 2: Crossover between two parents
    int p1_node, p2_node;
    do {
        p1_node = getRandomNode(p1_idx, old_nodes, old_node_counts, &state);
        p2_node = getRandomNode(p2_idx, old_nodes, old_node_counts, &state);
    } while ((p1_node + 1) + (p2_node + 1) >= old_capacity);

    // Perform subtree swap
    int old_parent1_offset = p1_idx * old_capacity;
    int new_offset = i * old_capacity;

    // Copy parent 1 to child until p1_node is reached
    for (int j = 0; j < p1_node; j++) {
        new_nodes[new_offset] = old_nodes[old_parent1_offset];
        new_values[new_offset] = old_values[old_parent1_offset];
        new_children[new_offset] = new_offset;

        old_parent1_offset++;
        new_offset++;
    }

    // Copy subtree from parent 2 until the nodes from parent 2 have no more children
    float* todo_idx = new float[old_capacity];
    int todo_count = 0;

    // Check the idx of parent 2 node
    int old_parent2_offset = p2_idx * old_capacity + p2_node;

    do {
        // Add the children nodes to the todo_idx array
        if (old_children[old_parent2_offset * 2] != -1) {
            todo_idx[todo_count++] = old_children[old_parent2_offset * 2];
        }
        if (old_children[old_parent2_offset * 2 + 1] != -1) {
            todo_idx[todo_count++] = old_children[old_parent2_offset * 2 + 1];
        }

        // Add the parent 2 node to the child
        new_nodes[new_offset] = old_nodes[old_parent2_offset];
        new_values[new_offset] = old_values[old_parent2_offset];
        new_children[new_offset] = new_offset;

        new_offset++;
        old_parent2_offset++;
    } while (todo_count > 0);


}

void SubtreeCrossover::crossoverGPU(GPUTree* old_population, GPUTree* new_population) {
    // Get number of trees in the population
    int pop_size = old_population->population;

    // TODO: Allocate new population if needed (optional)

    // Launch kernel
    unsigned long seed = 22;
    int threads_per_block = 256;
    int blocks = (pop_size + threads_per_block - 1) / threads_per_block;

    subtreeCrossoverKernel<<<blocks, threads_per_block>>>(
        // Input: Parent population
        old_population->nodes,
        old_population->values,
        old_population->children,
        old_population->node_counts,
        old_population->capacity,
        old_population->selection_parent1_idx,
        old_population->selection_parent2_idx,
        pop_size,

        // Output: Child population
        new_population->nodes,
        new_population->values,
        new_population->children,
        new_population->node_counts,

        seed
    );

    // Fetch cuda errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in SubtreeCrossover::crossoverGPU: " << cudaGetErrorString(err) << std::endl;
    }

}
