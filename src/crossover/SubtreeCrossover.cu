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
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // Each thread handles two children
    if (i >= population_size) return;

    curandState state;
    curand_init(seed + i, 0, 0, &state);

    int p1_idx = parent1_idx[i];
    int p2_idx = parent2_idx[i];

    // Case 1: No crossover (only reproduction)
    if (p2_idx == -1) {
        size_t node_count = old_node_counts[p1_idx];
        new_node_counts[i] = node_count;

        for (size_t j = 0; j < node_count; j++) {
            int old_offset = p1_idx * (int)old_capacity + (int)j;
            int new_offset = i * (int)old_capacity + (int)j;

            new_nodes[new_offset] = old_nodes[old_offset];
            new_values[new_offset] = old_values[old_offset];
            new_children[new_offset * 2]     = new_children[new_offset * 2]; // keep default if present
            new_children[new_offset * 2 + 1] = new_children[new_offset * 2 + 1];
        }
        return;
    }

    // Case 2: Crossover between two parents
    int cross_pos1, cross_pos2;
    // do {
    //     p1_node = getRandomNode(p1_idx, old_nodes, old_node_counts, &state);
    //     p2_node = getRandomNode(p2_idx, old_nodes, old_node_counts, &state);
    // } while ((p1_node + 1) + (p2_node + 1) >= old_capacity);

    cross_pos1 = 1; // For testing purposes, select fixed nodes
    cross_pos2 = 0; // For testing purposes, select fixed nodes

    // Calculate offsets (to start at the beggining of the tree)
    int old_parent1_offset = p1_idx * (int)old_capacity;
    int old_parent2_offset = p2_idx * (int)old_capacity;

    int new1_offset = i * (int)old_capacity;
    int new2_offset = (i + 1) * (int)old_capacity;

    // DEBUGGING: Zero out new slot ranges to make printed output clearer
    // for (size_t j = 0; j < old_capacity; ++j) {
    //     new_nodes[new1_offset + j] = -9999;
    //     new_values[new1_offset + j] = 0.0f;
    //     new_children[(new1_offset + j) * 2] = -1;
    //     new_children[(new1_offset + j) * 2 + 1] = -1;
    //
    //     new_nodes[new2_offset + j] = -9999;
    //     new_values[new2_offset + j] = 0.0f;
    //     new_children[(new2_offset + j) * 2] = -1;
    //     new_children[(new2_offset + j) * 2 + 1] = -1;
    // }

    // Copy from original (but without children)
    for (int j = 0; j < cross_pos1; j++) {
        new_nodes[new1_offset + j] = old_nodes[old_parent1_offset + j];
        new_values[new1_offset + j] = old_values[old_parent1_offset + j];
        new_children[(new1_offset + j) * 2] = old_children[(old_parent1_offset + j) * 2];
        new_children[(new1_offset + j) * 2 + 1] = old_children[(old_parent1_offset + j) * 2 + 1];
    }

    for (int j = 0; j < cross_pos2; j++) {
        new_nodes[new2_offset + j] = old_nodes[old_parent2_offset + j];
        new_values[new2_offset + j] = old_values[old_parent2_offset + j];
        new_children[(new2_offset + j) * 2] = old_children[(old_parent2_offset + j) * 2];
        new_children[(new2_offset + j) * 2 + 1] = old_children[(old_parent2_offset + j) * 2 + 1];
    }

    // Copy from pair parent
    int children_todo = 1;
    int delta_pos = cross_pos1 - cross_pos2;
    int crosspart1_len, crosspart2_len;

    // new_nodes[0] = -2786; // For debugging purposes
    // return;

    for (crosspart1_len = 0; children_todo > 0; crosspart1_len++, children_todo--) {
        int current_offset = new1_offset + cross_pos1 + crosspart1_len;
        int old_offset = old_parent2_offset + cross_pos2 + crosspart1_len;

        // Copy nodes and values
        new_nodes[current_offset] = old_nodes[old_offset];
        new_values[current_offset] = old_values[old_offset];

        // Copy both children
        new_children[(current_offset) * 2] = old_children[old_offset * 2];
        new_children[(current_offset) * 2 + 1] = old_children[old_offset * 2 + 1];

        // Check if nodes have actual children
        if (old_children[old_offset * 2] != -1) {
            children_todo++;
            new_children[current_offset * 2] += delta_pos;
        }
        if (old_children[old_offset * 2 + 1] != -1) {
            children_todo++;
            new_children[current_offset * 2 + 1] += delta_pos;
        }
    }

    // Reset for the second crossover
    children_todo = 1;

    for (crosspart2_len = 0; children_todo > 0; crosspart2_len++, children_todo--) {
        int current_offset = new2_offset + cross_pos2 + crosspart2_len;
        int old_offset = old_parent1_offset + cross_pos1 + crosspart2_len;

        // Copy nodes and values
        new_nodes[current_offset] = old_nodes[old_offset];
        new_values[current_offset] = old_values[old_offset];

        // Copy both children
        new_children[current_offset * 2] = old_children[old_offset * 2];
        new_children[current_offset * 2 + 1] = old_children[old_offset * 2 + 1];

        // Check if nodes have actual children
        if (old_children[old_offset * 2] != -1) {
            children_todo++;
            new_children[current_offset * 2] -= delta_pos;
        }
        if (old_children[old_offset * 2 + 1] != -1) {
            children_todo++;
            new_children[current_offset * 2 + 1] -= delta_pos;
        }
    }

    // Copy tail from original
    int delta_len = crosspart1_len - crosspart2_len;
    int tail1_len, tail2_len;

    for (tail1_len = 0; cross_pos1 + crosspart2_len + tail1_len < (int)old_node_counts[p1_idx]; tail1_len++) {
        int current_offset = new1_offset + cross_pos1 + crosspart1_len + tail1_len;
        int old_offset = old_parent1_offset + cross_pos1 + crosspart2_len + tail1_len;

        // Copy nodes and values
        new_nodes[current_offset] = old_nodes[old_offset];
        new_values[current_offset] = old_values[old_offset];

        // Copy both children
        new_children[current_offset * 2] = old_children[old_offset * 2];
        new_children[current_offset * 2 + 1] = old_children[old_offset * 2 + 1];

        if (old_children[old_offset * 2] != -1) {
            new_children[current_offset * 2] += delta_len;
        }
        if (old_children[old_offset * 2 + 1] != -1) {
            new_children[current_offset * 2 + 1] += delta_len;
        }
    }

    for (tail2_len = 0; cross_pos2 + crosspart1_len + tail2_len < (int)old_node_counts[p2_idx]; tail2_len++) {
        int current_offset = new2_offset + cross_pos2 + crosspart2_len + tail2_len;
        int old_offset = old_parent2_offset + cross_pos2 + crosspart1_len + tail2_len;

        // Copy nodes and values
        new_nodes[current_offset] = old_nodes[old_offset];
        new_values[current_offset] = old_values[old_offset];

        // Copy both children
        new_children[current_offset * 2] = old_children[old_offset * 2];
        new_children[current_offset * 2 + 1] = old_children[old_offset * 2 + 1];

        if (old_children[old_offset * 2] != -1) {
            new_children[current_offset * 2] -= delta_len;
        }
        if (old_children[old_offset * 2 + 1] != -1) {
            new_children[current_offset * 2 + 1] -= delta_len;
        }
    }

    // Repair children from 0 to cross_pos1 in new1
    for (int j = 0; j < cross_pos1; j++) {
        if (old_children[(old_parent1_offset + j) * 2] > cross_pos1) {
            new_children[(new1_offset + j) * 2] += delta_len;
        }
        if (old_children[(old_parent1_offset + j) * 2 + 1] > cross_pos1) {
            new_children[(new1_offset + j) * 2 + 1] += delta_len;
        }
    }

    for (int j = 0; j < cross_pos2; j++) {
        if (old_children[(old_parent2_offset + j) * 2] > cross_pos2) {
            new_children[(new2_offset + j) * 2] -= delta_len;
        }
        if (old_children[(old_parent2_offset + j) * 2 + 1] > cross_pos2) {
            new_children[(new2_offset + j) * 2 + 1] -= delta_len;
        }
    }

    // Fill in new_node_counts for both children (kernel didn't; we set them to parent's counts)
    new_node_counts[i] = cross_pos1 + crosspart1_len + tail1_len;
    if ((i + 1) < (int)population_size)
        new_node_counts[i + 1] = cross_pos2 + crosspart2_len + tail2_len;
}

void SubtreeCrossover::crossoverGPU(GPUTree* old_population, GPUTree* new_population) {
    // Get number of trees in the population
    int pop_size = old_population->population;

    // TODO: Allocate new population if needed (optional)

    // Launch kernel
    unsigned long seed = 22;
    int threads_per_block = 256;

    // Each thread will process one PAIR of trees, so we need half as many threads
    int num_pairs = pop_size / 2;
    int blocks = (num_pairs + threads_per_block - 1) / threads_per_block;

    // Print kernel sizes and other info
    std::cout << "Launching SubtreeCrossover kernel with " << blocks << " blocks of "
              << threads_per_block << " threads each for " << num_pairs
              << " tree pairs (population size " << pop_size << ")" << std::endl;

    std::cout << "Selection_parent1_idx: " << *old_population->selection_parent1_idx << std::endl;
    std::cout << "Selection_parent2_idx: " << *old_population->selection_parent2_idx << std::endl;

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

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    // Fetch cuda errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in SubtreeCrossover::crossoverGPU: " << cudaGetErrorString(err) << std::endl;
    }

}
