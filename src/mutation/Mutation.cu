//
// Created by aleks on 14. 11. 25.
//

#include "Mutation.h"
#include <curand_kernel.h>
#include <chrono>
#include <iostream>

__device__ int getRandomInt(curandState* state, int min, int max) {
    return min + curand(state) % (max - min + 1);
}

__device__ float getRandomFloat(curandState* state, float min, float max) {
    return min + curand_uniform(state) * (max - min);
}

__device__ void mutateNode(int* nodes, float* values, int node_index, curandState* state, size_t numberOfVariables) {
    int node_type = nodes[node_index];

    // Different mutation strategies based on node type
    switch(node_type) {
        case 0:
        case 1:
            // Choose random between variable and constant mutation
            if (curand_uniform(state) < 0.5f) {
                // Variable node - change to constant or different variable?
                if (numberOfVariables > 0) {
                    int new_var = getRandomInt(state, 0, static_cast<int>(numberOfVariables) - 1);
                    values[node_index] = new_var; // Assuming variable nodes are represented by their index
                    nodes[node_index] = 0; // Set to variable node type
                }
            } else {
                // Constant node - perturb the value
                nodes[node_index] = 1; // Set to constant node type
                values[node_index] = getRandomFloat(state, -10.0f, 10.0f);
                // printf("Random float: %f\n", values[node_index]);
            }
            break;

        // case 0: // Variable node - change to constant or different variable?
        //     if (numberOfVariables > 0) {
        //         int new_var = getRandomInt(state, 0, static_cast<int>(numberOfVariables) - 1);
        //         nodes[node_index] = new_var; // Assuming variable nodes are represented by their index
        //     }
        //     break;
        //
        // case 1: // Constant node - perturb the value
        //     values[node_index] = getRandomFloat(state, -10.0f, 10.0f);
        //     // printf("Random float: %f\n", values[node_index]);
        //     break;

        default: // Function node - could change operator type
            int new_op = getRandomInt(state, 2, 5); // Assuming 2-5 are function operators
            nodes[node_index] = new_op;
            break;
    }
}

__global__ void mutationKernel(
    int* nodes,
    float* values,
    int* children,
    size_t* node_counts,
    size_t capacity,
    size_t population_size,
    float mutation_rate,
    size_t numberOfVariables,
    unsigned long seed
) {
    int tree_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tree_idx >= population_size) return;

    curandState state;
    curand_init(seed + tree_idx, 0, 0, &state);

    size_t node_count = node_counts[tree_idx];
    if (node_count == 0) return;

    int tree_offset = tree_idx * capacity;

    // Mutate each node with probability = mutation_rate
    for (size_t i = 0; i < node_count; i++) {
        if (curand_uniform(&state) < mutation_rate) {
            mutateNode(nodes, values, tree_offset + i, &state, numberOfVariables);
        }
    }
}

void Mutation::mutationGPU(GPUTree *population, size_t numberOfVariables = 0) {
    if (population == nullptr || population->population == 0) return;

    // Generate random seed
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    unsigned long seed = static_cast<unsigned long>(nanos);

    // Kernel configuration
    int threads_per_block = 256;
    int blocks = (population->population + threads_per_block - 1) / threads_per_block;

    // Launch mutation kernel
    mutationKernel<<<blocks, threads_per_block>>>(
        population->nodes,
        population->values,
        population->children,
        population->node_counts,
        population->capacity,
        population->population,
        mutationRate,
        numberOfVariables,
        seed
    );

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Mutation kernel error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Mutation kernel failed");
    }

    cudaDeviceSynchronize();
}