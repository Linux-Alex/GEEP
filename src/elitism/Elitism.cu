//
// Created by aleks on 30. 11. 25.
//

#include "Elitism.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <iostream>

__global__ void copyIndividualKernel(
    const int* src_nodes, const float* src_values, const int* src_children, const size_t* src_counts,
    int* dst_nodes, float* dst_values, int* dst_children, size_t* dst_counts,
    size_t capacity, int src_index, int dst_index
) {
    // This kernel copies one individual from source to destination
    size_t node_count = src_counts[src_index];
    dst_counts[dst_index] = node_count;

    int src_offset = src_index * capacity;
    int dst_offset = dst_index * capacity;

    for (size_t i = 0; i < node_count && i < capacity; i++) {
        dst_nodes[dst_offset + i] = src_nodes[src_offset + i];
        dst_values[dst_offset + i] = src_values[src_offset + i];
        dst_children[(dst_offset + i) * 2] = src_children[(src_offset + i) * 2];
        dst_children[(dst_offset + i) * 2 + 1] = src_children[(src_offset + i) * 2 + 1];
    }
}

void Elitism::applyElitismGPU(GPUTree* old_population, GPUTree* new_population) {
    if (old_population == nullptr || new_population == nullptr) return;
    if (elitismCount == 0) return;

    size_t population_size = old_population->population;
    elitismCount = std::min(elitismCount, population_size);

    // Use Thrust to sort indices by fitness (ascending - lower MSE is better)
    thrust::device_ptr<float> fitness_ptr(old_population->fitness_values);
    thrust::device_ptr<int> indices(old_population->selection_parent1_idx); // Reuse this array for indices

    // Create sequence of indices [0, 1, 2, ..., population_size-1]
    thrust::sequence(indices, indices + population_size);

    // Sort indices based on fitness (stable sort to preserve order for equal fitness)
    thrust::stable_sort_by_key(fitness_ptr, fitness_ptr + population_size, indices);

    // Copy the best 'elitismCount' individuals from old to new population
    for (size_t i = 0; i < elitismCount; i++) {
        int best_index = 0;
        cudaMemcpy(&best_index, thrust::raw_pointer_cast(indices + i), sizeof(int), cudaMemcpyDeviceToHost);

        // Launch kernel to copy this individual
        copyIndividualKernel<<<1, 1>>>(
            old_population->nodes, old_population->values, old_population->children, old_population->node_counts,
            new_population->nodes, new_population->values, new_population->children, new_population->node_counts,
            old_population->capacity, best_index, i
        );

        cudaDeviceSynchronize();
    }

    // Debug output
    // std::cout << "Elitism: Preserved " << elitismCount << " best individuals" << std::endl;
}