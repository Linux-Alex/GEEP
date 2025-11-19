#include "TournamentSelection.h"
#include <curand_kernel.h>

__global__ void tournamentSelectionKernel(
    int* parent1_idx, int* parent2_idx,     // output arrays
    float* fitnesses,                       // fitness data
    size_t population_size,                 // total trees
    float reproduction_rate,                // reproduction rate
    int tournament_size,                    // size of the tournament
    unsigned long seed                      // random seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= population_size) return; // Out of bounds

    // Initialize random state
    curandState state;
    curand_init(seed + i, 0, 0, &state);

    // Tournament for parent 1
    int best_idx = -1;
    float best_fitness = INFINITY;

    for (int j = 0; j < tournament_size; j++) {
        int candidate = curand(&state) % population_size;
        float fitness = fitnesses[candidate];

        if (fitness < best_fitness) {
            best_fitness = fitness;
            best_idx = candidate;
        }
    }

    parent1_idx[i] = best_idx;

    // Tournament for parent 2
    if (curand_uniform(&state) <= reproduction_rate) {
        parent2_idx[i] = -1; // No crossover (reproduction only)
    }
    else {
        // Run another tournament for parent 2
        best_idx = -1;
        best_fitness = INFINITY;

        for (int j = 0; j < tournament_size; j++) {
            int candidate = curand(&state) % population_size;
            float fitness = fitnesses[candidate];

            if (fitness < best_fitness) {
                best_fitness = fitness;
                best_idx = candidate;
            }
        }

        parent2_idx[i] = best_idx;
    }
}

void TournamentSelection::getSelectedParentsForCrossoverGPU(GPUTree *population, float reproduction_rate) {
    size_t pop_size = population->population;

    // Allocate device memory for parrent indices (if not already)
    if (!population->selection_parent1_idx) {
        cudaMalloc(&population->selection_parent1_idx, pop_size * sizeof(int));
        cudaMalloc(&population->selection_parent2_idx, pop_size * sizeof(int));
    }

    // Launch kernel
    // unsigned long seed = 22;

    // Random seed
    unsigned long seed = rand() % 100000 + time(NULL);
    int threads_per_block = 256;
    int blocks = (pop_size + threads_per_block - 1) / threads_per_block;

    tournamentSelectionKernel<<<blocks, threads_per_block>>>(
        population->selection_parent1_idx,
        population->selection_parent2_idx,
        population->fitness_values,
        pop_size,
        reproduction_rate,
        this->tournamentSize,
        seed
    );
    cudaDeviceSynchronize(); // Synchronize device

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
}