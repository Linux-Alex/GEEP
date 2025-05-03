// SymbolicRegressionGPUTest.h
// Created by aleks on 3.5.2025.

#ifndef SYMBOLICREGRESSIONGPUTEST_H
#define SYMBOLICREGRESSIONGPUTEST_H

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstring>

class SymbolicRegressionGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        num_targets = 2;
        population = 5;
        max_nodes = 16;

        cudaMallocManaged(&d_nodes, population * max_nodes * sizeof(int));
        cudaMallocManaged(&d_values, population * max_nodes * sizeof(float));
        cudaMallocManaged(&d_children, population * max_nodes * 2 * sizeof(int));
        cudaMallocManaged(&d_counts, population * sizeof(size_t));
        cudaMallocManaged(&d_fitnesses, population * sizeof(float));
    }

    void TearDown() override {
        cudaFree(d_target_data);
        cudaFree(d_target_values);
        cudaFree(d_nodes);
        cudaFree(d_values);
        cudaFree(d_children);
        cudaFree(d_counts);
        cudaFree(d_fitnesses);
    }

    size_t num_targets, population;
    int max_nodes;
    float *d_target_data, *d_target_values;
    int *d_nodes;
    float *d_values;
    int *d_children;
    size_t *d_counts;
    float *d_fitnesses;
};

#endif // SYMBOLICREGRESSIONGPUTEST_H
