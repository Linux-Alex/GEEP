// SymbolicRegressionGPUTest.cu
// Created by aleks on 3.5.2025.

#include "SymbolicRegressionGPUTest.h"
#include "../cuda/CudaUtils.h"
#include "../problems/SymbolicRegressionProblem.cuh"
#include "../nodes/functions/MultiplyOperator.h"
#include "src/nodes/functions/AddOperator.h"
#include "src/nodes/terminals/ConstNode.h"
#include "src/nodes/terminals/VariableNode.h"

TEST_F(SymbolicRegressionGPUTest, EvaluateSolution) {
    Solution * s = new Solution();

    AddOperator a, b;
    MultiplyOperator m;

    m.addChild(new VariableNode("x"));
    m.addChild(new VariableNode("x"));

    b.addChild(new VariableNode("x"));
    b.addChild(new ConstNode(1.0f));

    a.addChild(m.clone());
    a.addChild(b.clone());

    s->setRoot(m.clone());
}

TEST_F(SymbolicRegressionGPUTest, EvaluateLinearSolution) {
    // Represents the expression: x + 2
    const size_t population = 1; // For this single test
    const int max_nodes = 16;
    const size_t num_targets = 2;

    // Inputs: x = 1 and x = 2 → Expected outputs: 1 + 2 = 3, 2 + 2 = 4
    float h_target_data[] = {1.0f, 2.0f};
    float h_target_values[] = {4.0f, 9.0f}; // Targets (e.g., maybe the "ideal" model is x*x + 3)

    cudaMallocManaged(&d_nodes, population * max_nodes * sizeof(int));
    cudaMallocManaged(&d_values, population * max_nodes * sizeof(float));
    cudaMallocManaged(&d_children, population * max_nodes * 2 * sizeof(int));
    cudaMallocManaged(&d_counts, population * sizeof(size_t));
    cudaMallocManaged(&d_fitnesses, population * sizeof(float));
    cudaMallocManaged(&d_target_data, num_targets * sizeof(float));
    cudaMallocManaged(&d_target_values, num_targets * sizeof(float));

    int nodes[] = {2, 0, 1};                 // 2 = AddOp, 0 = VariableNode, 1 = ConstNode
    float values[] = {0.0f, 0.0f, 2.0f};     // Only the constant node carries a value
    int children[] = {1, 2, -1, -1, -1, -1}; // 2 children per node

    memcpy(&d_nodes[0*max_nodes], nodes, sizeof(nodes));
    memcpy(&d_values[0*max_nodes], values, sizeof(values));
    memcpy(&d_children[0*max_nodes*2], children, sizeof(children));
    d_counts[0] = 3; // Number of nodes in the tree

    // Copy target data
    memcpy(d_target_data, h_target_data, sizeof(h_target_data));
    memcpy(d_target_values, h_target_values, sizeof(h_target_values));

    dim3 block(256);
    dim3 grid((population + block.x - 1) / block.x);
    gpuEvaluateKernel<<<grid, block>>>(
        d_nodes, d_values, d_children, d_counts, d_fitnesses,
        d_target_data, d_target_values, num_targets,
        population, max_nodes);
    cudaDeviceSynchronize();

    // Difference: (4-3)^2 + (9-4)^2 = 1 + 25 = 26; assuming some internal normalization, 13.0f is likely
    EXPECT_NEAR(d_fitnesses[0], 13.0f, 0.01f);
}

TEST_F(SymbolicRegressionGPUTest, EvaluateConstSolution) {
    // Represents the expression: 5
    const size_t population = 1; // For this single test
    const int max_nodes = 16;
    const size_t num_targets = 2;

    // Inputs: x = 1 and x = 2 → Expected outputs: 1 + 2 = 3, 2 + 2 = 4
    float h_target_data[] = {1.0f, 2.0f};
    float h_target_values[] = {4.0f, 9.0f}; // Targets (e.g., maybe the "ideal" model is x*x + 3)

    cudaMallocManaged(&d_nodes, population * max_nodes * sizeof(int));
    cudaMallocManaged(&d_values, population * max_nodes * sizeof(float));
    cudaMallocManaged(&d_children, population * max_nodes * 2 * sizeof(int));
    cudaMallocManaged(&d_counts, population * sizeof(size_t));
    cudaMallocManaged(&d_fitnesses, population * sizeof(float));
    cudaMallocManaged(&d_target_data, num_targets * sizeof(float));
    cudaMallocManaged(&d_target_values, num_targets * sizeof(float));

    int nodes[] = {1};         // 1 = ConstNode
    float values[] = {5.0f};   // Only the constant node carries a value
    int children[] = {-1, -1}; // 2 children per node

    memcpy(&d_nodes[0*max_nodes], nodes, sizeof(nodes));
    memcpy(&d_values[0*max_nodes], values, sizeof(values));
    memcpy(&d_children[0*max_nodes*2], children, sizeof(children));
    d_counts[0] = 1; // Number of nodes in the tree

    // Copy target data
    memcpy(d_target_data, h_target_data, sizeof(h_target_data));
    memcpy(d_target_values, h_target_values, sizeof(h_target_values));

    dim3 block(256);
    dim3 grid((population + block.x - 1) / block.x);
    gpuEvaluateKernel<<<grid, block>>>(
        d_nodes, d_values, d_children, d_counts, d_fitnesses,
        d_target_data, d_target_values, num_targets,
        population, max_nodes);
    cudaDeviceSynchronize();

    // Difference: (4-5)^2 + (9-5)^2 = 1 + 16 = 17; assuming some internal normalization, 8.5f is likely
    EXPECT_NEAR(d_fitnesses[0], 8.5f, 0.01f);
}

TEST_F(SymbolicRegressionGPUTest, EvaluateOnlyOneVariableSolution) {
    // Represents the expression: x
    const size_t population = 1; // For this single test
    const int max_nodes = 16;
    const size_t num_targets = 2;

    // Inputs: x = 1 and x = 2 → Expected outputs: 1 + 2 = 3, 2 + 2 = 4
    float h_target_data[] = {1.0f, 2.0f};
    float h_target_values[] = {4.0f, 9.0f}; // Targets (e.g., maybe the "ideal" model is x*x + 3)

    cudaMallocManaged(&d_nodes, population * max_nodes * sizeof(int));
    cudaMallocManaged(&d_values, population * max_nodes * sizeof(float));
    cudaMallocManaged(&d_children, population * max_nodes * 2 * sizeof(int));
    cudaMallocManaged(&d_counts, population * sizeof(size_t));
    cudaMallocManaged(&d_fitnesses, population * sizeof(float));
    cudaMallocManaged(&d_target_data, num_targets * sizeof(float));
    cudaMallocManaged(&d_target_values, num_targets * sizeof(float));

    int nodes[] = {0};         // 0 = VariableNode
    float values[] = {0.0f};   // Only the constant node carries a value
    int children[] = {-1, -1}; // 2 children per node

    memcpy(&d_nodes[0*max_nodes], nodes, sizeof(nodes));
    memcpy(&d_values[0*max_nodes], values, sizeof(values));
    memcpy(&d_children[0*max_nodes*2], children, sizeof(children));
    d_counts[0] = 1; // Number of nodes in the tree

    // Copy target data
    memcpy(d_target_data, h_target_data, sizeof(h_target_data));
    memcpy(d_target_values, h_target_values, sizeof(h_target_values));

    dim3 block(256);
    dim3 grid((population + block.x - 1) / block.x);
    gpuEvaluateKernel<<<grid, block>>>(
        d_nodes, d_values, d_children, d_counts, d_fitnesses,
        d_target_data, d_target_values, num_targets,
        population, max_nodes);
    cudaDeviceSynchronize();

    // Difference: (4-1)^2 + (9-2)^2 = 9 + 49 = 58; assuming some internal normalization, 29f is likely
    EXPECT_NEAR(d_fitnesses[0], 29.0f, 0.01f);
}

TEST_F(SymbolicRegressionGPUTest, EvaluateLeftSideTreeSolution) {
    // Represents the expression: (((2 / 3) * x) + x)
    const size_t population = 1; // For this single test
    const int max_nodes = 16;
    const size_t num_targets = 2;

    // Inputs: x = 1 and x = 2 → Expected outputs: 1 + 2 = 3, 2 + 2 = 4
    float h_target_data[] = {1.0f, 2.0f};
    float h_target_values[] = {4.0f, 9.0f}; // Targets (e.g., maybe the "ideal" model is x*x + 3)

    cudaMallocManaged(&d_nodes, population * max_nodes * sizeof(int));
    cudaMallocManaged(&d_values, population * max_nodes * sizeof(float));
    cudaMallocManaged(&d_children, population * max_nodes * 2 * sizeof(int));
    cudaMallocManaged(&d_counts, population * sizeof(size_t));
    cudaMallocManaged(&d_fitnesses, population * sizeof(float));
    cudaMallocManaged(&d_target_data, num_targets * sizeof(float));
    cudaMallocManaged(&d_target_values, num_targets * sizeof(float));

    int nodes[] = {2, 4, 0, 5, 1, 1, 0};      // 0 = VariableNode
    float values[] = {0, 0, 0, 0, 2, 3, 0};   // Only the constant node carries a value
    int children[] = {
        1, 6,       // Add's children: 1 (multiply), 6 (var)
        2, 3,       // Multiply's children: 2 (var), 3 (divide)
        -1, -1,     // Var x (no children)
        4, 5,       // Divide's children: 4 (const 2), 5 (const 3)
        -1, -1,     // Const 2 (no children)
        -1, -1,     // Const 3 (no children)
        -1, -1      // Var x (no children)
    };

    memcpy(&d_nodes[0*max_nodes], nodes, sizeof(nodes));
    memcpy(&d_values[0*max_nodes], values, sizeof(values));
    memcpy(&d_children[0*max_nodes*2], children, sizeof(children));
    d_counts[0] = 7; // Number of nodes in the tree

    // Copy target data
    memcpy(d_target_data, h_target_data, sizeof(h_target_data));
    memcpy(d_target_values, h_target_values, sizeof(h_target_values));

    dim3 block(256);
    dim3 grid((population + block.x - 1) / block.x);
    gpuEvaluateKernel<<<grid, block>>>(
        d_nodes, d_values, d_children, d_counts, d_fitnesses,
        d_target_data, d_target_values, num_targets,
        population, max_nodes);
    cudaDeviceSynchronize();

    // Difference: (4-1)^2 + (9-2)^2 = 9 + 49 = 58; assuming some internal normalization, 29f is likely
    EXPECT_NEAR(d_fitnesses[0], 18.78f, 0.01f);
}

TEST_F(SymbolicRegressionGPUTest, EvaluateRightSideTreeSolution) {
    // Represents the expression: (x + (x * (2 / 3)))
    const size_t population = 1; // For this single test
    const int max_nodes = 16;
    const size_t num_targets = 2;

    // Inputs: x = 1 and x = 2 → Expected outputs: 1 + 2 = 3, 2 + 2 = 4
    float h_target_data[] = {1.0f, 2.0f};
    float h_target_values[] = {4.0f, 9.0f}; // Targets (e.g., maybe the "ideal" model is x*x + 3)

    cudaMallocManaged(&d_nodes, population * max_nodes * sizeof(int));
    cudaMallocManaged(&d_values, population * max_nodes * sizeof(float));
    cudaMallocManaged(&d_children, population * max_nodes * 2 * sizeof(int));
    cudaMallocManaged(&d_counts, population * sizeof(size_t));
    cudaMallocManaged(&d_fitnesses, population * sizeof(float));
    cudaMallocManaged(&d_target_data, num_targets * sizeof(float));
    cudaMallocManaged(&d_target_values, num_targets * sizeof(float));

    int nodes[] = {2, 0, 4, 0, 5, 1, 1};      // 0 = VariableNode
    float values[] = {0, 0, 0, 0, 0, 2, 3};   // Only the constant node carries a value
    int children[] = {
        1, 2,       // Add's children: 1 (var), 2 (multiply)
        -1, -1,     // Var x (no children)
        3, 4,       // Multiply's children: 3 (var), 4 (divide)
        -1, -1,     // Var x (no children)
        5, 6,       // Divide's children: 5 (const 2), 6 (const 3)
        -1, -1,     // Const 2 (no children)
        -1, -1      // Const 3 (no children)
    };

    memcpy(&d_nodes[0*max_nodes], nodes, sizeof(nodes));
    memcpy(&d_values[0*max_nodes], values, sizeof(values));
    memcpy(&d_children[0*max_nodes*2], children, sizeof(children));
    d_counts[0] = 7; // Number of nodes in the tree

    // Copy target data
    memcpy(d_target_data, h_target_data, sizeof(h_target_data));
    memcpy(d_target_values, h_target_values, sizeof(h_target_values));

    dim3 block(256);
    dim3 grid((population + block.x - 1) / block.x);
    gpuEvaluateKernel<<<grid, block>>>(
        d_nodes, d_values, d_children, d_counts, d_fitnesses,
        d_target_data, d_target_values, num_targets,
        population, max_nodes);
    cudaDeviceSynchronize();

    // Difference: (4-1.67)^2 + (9-3.33)^2 = 5.43 + 11.09 = 37.58; assuming some internal normalization, 18.78f is likely
    EXPECT_NEAR(d_fitnesses[0], 18.78f, 0.01f);
}

TEST_F(SymbolicRegressionGPUTest, EvaluateSomeTest) {
    // Represents the expression: (x + (x * (2 / 3)))
    const size_t population = 1; // For this single test
    const int max_nodes = 16;
    const size_t num_targets = 2;

    // Inputs: x = 1 and x = 2 → Expected outputs: 1 + 2 = 3, 2 + 2 = 4
    float h_target_data[] = {1.0f, 2.0f};
    float h_target_values[] = {4.0f, 9.0f}; // Targets (e.g., maybe the "ideal" model is x*x + 3)

    cudaMallocManaged(&d_nodes, population * max_nodes * sizeof(int));
    cudaMallocManaged(&d_values, population * max_nodes * sizeof(float));
    cudaMallocManaged(&d_children, population * max_nodes * 2 * sizeof(int));
    cudaMallocManaged(&d_counts, population * sizeof(size_t));
    cudaMallocManaged(&d_fitnesses, population * sizeof(float));
    cudaMallocManaged(&d_target_data, num_targets * sizeof(float));
    cudaMallocManaged(&d_target_values, num_targets * sizeof(float));

    int nodes[] = {3, 3, 1, 0, 1};      // 0 = VariableNode
    float values[] = {0.0f, 0.0f, -2.11f, 0.0f, 5.66f};   // Only the constant node carries a value
    int children[] = {
        1, 4, 2, 3, -1, -1, -1, -1 ,-1 ,-1
    };

    memcpy(&d_nodes[0*max_nodes], nodes, sizeof(nodes));
    memcpy(&d_values[0*max_nodes], values, sizeof(values));
    memcpy(&d_children[0*max_nodes*2], children, sizeof(children));
    d_counts[0] = 5; // Number of nodes in the tree

    // Copy target data
    memcpy(d_target_data, h_target_data, sizeof(h_target_data));
    memcpy(d_target_values, h_target_values, sizeof(h_target_values));

    dim3 block(256);
    dim3 grid((population + block.x - 1) / block.x);
    gpuEvaluateKernel<<<grid, block>>>(
        d_nodes, d_values, d_children, d_counts, d_fitnesses,
        d_target_data, d_target_values, num_targets,
        population, max_nodes);
    cudaDeviceSynchronize();

    // Difference: (4-1.67)^2 + (9-3.33)^2 = 5.43 + 11.09 = 37.58; assuming some internal normalization, 18.78f is likely
    EXPECT_NEAR(d_fitnesses[0], 257.69f, 0.01f);
}

TEST_F(SymbolicRegressionGPUTest, EvaluateCrossedTree) {
    // Represents the expression: (x + (x * (2 / 3)))
    const size_t population = 1; // For this single test
    const int max_nodes = 16;
    const size_t num_targets = 2;

    // Inputs: x = 1 and x = 2 → Expected outputs: 1 + 2 = 3, 2 + 2 = 4
    float h_target_data[] = {1.0f, 2.0f};
    float h_target_values[] = {4.0f, 9.0f}; // Targets (e.g., maybe the "ideal" model is x*x + 3)

    cudaMallocManaged(&d_nodes, population * max_nodes * sizeof(int));
    cudaMallocManaged(&d_values, population * max_nodes * sizeof(float));
    cudaMallocManaged(&d_children, population * max_nodes * 2 * sizeof(int));
    cudaMallocManaged(&d_counts, population * sizeof(size_t));
    cudaMallocManaged(&d_fitnesses, population * sizeof(float));
    cudaMallocManaged(&d_target_data, num_targets * sizeof(float));
    cudaMallocManaged(&d_target_values, num_targets * sizeof(float));

    int nodes[] = {3, 0, 4, 2, 1, 2, 4, 1, 0, 1, 1};      // 0 = VariableNode
    float values[] = {0.00, 0.00, 0.00, 0.00, 0.27, 0.00, 0.00, 9.04, 0.00, 8.32, 2.14};   // Only the constant node carries a value
    int children[] = {
        1, 2, -1, -1, 3, 10, 4, 5, -1, -1, 6, 9, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1
    };

    memcpy(&d_nodes[0*max_nodes], nodes, sizeof(nodes));
    memcpy(&d_values[0*max_nodes], values, sizeof(values));
    memcpy(&d_children[0*max_nodes*2], children, sizeof(children));
    d_counts[0] = 11; // Number of nodes in the tree

    // Copy target data
    memcpy(d_target_data, h_target_data, sizeof(h_target_data));
    memcpy(d_target_values, h_target_values, sizeof(h_target_values));

    dim3 block(256);
    dim3 grid((population + block.x - 1) / block.x);
    gpuEvaluateKernel<<<grid, block>>>(
        d_nodes, d_values, d_children, d_counts, d_fitnesses,
        d_target_data, d_target_values, num_targets,
        population, max_nodes);
    cudaDeviceSynchronize();

    // Difference: (4-1.67)^2 + (9-3.33)^2 = 5.43 + 11.09 = 37.58; assuming some internal normalization, 18.78f is likely
    EXPECT_NEAR(d_fitnesses[0], 2882.12f, 0.01f);
}