#include "SymbolicRegressionProblem.cuh"

#include <c++/13/cfloat>

SymbolicRegressionProblem::~SymbolicRegressionProblem() {
    // Free GPU memory
    if (flattened_targets.size() > 0) {
        cudaFree(flattened_targets.data());
        cudaFree(target_values.data());
    }
}

__device__ float evaluateTreeWithX(const int* nodes, const float* values,
                                 const int* children, size_t node_count,
                                 float x_value) {
    // Temporary storage for computed values
    float temp_values[32];  // Adjust size based on max tree depth
    bool computed[32] = {false};  // Track which nodes have been computed

    // Process nodes from last to first
    for (int i = node_count - 1; i >= 0; i--) {
        // Skip if already computed
        if (computed[i]) continue;

        // Check if this is a terminal node (no children)
        if (children[i*2] == -1 && children[i*2+1] == -1) {
            switch (nodes[i]) {
                case 0:  // Variable node
                    temp_values[i] = x_value;
                    break;
                case 1:  // Constant node
                    temp_values[i] = values[i];
                    break;
                default:
                    temp_values[i] = NAN;  // Invalid terminal
            }
            computed[i] = true;
        }
        else {
            // Check if children are computed
            bool left_ready = (children[i*2] == -1) || computed[children[i*2]];
            bool right_ready = (children[i*2+1] == -1) || computed[children[i*2+1]];

            if (left_ready && right_ready) {
                // Get left value (either from temp or original)
                float left_val = (children[i*2] == -1) ?
                    ((nodes[i*2] == 0) ? x_value : values[i*2]) :
                    temp_values[children[i*2]];

                // Get right value (either from temp or original)
                float right_val = (children[i*2+1] == -1) ?
                    ((nodes[i*2+1] == 0) ? x_value : values[i*2+1]) :
                    temp_values[children[i*2+1]];

                // Compute operation
                switch (nodes[i]) {
                    case 2:  // Add
                        temp_values[i] = left_val + right_val;
                        break;
                    case 3:  // Subtract
                        temp_values[i] = left_val - right_val;
                        break;
                    case 4:  // Multiply
                        temp_values[i] = left_val * right_val;
                        break;
                    case 5:  // Divide
                        temp_values[i] = (right_val != 0.0f) ?
                                         left_val / right_val : FLT_MAX;
                        break;
                    default:
                        temp_values[i] = NAN;  // Invalid operator
                }
                computed[i] = true;
            }
        }
    }

    return temp_values[0];  // Root node value
}

__global__ void gpuEvaluateKernel(
    const int* nodes, const float* values, const int* children,
    const size_t* counts, float* fitnesses,
    const float* target_data, const float* target_values,
    size_t num_targets, size_t population, int max_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population) return;

    int offset = idx * max_nodes;
    size_t node_count = counts[idx];

    // Simple validation
    if (node_count == 0) {
        fitnesses[idx] = FLT_MAX;
        return;
    }

    // Print target_data content
    // if (idx == 0) {
    //     printf("Target data: ");
    //     for (size_t i = 0; i < num_targets; i++) {
    //         printf("%.2f ", target_data[i]);
    //     }
    //     printf("\n");
    //
    //     printf("Target values: ");
    //     for (size_t i = 0; i < num_targets; i++) {
    //         printf("%.2f ", target_values[i]);
    //     }
    //     printf("\n");
    // }

    float total_error = 0.0f;
    for (size_t i = 0; i < num_targets; i++) {
        float x = target_data[i];
        // printf("X: %.2f\n", x);
        float predicted = evaluateTreeWithX(
            nodes + offset,
            values + offset,
            children + (offset * 2),
            node_count,
            x);
        // Print debug info
        // printf("X: %.2f (after evaluateTreeWithX)\n", x);
        // printf("Target : x=%.2f => pred=%.2f (expected=%.2f)\n", x, predicted, target_values[i]);
        float error = predicted - target_values[i];
        total_error += error * error;
        // printf("Error: %.2f\n", error);
    }

    fitnesses[idx] = total_error / num_targets;
}

void SymbolicRegressionProblem::gpuEvaluate(GPUTree &trees, float *fitnesses) {
    printf("\n=== Starting GPU Evaluation ===\n");

    // 1. Validate inputs
    if (!trees.nodes || !trees.values || !trees.children || !trees.node_counts) {
        printf("ERROR: GPUTree pointers not initialized!\n");
        return;
    }

    if (!this->getTargetData() || !this->getTargetValues()) {
        printf("ERROR: Target data not prepared! Call prepareTargetData() first.\n");
        return;
    }

    // 2. Print debug info
    printf("Population size: %zu\n", trees.population);
    printf("Max nodes per tree: %zu\n", this->getMaxNodes());
    printf("Number of targets: %zu\n", this->getNumTargets());

    // 3. Configure kernel launch
    dim3 block(256);
    dim3 grid((trees.population + block.x - 1) / block.x);
    printf("Launching kernel with %d blocks, %d threads\n", grid.x, block.x);

    // 4. Enable GPU printf
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024);

    // 5. Create debug buffer
    float* debug_output;
    cudaMallocManaged(&debug_output, 5 * sizeof(float));

    // 6. Launch kernel with error checking
    printf("Launching kernel...\n");
    // Launch kernel with correct number of parameters
    gpuEvaluateKernel<<<grid, block>>>(
        trees.nodes,
        trees.values,
        trees.children,
        trees.node_counts,
        fitnesses,
        this->getTargetData(),
        this->getTargetValues(),
        this->getNumTargets(),
        trees.population,
        this->getMaxNodes());

    // 7. Check for launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        cudaFree(debug_output);
        return;
    }

    // 8. Synchronize and check execution
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(syncErr));

        // Print debug info if available
        printf("Debug output: ");
        for (int i = 0; i < 5; i++) {
            printf("%.2f ", debug_output[i]);
        }
        printf("\n");

        cudaFree(debug_output);
        return;
    }

    // 9. Print results
    printf("Evaluation completed successfully.\n");
    printf("Sample fitness values:\n");
    for (size_t i = 0; i < min(5ul, trees.population); i++) {
        printf("  Solution %zu: %.2f\n", i, fitnesses[i]);
    }

    // 10. Cleanup
    cudaFree(debug_output);
    printf("=== GPU Evaluation Complete ===\n\n");
}

void SymbolicRegressionProblem::testGPUComputing() {
    printf("\n=== STARTING PROPER GPU VALIDATION TEST ===\n");

    // 1. Test Configuration
    const size_t num_targets = 2;
    const size_t population = 5;
    const int max_nodes = 16;

    // 2. Test Targets (x=1.0->4.0, x=2.0->9.0)
    float h_target_data[] = {1.0f, 2.0f};
    float h_target_values[] = {4.0f, 9.0f};

    // 3. Allocate Unified Memory for everything
    float *d_target_data, *d_target_values;
    int *d_nodes;
    float *d_values;
    int *d_children;
    size_t *d_counts;
    float *d_fitnesses;

    cudaMallocManaged(&d_target_data, num_targets * sizeof(float));
    cudaMallocManaged(&d_target_values, num_targets * sizeof(float));
    cudaMallocManaged(&d_nodes, population * max_nodes * sizeof(int));
    cudaMallocManaged(&d_values, population * max_nodes * sizeof(float));
    cudaMallocManaged(&d_children, population * max_nodes * 2 * sizeof(int));
    cudaMallocManaged(&d_counts, population * sizeof(size_t));
    cudaMallocManaged(&d_fitnesses, population * sizeof(float));

    // 4. Initialize Test Solutions
    // Solution 0: x + 2
    int tree0_nodes[] = {2, 0, 1}; // Add, Var, Const
    float tree0_values[] = {0, 0, 2};
    int tree0_children[] = {1, 2, -1, -1, -1, -1}; // 2 children per node

    // Solution 1: x
    int tree1_nodes[] = {0}; // Just variable
    float tree1_values[] = {0};
    int tree1_children[] = {-1, -1};

    // Solution 2: 5 (constant)
    int tree2_nodes[] = {1}; // Just constant
    float tree2_values[] = {5};
    int tree2_children[] = {-1, -1};

    // Solution 3: (x + (x * (2 / 3)))
    int tree3_nodes[] = {2, 0, 4, 0, 5, 1, 1}; // Add, Var, Multiply, Var, Divide, Const, Const
    float tree3_values[] = {0, 0, 0, 0, 0, 2, 3}; // Only constants have values
    int tree3_children[] = {
        1, 2,       // Add's children: 1 (var), 2 (multiply)
        -1, -1,     // Var x (no children)
        3, 4,       // Multiply's children: 3 (var), 4 (divide)
        -1, -1,     // Var x (no children)
        5, 6,       // Divide's children: 5 (const 2), 6 (const 3)
        -1, -1,     // Const 2 (no children)
        -1, -1      // Const 3 (no children)
    };

    // Solution 4: (((2 / 3) * x) + x)
    int tree4_nodes[] = {2, 4, 0, 5, 1, 1, 0}; // Add, Multiply, Var, Divide, Const, Const, Var
    float tree4_values[] = {0, 0, 0, 0, 2, 3, 0}; // Only constants have values
    int tree4_children[] = {
        1, 6,       // Add's children: 1 (multiply), 6 (var)
        2, 3,       // Multiply's children: 2 (var), 3 (divide)
        -1, -1,     // Var x (no children)
        4, 5,       // Divide's children: 4 (const 2), 5 (const 3)
        -1, -1,     // Const 2 (no children)
        -1, -1,     // Const 3 (no children)
        -1, -1      // Var x (no children)
    };

    // Copy to unified memory
    // Solution 0
    memcpy(&d_nodes[0*max_nodes], tree0_nodes, sizeof(tree0_nodes));
    memcpy(&d_values[0*max_nodes], tree0_values, sizeof(tree0_values));
    memcpy(&d_children[0*max_nodes*2], tree0_children, sizeof(tree0_children));
    d_counts[0] = 3;

    // Solution 1
    memcpy(&d_nodes[1*max_nodes], tree1_nodes, sizeof(tree1_nodes));
    memcpy(&d_values[1*max_nodes], tree1_values, sizeof(tree1_values));
    memcpy(&d_children[1*max_nodes*2], tree1_children, sizeof(tree1_children));
    d_counts[1] = 1;

    // Solution 2
    memcpy(&d_nodes[2*max_nodes], tree2_nodes, sizeof(tree2_nodes));
    memcpy(&d_values[2*max_nodes], tree2_values, sizeof(tree2_values));
    memcpy(&d_children[2*max_nodes*2], tree2_children, sizeof(tree2_children));
    d_counts[2] = 1;

    // Add to your existing initialization code:

    // Solution 3
    memcpy(&d_nodes[3*max_nodes], tree3_nodes, sizeof(tree3_nodes));
    memcpy(&d_values[3*max_nodes], tree3_values, sizeof(tree3_values));
    memcpy(&d_children[3*max_nodes*2], tree3_children, sizeof(tree3_children));
    d_counts[3] = 7;

    // Solution 4
    memcpy(&d_nodes[4*max_nodes], tree4_nodes, sizeof(tree4_nodes));
    memcpy(&d_values[4*max_nodes], tree4_values, sizeof(tree4_values));
    memcpy(&d_children[4*max_nodes*2], tree4_children, sizeof(tree4_children));
    d_counts[4] = 7;

    // Copy target data
    memcpy(d_target_data, h_target_data, sizeof(h_target_data));
    memcpy(d_target_values, h_target_values, sizeof(h_target_values));

    // 5. Launch Kernel
    dim3 block(256);
    dim3 grid((population + block.x - 1) / block.x);

    gpuEvaluateKernel<<<grid, block>>>(
        d_nodes, d_values, d_children, d_counts, d_fitnesses,
        d_target_data, d_target_values, num_targets,
        population, max_nodes);

    cudaDeviceSynchronize();

    // 6. Print Results
    printf("\nValidation Results:\n");
    printf("Solution 0 (x + 2): MSE = %.2f (expected 13.00)\n", d_fitnesses[0]);
    printf("Solution 1 (x):     MSE = %.2f (expected 29.00)\n", d_fitnesses[1]);
    printf("Solution 2 (5):     MSE = %.2f (expected 8.50)\n", d_fitnesses[2]);
    printf("Solution 3 (x + (x * (2 / 3))): MSE = %.2f (expected 18.78)\n", d_fitnesses[3]);
    printf("Solution 4 (((2 / 3) * x) + x): MSE = %.2f (expected 18.78)\n", d_fitnesses[4]);

    // 7. Cleanup
    cudaFree(d_target_data);
    cudaFree(d_target_values);
    cudaFree(d_nodes);
    cudaFree(d_values);
    cudaFree(d_children);
    cudaFree(d_counts);
    cudaFree(d_fitnesses);
}

void SymbolicRegressionProblem::prepareTargetData() {
    // Clear existing data
    flattened_targets.clear();
    target_values.clear();

    // Find all unique variable names
    std::set<std::string> unique_vars;
    for (const auto& target : targets) {
        for (const auto& [var_name, _] : target.getState()) {
            unique_vars.insert(var_name);
        }
    }
    this->num_variables = unique_vars.size();

    // Create mapping of variable name to index
    std::map<std::string, size_t> var_indices;
    size_t index = 0;
    for (const auto& var_name : unique_vars) {
        var_indices[var_name] = index++;
    }

    // Flatten targets into GPU-friendly format
    for (const auto& target : targets) {
        // Initialize all variables to 0
        std::vector<float> vars(num_variables, 0.0f);

        // Set values for variables present in this targetmake
        for (const auto& [var_name, value] : target.getState()) {
            vars[var_indices[var_name]] = static_cast<float>(value);
        }

        // Add to flattened data
        flattened_targets.insert(flattened_targets.end(), vars.begin(), vars.end());
        target_values.push_back(static_cast<float>(target.getTargetValue()));
    }

    // Debug: Print flattened targets
    printf("Flattened targets (x values): ");
    for (float x : flattened_targets) {
        printf("%.2f ", x);
    }
    printf("\nTarget values (y expected): ");
    for (float y : target_values) {
        printf("%.2f ", y);
    }
    printf("\n");
}