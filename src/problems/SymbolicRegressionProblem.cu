#include "SymbolicRegressionProblem.h"

__device__ float evaluateTreeWithX(const int* nodes, const float* values, const int* children, size_t node_count, float x_value) {
    // Stack-based evaluation for GPU efficiency
    float stack[64]; // Adjust size based on your max tree depth
    int stack_ptr = -1;

    for (size_t i = 0; i < node_count; i++) {
        int node_type = nodes[i];

        if (node_type == 0) { // Variable node (assuming 0 = variable)
            stack[++stack_ptr] = x_value;
        }
        else if (node_type == 1) { // Constant node (assuming 1 = constant)
            stack[++stack_ptr] = values[i];
        }
        else { // Operator node
            float right = stack[stack_ptr--];
            float left = stack[stack_ptr--];

            switch (node_type) {
                case 2: // Addition
                    stack[++stack_ptr] = left + right;
                    break;
                case 3: // Subtraction
                    stack[++stack_ptr] = left - right;
                    break;
                case 4: // Multiplication
                    stack[++stack_ptr] = left * right;
                    break;
                case 5: // Division
                    stack[++stack_ptr] = right != 0 ? left / right : 0; // Avoid division by zero
                    break;
                default:
                    stack[++stack_ptr] = 0; // Unknown operation, push 0
            }
        }
    }

    return stack[0]; // Final result
}

__device__ float evaluateSingleTree(const int* nodes, const float* values, const int* children, size_t node_count, const float* target_data, const float* target_values, size_t num_targets) {
    // TODO: GPU implementation of tree evaluation
    float total_error = 0.0f;
    for (size_t i = 0; i < num_targets; i++) {
        float x = target_data[i]; // Get input value for target i
        float predicted = evaluateTreeWithX(nodes, values, children, node_count, x);
        float expected = target_values[i];
        total_error += (predicted - expected) * (predicted - expected);
    }

    return total_error;
}

__global__ void gpuEvaluateKernel(const int* nodes, const float* values, const int* children, const size_t* counts, float* fitnesses, const float* target_data, const float* target_values, size_t num_targets, size_t population, int max_nodes_per_tree) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population) {
        int offset = idx * max_nodes_per_tree;
        fitnesses[idx] = evaluateSingleTree(&nodes[offset], &values[offset], &children[offset * 2], counts[idx], target_data, target_values, num_targets);
    }
}

void SymbolicRegressionProblem::gpuEvaluate(GPUTree &trees, float *fitnesses) {
    dim3 block(256);
    dim3 grid((trees.population + block.x - 1) / block.x);

    gpuEvaluateKernel<<<grid, block>>>(
        trees.nodes, trees.values, trees.children, trees.node_counts,
        fitnesses,
        this->getTargetData(), this->getTargetValues(), this->getNumTargets(),
        trees.population, this->getMaxNodes());

    cudaDeviceSynchronize();
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

        // Set values for variables present in this target
        for (const auto& [var_name, value] : target.getState()) {
            vars[var_indices[var_name]] = static_cast<float>(value);
        }

        // Add to flattened data
        flattened_targets.insert(flattened_targets.end(), vars.begin(), vars.end());
        target_values.push_back(static_cast<float>(target.getTargetValue()));
    }
}