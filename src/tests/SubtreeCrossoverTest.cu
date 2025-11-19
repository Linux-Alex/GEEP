//
// Created by aleks on 24.5.2025.
//

#include "SubtreeCrossoverTest.h"
#include <gtest/gtest.h>
#include "src/cuda/GPUTree.h"
#include "src/crossover/SubtreeCrossover.h"
#include <iostream>
#include "src/nodes/functions/AddOperator.h"
#include "src/nodes/functions/MultiplyOperator.h"
#include "src/nodes/terminals/ConstNode.h"
#include "src/nodes/terminals/VariableNode.h"


void printTree(const GPUTree& population, size_t tree_idx) {
    size_t offset = tree_idx * population.capacity;
    std::cout << "Tree " << tree_idx << " structure:\n";
    for (size_t i = 0; i < population.node_counts[tree_idx]; i++) {
        std::cout << "Node " << i << ": type=" << population.nodes[offset+i]
                  << ", parent=" << population.parent_indices[offset+i]
                  << ", children=[" << population.children[(offset+i)*2]
                  << "," << population.children[(offset+i)*2+1] << "]"
                  << ", value=" << population.values[offset+i] << "\n";

    }
}

TEST_F(SubtreeCrossoverTest, ValidateGPUCrossover) {
    // Create a test population with 2 trees
    GPUTree population;
    const size_t capacity = 10;
    const size_t population_size = 2;
    population.allocate(capacity, population_size);

    // Tree 1: (x + 5) + y
    // Nodes: [Add, Add, x, 5, y]
    population.nodes[0] = 2;  // Add (root)
    population.nodes[1] = 2;  // Add
    population.nodes[2] = 0;  // x
    population.values[2] = 1.0f; // Variable x
    population.nodes[3] = 1;  // 5
    population.values[3] = 5.0f; // Constant value
    population.nodes[4] = 0;  // y
    population.values[4] = 2.0f; // Variable y
    population.node_counts[0] = 5;

    // Allocate default children to -1
    for (size_t i = 0; i < capacity * population_size * 2; i++) {
        population.children[i] = -1;
    }

    // Children for Tree 1
    population.children[0] = 1;  // Root's left child (Add)
    population.children[1] = 4;  // Root's right child (y)
    population.children[2] = 2;   // Add's left child (x)
    population.children[3] = 3;   // Add's right child (5)

    // Tree 2: (8 * z)
    // Nodes: [Multiply, 8, z]
    population.nodes[capacity] = 4;  // Multiply (root)
    population.nodes[capacity+1] = 1; // 8
    population.nodes[capacity+2] = 0; // z
    population.values[capacity+2] = 3.0f; // Variable z
    population.values[capacity+1] = 8.0f;
    population.node_counts[1] = 3;

    // Children for Tree 2
    population.children[capacity*2] = 1;   // Multiply's left child (8)
    population.children[capacity*2+1] = 2; // Multiply's right child (z)

    // Parent indices for both trees
    population.parent_indices[0] = -1; // Root has no parent
    population.parent_indices[1] = 0;
    population.parent_indices[4] = 0;
    population.parent_indices[2] = 1;
    population.parent_indices[3] = 1;
    population.parent_indices[capacity] = -1; // Multiply is root
    population.parent_indices[capacity+1] = capacity;
    population.parent_indices[capacity+2] = capacity;

    // Add this verification before crossover
    ASSERT_EQ(population.parent_indices[1], 0); // Add node's parent is root
    ASSERT_EQ(population.parent_indices[capacity], -1); // Multiply is root

    // Print the population for debugging
    std::cout << "Initial population:" << std::endl;
    printTree(population, 0);
    printTree(population, 1);

    // Create new population
    GPUTree new_population;
    new_population.allocate(capacity, population_size);

    // Select parents for crossover
    *population.selection_parent1_idx = 0;
    *population.selection_parent2_idx = 1;

    // Perform crossover - swap the first Add (node 1) with Multiply (node capacity)
    SubtreeCrossover crossover;
    crossover.crossoverGPU(&population, &new_population);

    // Set node counts for new population
    new_population.node_counts[0] = 5; // New tree 1 has
    new_population.node_counts[1] = 5; // New tree 2 has

    // Print the population for debugging
    std::cout << "Population after crossover:" << std::endl;
    printTree(new_population, 0);
    printTree(new_population, 1);

    // Verify Tree 1 is now: (Multiply) + y
    EXPECT_EQ(new_population.nodes[0], 2);  // Root still Add
    EXPECT_EQ(new_population.nodes[1], 4);  // Now Multiply
    EXPECT_EQ(new_population.nodes[2], 1);  // value 8
    EXPECT_EQ(new_population.nodes[3], 0);  // variable z
    EXPECT_EQ(new_population.nodes[4], 0);  // y remains

    // Verify Tree 1's children
    EXPECT_EQ(new_population.children[0], 1);  // Root's left child now Multiply
    EXPECT_EQ(new_population.children[1], 4);  // Root's right child still y
    EXPECT_EQ(new_population.children[2], 2); // Multiply's left child (8)
    EXPECT_EQ(new_population.children[3], 3); // Multiply's right child (z)

    // Verify Tree 2 is now: (x + 5)
    EXPECT_EQ(new_population.nodes[capacity], 2);  // Root still Multiply
    EXPECT_EQ(new_population.nodes[capacity+1], 0); // variable x
    EXPECT_EQ(new_population.nodes[capacity+2], 1); // value 5

    // Verify Tree 2's children
    EXPECT_EQ(new_population.children[capacity*2], 1); // Multiply's left now Add
    EXPECT_EQ(new_population.children[capacity*2+1], 2); // Multiply's right still z
    EXPECT_EQ(new_population.children[(capacity+1)*2], -1); //
    EXPECT_EQ(new_population.children[(capacity+1)*2+1], -1); //

    new_population.free();
    population.free();
}

TEST_F(SubtreeCrossoverTest, Tree2GPUTree) {
    // Make node tree 1: // (x + 5) + y
    AddOperator *root1 = new AddOperator();
    AddOperator *add1 = new AddOperator();
    VariableNode *x = new VariableNode("x");
    ConstNode *five = new ConstNode(5.0f);
    VariableNode *y = new VariableNode("y");

    root1->addChildren({add1, y});
    add1->addChildren({x, five});

    // Make node tree 2: // (8 * z)
    MultiplyOperator *root2 = new MultiplyOperator();
    ConstNode *eight = new ConstNode(8.0f);
    VariableNode *z = new VariableNode("z");

    root2->addChildren({eight, z});

    // Make GPUTree population
    GPUTree population;
    const size_t capacity = 10;
    const size_t population_size = 2;
    population.allocate(capacity, population_size);

    // Add first tree
    population.addSolution(0, &(new Solution())->setRoot(root1));
    // Add second tree
    population.addSolution(1, &(new Solution())->setRoot(root2));

    // Print the population for debugging
    std::cout << "Initial population:" << std::endl;
    printTree(population, 0);
    printTree(population, 1);

}


TEST_F(SubtreeCrossoverTest, BasicCUDATest) {
    // Simple CUDA test to verify the environment is working
    int *d_test, h_test = 42;
    cudaMalloc(&d_test, sizeof(int));
    cudaMemcpy(d_test, &h_test, sizeof(int), cudaMemcpyHostToDevice);

    int result;
    cudaMemcpy(&result, d_test, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result, 42);
    std::cout << "Basic CUDA test passed - result: " << result << std::endl;

    cudaFree(d_test);
}