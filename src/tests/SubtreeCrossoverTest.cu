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
    population.children[capacity*2] = capacity+1;   // Multiply's left child (8)
    population.children[capacity*2+1] = capacity+2; // Multiply's right child (z)

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

    // Perform crossover - swap the first Add (node 1) with Multiply (node capacity)
    SubtreeCrossover crossover;
    crossover.crossoverGPU(&population, &new_population);

    // Print the population for debugging
    std::cout << "Population after crossover:" << std::endl;
    printTree(new_population, 0);
    printTree(new_population, 1);

    // Verify Tree 1 is now: (Multiply) + y
    EXPECT_EQ(new_population.nodes[0], 2);  // Root still Add
    EXPECT_EQ(new_population.nodes[1], 4);  // Now Multiply
    EXPECT_EQ(new_population.nodes[2], 0);  // x should be gone
    EXPECT_EQ(new_population.nodes[3], 1);  // 5 should be gone
    EXPECT_EQ(new_population.nodes[4], 0);  // y remains

    // Verify Tree 1's children
    EXPECT_EQ(new_population.children[0], 1);  // Root's left child now Multiply
    EXPECT_EQ(new_population.children[1], 4);  // Root's right child still y
    EXPECT_EQ(new_population.children[2], capacity+1); // Multiply's left child (8)
    EXPECT_EQ(new_population.children[3], capacity+2); // Multiply's right child (z)

    // Verify Tree 2 is now: (Add) * z
    EXPECT_EQ(new_population.nodes[capacity], 4);  // Root still Multiply
    EXPECT_EQ(new_population.nodes[capacity+1], 2); // Now Add
    EXPECT_EQ(new_population.nodes[capacity+2], 0); // z remains

    // Verify Tree 2's children
    EXPECT_EQ(new_population.children[capacity*2], capacity+1); // Multiply's left now Add
    EXPECT_EQ(new_population.children[capacity*2+1], capacity+2); // Multiply's right still z
    EXPECT_EQ(new_population.children[(capacity+1)*2], 2); // Add's left child (x)
    EXPECT_EQ(new_population.children[(capacity+1)*2+1], 3); // Add's right child (5)

    // Verify parent indices
    EXPECT_EQ(new_population.parent_indices[1], 0);
    EXPECT_EQ(new_population.parent_indices[capacity+1], capacity);
    EXPECT_EQ(new_population.parent_indices[capacity+1+2], capacity+1); // x's parent is Add
    EXPECT_EQ(new_population.parent_indices[capacity+1+3], capacity+1); // 5's parent is Add

    // Add these checks
    ASSERT_NE(new_population.parent_indices[1], -1); // Swapped node should have parent
    ASSERT_EQ(new_population.parent_indices[capacity], 0); // Now should be child of root

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

    // Perform crossover - swap the first Add (node 1) with Multiply (node 0 in second tree)
    // SubtreeCrossover crossover;
    // crossover.crossoverGPU(&population, 0, 1, 1, 0);

}