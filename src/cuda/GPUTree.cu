#include "GPUTree.h"

#include <iostream>

#include "CudaUtils.h"
#include "../solutions/Solution.h"
#include "../nodes/Node.h"
#include "../nodes/FunctionNode.h"
#include "../nodes/TerminalNode.h"

#include "../nodes/functions/AddOperator.h"
#include "../nodes/functions/SubtractOperator.h"
#include "../nodes/functions/MultiplyOperator.h"
#include "../nodes/functions/DivideOperator.h"

#include "../nodes/terminals/VariableNode.h"
#include "../nodes/terminals/ConstNode.h"

void GPUTree::allocate(size_t max_nodes, size_t population_size) {
   capacity = max_nodes;
   population = population_size;

   cudaMallocManaged(&nodes, max_nodes * population_size * sizeof(int));
   cudaMallocManaged(&values, max_nodes * population_size * sizeof(float));
   cudaMallocManaged(&children, max_nodes * 2 * population_size * sizeof(int));
   cudaMallocManaged(&parent_indices, max_nodes * population_size * sizeof(int));
   cudaMallocManaged(&node_counts, max_nodes * population_size * sizeof(size_t));
}

void GPUTree::free() {
   cudaFree(nodes);
   cudaFree(values);
   cudaFree(children);
   cudaFree(parent_indices);
   cudaFree(node_counts);
}

__host__ void GPUTree::addSolution(int index, Solution *solution) {
   if (index >= population) {
      throw std::out_of_range("Solution index exceeds population size");
   }

   size_t offset = index * capacity;
   node_counts[index] = 0; // Initialize count

   // Use stack for iterative tree traversal
   std::stack<std::pair<Node*, int>> node_stack;
   Node* root = solution->getRoot();

   if (!root) {
      std::cerr << "Error: Solution has no root node." << std::endl;
      return;
   }

   // std::cout << "Root node: " << root->toString() << std::endl;

   node_stack.push({root, -1}); // -1 indicates no parent
   int current_pos = 0;

   while (!node_stack.empty() && current_pos < capacity) {
      auto [node, parent_pos] = node_stack.top();
      node_stack.pop();

      // Store node information
      int node_type;
      float node_value = 0.0f;

      if (dynamic_cast<FunctionNode*>(node)) {
         FunctionNode *fn = dynamic_cast<FunctionNode*>(node);
         if (dynamic_cast<AddOperator*>(fn)) {
            node_type = 2; // Function node
         }
         else if (dynamic_cast<SubtractOperator*>(fn)) {
            node_type = 3; // Function node
         }
         else if (dynamic_cast<MultiplyOperator*>(fn)) {
            node_type = 4; // Function node
         }
         else if (dynamic_cast<DivideOperator*>(fn)) {
            node_type = 5; // Function node
         }
         else {
            throw std::runtime_error("Unknown function type");
         }
      }
      else if (dynamic_cast<TerminalNode*>(node)) {
         TerminalNode *tn = dynamic_cast<TerminalNode*>(node);
         if (dynamic_cast<VariableNode*>(tn)) {
            node_type = 0; // Variable node
            node_value = 0.0f; // Explicitly set to 0 (value will come from target_data)
         }
         else if (dynamic_cast<ConstNode*>(tn)) {
            node_type = 1; // Constant node
            node_value = tn->evaluate({}); // Evaluate constant value
         }
         else {
            throw std::runtime_error("Unknown terminal type");
         }
      }
      else {
         throw std::runtime_error("Unknown node type");
      }

      nodes[offset + current_pos] = node_type;
      values[offset + current_pos] = node_value;
      parent_indices[offset + current_pos] = parent_pos;

      // Handle children (reverse order from stack)
      if (dynamic_cast<FunctionNode*>(node)) {
         FunctionNode* fn = dynamic_cast<FunctionNode*>(node);
         const auto& children_nodes = fn->getChildren();

         // Store children indices (-1 for now)
         children[(offset + current_pos) * 2] = -1; // Left child
         children[(offset + current_pos) * 2 + 1] = -1; // Right child

         // Push children to stack with current position as parent
         for (auto it = children_nodes.rbegin(); it != children_nodes.rend(); ++it) {
            node_stack.push({*it, current_pos});
         }
      }
      else {
         // Terminal nodes have no children
         children[(offset + current_pos) * 2] = -1; // Left child
         children[(offset + current_pos) * 2 + 1] = -1; // Right child
      }

      // Update parent's child pointer if needed
      if (parent_pos >= 0) {
         int child_index = 0;
         while (child_index < 2 && children[(offset + parent_pos) * 2 + child_index] != -1) {
            child_index++;
         }

         if (child_index < 2) {
            children[(offset + parent_pos) * 2 + child_index] = current_pos;
         }
      }

      current_pos++;
   }

   node_counts[index] = current_pos; // Store actual node count
}
