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

   cudaError_t err;
   err = cudaMallocManaged(&nodes, max_nodes * population_size * sizeof(int));
   if (err != cudaSuccess) throw std::runtime_error("Failed to allocate nodes");

   err = cudaMallocManaged(&values, max_nodes * population_size * sizeof(float));
   if (err != cudaSuccess) throw std::runtime_error("Failed to allocate values");

   err = cudaMallocManaged(&children, max_nodes * 2 * population_size * sizeof(int));
   if (err != cudaSuccess) throw std::runtime_error("Failed to allocate children");

   err = cudaMallocManaged(&parent_indices, max_nodes * population_size * sizeof(int));
   if (err != cudaSuccess) throw std::runtime_error("Failed to allocate parent_indices");

   err = cudaMallocManaged(&node_counts, population_size * sizeof(size_t));
   if (err != cudaSuccess) throw std::runtime_error("Failed to allocate node_counts");

   err = cudaMallocManaged(&selection_parent1_idx, population_size * sizeof(size_t));
   if (err != cudaSuccess) throw std::runtime_error("Failed to allocate selection_parent1_idx");

   err = cudaMallocManaged(&selection_parent2_idx, population_size * sizeof(size_t));
   if (err != cudaSuccess) throw std::runtime_error("Failed to allocate selection_parent2_idx");

   err = cudaMallocManaged(&fitness_values, population_size * sizeof(float));
   if (err != cudaSuccess) throw std::runtime_error("Failed to allocate fitness_values");
}

void GPUTree::free() {
   cudaFree(nodes);
   cudaFree(values);
   cudaFree(children);
   cudaFree(parent_indices);
   cudaFree(node_counts);

   cudaFree(selection_parent1_idx);
   cudaFree(selection_parent2_idx);

   cudaFree(fitness_values);
}

__host__ void GPUTree::addSolution(int index, Solution *solution) {
   if (index >= population) {
      throw std::out_of_range("Solution index exceeds population size");
   }

   size_t offset = index * capacity;
   node_counts[index] = 0; // Initialize count

   // Use stack for iterative tree traversal
   std::stack<std::pair<Node *, int> > node_stack;
   Node *root = solution->getRoot();

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
      int node_type = 0;
      float node_value = 0.0f;

      if (dynamic_cast<FunctionNode *>(node)) {
         FunctionNode *fn = dynamic_cast<FunctionNode *>(node);
         if (dynamic_cast<AddOperator *>(fn)) {
            node_type = 2; // Function node
         } else if (dynamic_cast<SubtractOperator *>(fn)) {
            node_type = 3; // Function node
         } else if (dynamic_cast<MultiplyOperator *>(fn)) {
            node_type = 4; // Function node
         } else if (dynamic_cast<DivideOperator *>(fn)) {
            node_type = 5; // Function node
         } else {
            throw std::runtime_error("Unknown function type");
         }
      } else if (dynamic_cast<TerminalNode *>(node)) {
         TerminalNode *tn = dynamic_cast<TerminalNode *>(node);
         if (dynamic_cast<VariableNode *>(tn)) {
            node_type = 0; // Variable node
            // OLD: node_value = 0.0f; // Explicitly set to 0 (value will come from target_data)
            // Set node_value as the variable ID
            VariableNode *vn = dynamic_cast<VariableNode *>(tn);
            node_value = static_cast<float>(getVariableId(vn->toString()));
         } else if (dynamic_cast<ConstNode *>(tn)) {
            node_type = 1; // Constant node
            node_value = tn->evaluate({}); // Evaluate constant value
         } else {
            throw std::runtime_error("Unknown terminal type");
         }
      } else {
         throw std::runtime_error("Unknown node type");
      }

      nodes[offset + current_pos] = node_type;
      values[offset + current_pos] = node_value;
      parent_indices[offset + current_pos] = parent_pos;

      // Handle children (reverse order from stack)
      if (dynamic_cast<FunctionNode *>(node)) {
         FunctionNode *fn = dynamic_cast<FunctionNode *>(node);
         const auto &children_nodes = fn->getChildren();

         // Store children indices (-1 for now)
         children[(offset + current_pos) * 2] = -1; // Left child
         children[(offset + current_pos) * 2 + 1] = -1; // Right child

         // Push children to stack with current position as parent
         for (auto it = children_nodes.rbegin(); it != children_nodes.rend(); ++it) {
            node_stack.push({*it, current_pos});
         }
      } else {
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

std::vector<Solution *> GPUTree::getCPUSolutions() {
   // Check if population is valid
   if (population <= 0) {
      throw std::runtime_error("Population size is not valid");
   }

   // Reverse mapping for variable indices
   std::unordered_map<int, std::string> variable_reverse_map;
   for (const auto &pair: variable_indices) {
      variable_reverse_map[pair.second] = pair.first;
   }

   // Allocate memory for CPU solutions
   std::vector<Solution *> solutions(population);

   // Generate solutions
   try {
      for (size_t i = 0; i < population; i++) {
         size_t offset = i * capacity;
         size_t node_count = node_counts[i];

         if (node_count == 0) {
            solutions[i] = new Solution();
            continue; // Empty solution
         }

         // Create a mapping from positions to nodes
         std::vector<Node *> node_map(node_count, nullptr);

         // Preprocess nodes in reverse order (children before parents)
         for (int pos = node_count - 1; pos >= 0; pos--) {
            int absolute_pos = offset + pos;
            int node_type = nodes[absolute_pos];
            float node_value = values[absolute_pos];

            Node *node = nullptr;

            // Create appropriate node type
            switch (node_type) {
               case 0: {
                  // VariableNode
                  int varId = static_cast<int>(values[absolute_pos]);
                  auto it = variable_reverse_map.find(varId);

                  if (it == variable_reverse_map.end()) {
                     throw std::runtime_error("Variable ID not found in reverse map");
                  }

                  node = new VariableNode(it->second);
                  break;
               }
               case 1: // ConstNode
                  node = new ConstNode(node_value);
                  break;
               case 2: // AddOperator
                  node = new AddOperator();
                  break;
               case 3: // SubtractOperator
                  node = new SubtractOperator();
                  break;
               case 4: // MultiplyOperator
                  node = new MultiplyOperator();
                  break;
               case 5: // DivideOperator
                  node = new DivideOperator();
                  break;
               default:
                  throw std::runtime_error("Unknown node type: " + std::to_string(node_type));
            }

            node_map[pos] = node;

            // If it's a function node, add the children
            if (node_type >= 2) {
               FunctionNode *fn = dynamic_cast<FunctionNode *>(node);
               if (!fn) {
                  throw std::runtime_error("Type mismatch: Node is not a function node");
               }

               int left_child = children[absolute_pos * 2];
               int right_child = children[absolute_pos * 2 + 1];

               if (left_child != -1) {
                  int child_pos = left_child - offset;
                  if (child_pos < 0 || child_pos >= node_count) {
                     throw std::runtime_error("Invalid left child index: " + std::to_string(left_child));
                  }
                  fn->addChild(node_map[child_pos]);
               }

               if (right_child != -1) {
                  int child_pos = right_child - offset;
                  if (child_pos < 0 || child_pos >= node_count) {
                     throw std::runtime_error("Invalid right child index: " + std::to_string(right_child));
                  }
                  fn->addChild(node_map[child_pos]);
               }
            }
         }

         // Create the solution with the root node
         solutions[i] = new Solution();
         solutions[i]->setRoot(node_map[0]);
      }
   } catch (const std::exception &e) {
      std::cerr << "Error generating CPU solutions: " << e.what() << std::endl;
      for (Solution *s: solutions) {
         if (s)
            delete s;
      }
      throw;
   }

   return solutions;
}

GPUTree GPUTree::extractTree(int index) {
   // Check if index is valid
   if (index >= population) {
      throw std::out_of_range("Solution index exceeds population size");
   }

   GPUTree singleTree;
   size_t offset = index * capacity;

   // Points to the same memory location as the original
   singleTree.nodes = nodes + offset;
   singleTree.values = values + offset;
   singleTree.children = children + offset * 2;
   singleTree.parent_indices = parent_indices + offset;
   singleTree.node_counts = node_counts + index;

   singleTree.capacity = capacity;
   singleTree.population = 1; // Only one tree in this instance

   return singleTree;
}

int GPUTree::getVariableId(const std::string &name) {
   auto it = variable_indices.find(name);
   if (it == variable_indices.end()) {
      variable_indices[name] = nextVariableId;
      return nextVariableId++;
   }
   return it->second;
}

void GPUTree::clearVariableMapping() {
   variable_indices.clear();
   nextVariableId = 0;
}

