#include "GPUTree.h"
#include "../solutions/Solution.h"
#include "../nodes/Node.h"
#include "../nodes/FunctionNode.h"
#include "../nodes/TerminalNode.h"

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
   // TODO: Implementation to convert Solution to linear GPU format
}
