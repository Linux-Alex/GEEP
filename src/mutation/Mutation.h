//
// Created by aleks on 14. 11. 25.
//

#ifndef GEEP_MUTATION_H
#define GEEP_MUTATION_H
#include <c++/12/random>

#include "../solutions/Solution.h"
#include "../cuda/GPUTree.h"

class Problem;

class Mutation {
private:
    float mutationRate = 0.005f;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;

public:
    Mutation(float mutationRate = 0.005f) : mutationRate(mutationRate) {}

    ~Mutation() = default;

    void mutation(Solution *&tree, Problem *problem);
    void mutationGPU(GPUTree *population, size_t numberOfVariables);

    void mutate(std::vector<FunctionFactory> functionSet, std::vector<TerminalFactory> terminalSet, Node *&node);

    float getMutationRate() { return mutationRate; }
};


#endif //GEEP_MUTATION_H