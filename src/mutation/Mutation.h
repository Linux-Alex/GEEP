//
// Created by aleks on 14. 11. 25.
//

#ifndef GEEP_MUTATION_H
#define GEEP_MUTATION_H
#include "../solutions/Solution.h"
#include "../cuda/GPUTree.h"

class Mutation {
private:
    float mutationRate = 0.005f;

public:
    ~Mutation() = default;

    void mutation(Solution *tree);
    void mutationGPU(GPUTree *population);
};


#endif //GEEP_MUTATION_H