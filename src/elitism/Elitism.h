//
// Created by aleks on 30. 11. 25.
//

#ifndef GEEP_ELITISM_H
#define GEEP_ELITISM_H

#include <cstddef>
#include "../cuda/GPUTree.h"

class Elitism {
private:
    size_t elitismCount;

public:
    ~Elitism() = default;

    void applyElitismGPU(GPUTree* old_population, GPUTree* new_population);
    void setElitismCount(size_t count) { elitismCount = count; }
    size_t getElitismCount() const { return elitismCount; }
};


#endif //GEEP_ELITISM_H