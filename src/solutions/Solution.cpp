//
// Created by aleks on 28.2.2025.
//

#include "Solution.h"

std::atomic<size_t> Solution::ID_COUNTER = 0; // Initialize static member

Solution::Solution() : id(Solution::ID_COUNTER++) { }

Solution::Solution(const Solution &other): id(ID_COUNTER++) {
    values = other.values;
    parents = other.parents;
}

void Solution::addParent(Solution *parent) {
    parents.push_back(parent);
}

void Solution::addParents(const std::vector<Solution *> &parents) {
    this->parents.insert(this->parents.end(), parents.begin(), parents.end());
}

void Solution::setParent(Solution *parent) {
    parents.clear();
    parents.push_back(parent);
}

void Solution::setParents(const std::vector<Solution *> &parents) {
    this->parents = parents;
}

void Solution::setValue(double value) { values.push_back(value); }

void Solution::setValues(const std::vector<double> &values) { this->values = values; }
