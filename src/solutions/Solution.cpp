//
// Created by aleks on 28.2.2025.
//

#include "Solution.h"

std::atomic<size_t> Solution::ID_COUNTER = 0; // Initialize static member

Solution::Solution() : id(Solution::ID_COUNTER++) { }

Solution::Solution(const Solution &other): id(ID_COUNTER++) {
    values = other.values;
    ancestors = other.ancestors;
}

void Solution::addAncestor(Solution *ancestor) {
    ancestors.push_back(ancestor);
}

void Solution::addAncestor(const std::vector<Solution *> &ancestors) {
    this->ancestors.insert(this->ancestors.end(), ancestors.begin(), ancestors.end());
}

void Solution::setAncestor(Solution *ancestor) {
    ancestors.clear();
    ancestors.push_back(ancestor);
}

void Solution::setAncestors(const std::vector<Solution *> &ancestors) {
    this->ancestors = ancestors;
}

void Solution::setValue(double value) { values.push_back(value); }

void Solution::setValues(const std::vector<double> &values) { this->values = values; }

void Solution::setRoot(std::unique_ptr<Node> root) { this->root = std::move(root); }
