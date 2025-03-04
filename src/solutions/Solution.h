//
// Created by aleks on 28.2.2025.
//

#ifndef SOLUTION_H
#define SOLUTION_H

#include <atomic>
#include <vector>

class Solution {
protected:
    size_t id;

    std::vector<double> values;
    std::vector<Solution*> parents;

private:
    // Auto incrementing ID counter
    static std::atomic<size_t> ID_COUNTER;

public:
    // Constructor
    Solution();

    // Copy constructor
    Solution(const Solution& other);

    // Add parents
    void addParent(Solution* parent);
    void addParents(const std::vector<Solution*>& parents);

    // Set parents
    void setParent(Solution* parent);
    void setParents(const std::vector<Solution*>& parents);

    // Set values
    void setValue(double value);
    void setValues(const std::vector<double>& values);

    // Getters
    size_t getId() const { return id; }
    const std::vector<double>& getValues() const { return values; }
    const std::vector<Solution*>& getParents() const { return parents; }

    // Setters
    void setId(size_t id) { this->id = id; }
};



#endif //SOLUTION_H
