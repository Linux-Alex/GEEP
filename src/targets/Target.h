//
// Created by aleks on 4.3.2025.
//

#ifndef TARGET_H
#define TARGET_H
#include <map>
#include <string>
#include <vector>


class Target {
protected:
    std::map<std::string, double> state;
    double targetValue;

public:
    Target();
    Target(const std::map<std::string, double> &state, double targetValue);

    Target& setCondition(const std::string &key, double value);
    Target& setTargetValue(double targetValue);

    const std::map<std::string, double>& getState() const { return state; }
    double getTargetValue() const { return targetValue; }

    // Read targets from a CSV file
    static std::vector<Target> readTargetsFromCSV(const std::string &filename, char delimiter, const std::string &targetColumn);
};



#endif //TARGET_H
