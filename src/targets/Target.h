//
// Created by aleks on 4.3.2025.
//

#ifndef TARGET_H
#define TARGET_H
#include <map>
#include <string>


class Target {
protected:
    std::map<std::string, double> state;
    double targetValue;

public:
    Target();
    Target(const std::map<std::string, double> &state, double targetValue);

    Target& setCondition(const std::string &key, double value);
    Target& setTargetValue(double targetValue);
};



#endif //TARGET_H
