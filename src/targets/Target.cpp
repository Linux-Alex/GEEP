//
// Created by aleks on 4.3.2025.
//

#include "Target.h"

Target::Target() { }

Target::Target(const std::map<std::string, double> &state, double targetValue) : state(state), targetValue(targetValue) { }

Target & Target::setCondition(const std::string &key, double value) {
    state.insert({key, value});
    return *this;
}

Target & Target::setTargetValue(double targetValue) {
    this->targetValue = targetValue;
    return *this;
}
