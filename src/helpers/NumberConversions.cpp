//
// Created by aleks on 4.3.2025.
//

#include "NumberConversions.h"

std::string NumberConversions::doubleToString(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}
