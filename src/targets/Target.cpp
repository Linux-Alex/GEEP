//
// Created by aleks on 4.3.2025.
//

#include "Target.h"
#include <sstream>
#include <fstream>

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

std::vector<Target> Target::readTargetsFromCSV(const std::string &filename, char delimiter, const std::string &targetColumn) {
    std::vector<Target> targets;
    std::vector<std::string> headers;

    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;

    // Read headers from first line
    if (std::getline(file, line)) {
        std::stringstream headerStream(line);
        std::string header;

        while (std::getline(headerStream, header, delimiter)) {
            headers.push_back(header);
        }
    }
    else {
        throw std::runtime_error("File is empty: " + filename);
    }

    // Find target column index
    int targetColumnIndex = -1;
    for (size_t i = 0; i < headers.size(); i++) {
        if (headers[i] == targetColumn) {
            targetColumnIndex = static_cast<int>(i);
        }
    }

    if (targetColumnIndex == -1) {
        throw std::runtime_error("Target column not found: " + targetColumn);
    }

    // Read data rows
    int lineNumber = 1;
    while (std::getline(file, line)) {
        lineNumber++;

        // Skip empty lines
        if (line.empty())
            continue;

        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> values;
        std::vector<std::string> stringValues;

        // Parse all cells as string first
        while (std::getline(lineStream, cell, delimiter)) {
            stringValues.push_back(cell);
        }

        // Check if we have the right number of columns
        if (stringValues.size() != headers.size()) {
            throw std::runtime_error("Column count mismatch at line " + std::to_string(lineNumber));
        }

        // Convert to floats and validate
        float targetValue = 0.0f;

        for (size_t i = 0; i < stringValues.size(); i++) {
            try {
                float value = std::stof(stringValues[i]);
                values.push_back(value);

                // Store the target value
                if (static_cast<int>(i) == targetColumnIndex) {
                    targetValue = value;
                }
            }
            catch (const std::exception &e) {
                throw std::runtime_error("Invalid float value at line " + std::to_string(lineNumber) +
                                         ", column " + std::to_string(i + 1) + ": " + stringValues[i]);
            }
        }

        // Create Target object and set conditions
        Target target;
        for (size_t i = 0; i < headers.size(); i++) {
            if (static_cast<int>(i) != targetColumnIndex) {
                target.setCondition(headers[i], values[i]);
            }
        }

        // Set Target value
        target.setTargetValue(targetValue);

        targets.push_back(target);
    }

    return targets;
}
