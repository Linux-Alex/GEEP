//
// Created by aleks on 15.3.2025.
//

#ifndef STOPCRITERION_H
#define STOPCRITERION_H

#include <chrono>
#include <stdexcept>
#include <unordered_map>

#include "StopCriterionType.h"
#include "../../examples/LogHelper.h"

class StopCriterion {
private:
    // Stop criteria and their thresholds
    std::unordered_map<StopCriterionType, double> criteria;

    // Timer for time-based stop criterion
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
public:
    // Constructor
    StopCriterion() = default;

    // Add a criterion with a threshold
    StopCriterion& addCriterion(StopCriterionType type, double threshold) {
        this->criteria[type] = threshold;

        return *this;
    }

    // Start timer for time-based stop criterion
    void startTimer() {
        this->startTime = std::chrono::high_resolution_clock::now();
    }

    // Remove a criterion
    void removeCriterion(StopCriterionType type) {
        this->criteria.erase(type);
    }

    // Check if any criterion is met
    bool isMet(size_t evaluations, size_t generations, double mse) const {
        for (const auto& [type, threshold] : this->criteria) {
            switch (type) {
                case EVALUATIONS: {
                    if (evaluations >= threshold) {
                        LogHelper::logMessage("Stop criterion met: EVALUATIONS");
                        return true;
                    }
                    break;
                }
                case GENERATIONS: {
                    if (generations >= threshold) {
                        LogHelper::logMessage("Stop criterion met: GENERATIONS");
                        return true;
                    }
                    break;
                }
                case MSE: {
                    if (mse <= threshold) {
                        LogHelper::logMessage("Stop criterion met: MSE");
                        return true;
                    }
                    break;
                }
                case TIME: {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
                    if (elapsed >= threshold) {
                        LogHelper::logMessage("Stop criterion met: TIME");
                        return true;
                    }
                    break;
                }
                default: {
                    throw std::invalid_argument("Invalid stop criterion type");
                }
            }
        }

        return false;
    }
};



#endif //STOPCRITERION_H
