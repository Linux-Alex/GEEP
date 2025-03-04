//
// Created by aleks on 1.3.2025.
//

#include "LogHelper.h"

std::string LogHelper::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm = *std::localtime(&now_time_t);
    char buffer[24];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);

    char timestamp[32];
    snprintf(timestamp, sizeof(timestamp), "%s.%03ld", buffer, now_ms.count());

    return timestamp;
}

void LogHelper::logMessage(const std::string &message, bool isError) {
    std::string timestamp = getCurrentTimestamp();

    if (isError) {
        // For errors, use red for the timestamp
        std::cerr << RED << BOLD << "[" << timestamp << "]" << RESET << " " << RED << "ERROR: " << RESET << message << std::endl;
    } else {
        // For info, use bold cyan for the timestamp
        std::cout << CYAN << BOLD << "[" << timestamp << "]" << RESET << " " << GREEN << "INFO: " << RESET << message << std::endl;
    }
}
