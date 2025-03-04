//
// Created by aleks on 1.3.2025.
//

#ifndef LOGHELPER_H
#define LOGHELPER_H

#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

// Define color and style macros
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"

class LogHelper {
private:
    static std::string getCurrentTimestamp();

public:
    static void logMessage(const std::string& message, bool isError = false);
};



#endif //LOGHELPER_H
