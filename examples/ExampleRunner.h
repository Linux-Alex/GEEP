//
// Created by aleks on 1.3.2025.
//

#ifndef EXAMPLERUNNER_H
#define EXAMPLERUNNER_H

// ExampleRunner.h
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype>
#include <functional>

class ExampleRunner {
public:
    using RunFunction = std::function<void()>;

    // Run a program or XML example based on the input string
    static void run(const std::string& input);

    // List all programs, XMLs, or both
    static void listAll(bool listPrograms = true, bool listXMLs = true);

    // Register a program
    static void registerProgram(const std::string& name, RunFunction runFunction);

private:
    // Map of program names to their run functions
    static std::map<std::string, RunFunction>& getProgramMap();

    // Helper methods
    static std::string getProgramName(const std::string& number);
    static std::string searchProgram(const std::string& keyword);
    static void runProgram(const std::string& programName);
    static std::vector<std::string> getProgramNames();
    static std::vector<std::string> getXMLNames();
    static std::string formatNumber(const std::string& number);
    static std::string normalizeString(const std::string& str);
};

// Macro to register a program
#define REGISTER_PROGRAM(name) \
    static void run(); \
    namespace { \
        struct ProgramRegistrar_##name { \
            ProgramRegistrar_##name() { \
                ExampleRunner::registerProgram(#name, &run); \
            } \
        }; \
        static ProgramRegistrar_##name programRegistrar_##name; \
    } \
    void run()

#endif //EXAMPLERUNNER_H
