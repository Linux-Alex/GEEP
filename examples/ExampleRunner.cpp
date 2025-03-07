//
// Created by aleks on 1.3.2025.
//

#include "ExampleRunner.h"

// ExampleRunner.cpp
#include "ExampleRunner.h"

#include "LogHelper.h"

// Run a program or XML example based on the input string
void ExampleRunner::run(const std::string& input) {
    if (input.rfind("p:", 0) == 0) { // Check if input starts with "p:"
        std::string identifier = input.substr(2);
        if (identifier.find('%') != std::string::npos) {
            // Search for a program
            std::string keyword = identifier.substr(1);
            std::string programName = searchProgram(keyword);
            if (!programName.empty()) {
                runProgram(programName);
            } else {
                std::cerr << "No program found for keyword: " << keyword << std::endl;
            }
        } else {
            // Run a specific program by number
            std::string programName = getProgramName(identifier);
            if (!programName.empty()) {
                runProgram(programName);
            } else {
                std::cerr << "Program not found: " << identifier << std::endl;
            }
        }
    } else if (input.rfind("xml:", 0) == 0) { // Check if input starts with "xml:"
        std::cerr << "XML execution is not implemented yet." << std::endl;
    } else {
        std::cerr << "Invalid input format. Use 'p:' for programs or 'xml:' for XMLs." << std::endl;
    }
}

// List all programs, XMLs, or both
void ExampleRunner::listAll(bool listPrograms, bool listXMLs) {
    if (listPrograms) {
        std::cout << "Programs:" << std::endl;
        for (const auto& program : getProgramNames()) {
            std::cout << "  " << program << std::endl;
        }
    }
    if (listXMLs) {
        std::cout << "XMLs:" << std::endl;
        for (const auto& xml : getXMLNames()) {
            std::cout << "  " << xml << std::endl;
        }
    }
}

// Register a program
void ExampleRunner::registerProgram(const std::string& name, RunFunction runFunction) {
    getProgramMap()[name] = runFunction;
}

// Get the map of program names to their run functions
std::map<std::string, ExampleRunner::RunFunction>& ExampleRunner::getProgramMap() {
    static std::map<std::string, RunFunction> programMap;
    return programMap;
}

// Helper methods
std::string ExampleRunner::getProgramName(const std::string& number) {
    std::string formattedNumber = formatNumber(number);
    for (const auto& program : getProgramNames()) {
        if (program.find(formattedNumber) != std::string::npos) {
            return program;
        }
    }
    return "";
}

std::string ExampleRunner::searchProgram(const std::string& keyword) {
    std::string normalizedKeyword = normalizeString(keyword);
    for (const auto& program : getProgramNames()) {
        std::string normalizedProgram = normalizeString(program);
        if (normalizedProgram.find(normalizedKeyword) != std::string::npos) {
            return program;
        }
    }
    return "";
}

void ExampleRunner::runProgram(const std::string& programName) {
    const auto& programMap = getProgramMap();
    auto it = programMap.find(programName);
    if (it != programMap.end()) {
        LogHelper::logMessage("Program selected: " + programName);
        it->second(); // Call the run function
    } else {
        std::cerr << "Program not found: " << programName << std::endl;
    }
}

std::vector<std::string> ExampleRunner::getProgramNames() {
    std::vector<std::string> programNames;
    for (const auto& entry : getProgramMap()) {
        programNames.push_back(entry.first);
    }
    return programNames;
}

std::vector<std::string> ExampleRunner::getXMLNames() {
    return {"001_find_function_by_points.xml"}; // Add more XMLs here
}

std::string ExampleRunner::formatNumber(const std::string& number) {
    if (number.length() == 1) {
        return "00" + number;
    } else if (number.length() == 2) {
        return "0" + number;
    }
    return number;
}

std::string ExampleRunner::normalizeString(const std::string& str) {
    std::string normalized = str;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    std::replace(normalized.begin(), normalized.end(), ' ', '_');
    return normalized;
}