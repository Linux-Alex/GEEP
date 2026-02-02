//
// Created by aleks on 30. 11. 25.
//

#include "XMLParser.h"
#include <iostream>
#include <QTextStream>
#include <QTemporaryFile>
#include "../nodes/functions/AddOperator.h"
#include "../nodes/functions/MultiplyOperator.h"
#include "../nodes/functions/DivideOperator.h"
#include "../nodes/functions/SubtractOperator.h"
#include "../nodes/terminals/VariableNode.h"
#include "../nodes/terminals/ConstNode.h"
#include "../targets/Target.h"

XMLParser::XMLParser() : bestFitness(std::numeric_limits<double>::max()), generationsRun(0) {}

bool XMLParser::parseAndRun(const QString& xmlContent) {
    QXmlStreamReader xml(xmlContent);

    try {
        while (!xml.atEnd() && !xml.hasError()) {
            if (xml.isStartElement() && xml.name() == QString("genetic_algorithm")) {
                // Parse the entire genetic algorithm configuration
                SymbolicRegressionProblem problem = parseGeneticAlgorithm(xml);

                // Create and configure the task (execution mode is already set in parseGeneticAlgorithm)
                Task task("XML Configured Symbolic Regression");
                task.setProblem(&problem);

                // Run the task
                task.run();

                // Store results
                bestSolution = problem.getBestSolution();
                bestFitness = problem.getBestFitness();
                generationsRun = problem.getGenerationsRunned();

                std::cout << "Genetic algorithm completed successfully!" << std::endl;
                std::cout << "Best fitness: " << bestFitness << std::endl;
                std::cout << "Generations run: " << generationsRun << std::endl;
                std::cout << "Best solution: " << bestSolution << std::endl;

                return true;
            }
            xml.readNext();
        }

        if (xml.hasError()) {
            std::cerr << "XML parsing error: " << xml.errorString().toStdString() << std::endl;
            return false;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error running genetic algorithm: " << e.what() << std::endl;
        return false;
    }

    return false;
}

SymbolicRegressionProblem XMLParser::parseGeneticAlgorithm(QXmlStreamReader& xml) {
    SymbolicRegressionProblem problem("XML Configured Problem");
    Task task("XML Configured Symbolic Regression"); // Create task here
    QString executionMode = "cpu"; // Default

    while (!(xml.isEndElement() && xml.name() == QString("genetic_algorithm"))) {
        xml.readNext();

        if (xml.isStartElement()) {
            if (xml.name() == QString("problem_name")) {
                problem = SymbolicRegressionProblem(xml.readElementText().toStdString());
            }
            else if (xml.name() == QString("stopping_criteria")) {
                parseStoppingCriteria(xml, problem);
            }
            else if (xml.name() == QString("algorithm_parameters")) {
                parseAlgorithmParameters(xml, problem);
            }
            else if (xml.name() == QString("selection_method")) {
                parseSelectionMethod(xml, problem);
            }
            else if (xml.name() == QString("crossover_method")) {
                parseCrossoverMethod(xml, problem);
            }
            else if (xml.name() == QString("mutation_method")) {
                parseMutationMethod(xml, problem);
            }
            else if (xml.name() == QString("function_set")) {
                parseFunctionSet(xml, problem);
            }
            else if (xml.name() == QString("terminal_set")) {
                parseTerminalSet(xml, problem);
            }
            else if (xml.name() == QString("execution_mode")) {
                // Read execution mode directly here
                executionMode = xml.readElementText().trimmed();
            }
            else if (xml.name() == QString("target_data")) {
                parseTargetData(xml, problem);
            }
        }
    }

    // Set execution mode after parsing everything
    if (executionMode == "gpu") {
        task.setExecutionMode(Task::ExecutionMode::GPU);
    } else {
        task.setExecutionMode(Task::ExecutionMode::CPU);
    }

    return problem;
}

void XMLParser::parseStoppingCriteria(QXmlStreamReader& xml, SymbolicRegressionProblem& problem) {
    StopCriterion stopCrit;

    while (!(xml.isEndElement() && xml.name() == QString("stopping_criteria"))) {
        xml.readNext();

        if (xml.isStartElement() && xml.name() == QString("criterion")) {
            QString type = xml.attributes().value("type").toString();
            int value = xml.readElementText().toInt();

            if (type == "generations") {
                stopCrit.addCriterion(GENERATIONS, value);
            }
            else if (type == "mse") {
                stopCrit.addCriterion(MSE, value);
            }
        }
    }

    problem.setStopCrit(stopCrit);
}

void XMLParser::parseAlgorithmParameters(QXmlStreamReader& xml, SymbolicRegressionProblem& problem) {
    while (!(xml.isEndElement() && xml.name() == QString("algorithm_parameters"))) {
        xml.readNext();

        if (xml.isStartElement()) {
            if (xml.name() == QString("population_size")) {
                problem.setPopulationSize(xml.readElementText().toInt());
            }
            else if (xml.name() == QString("elitism_count")) {
                problem.setElitism(xml.readElementText().toInt());
            }
            else if (xml.name() == QString("max_depth")) {
                problem.setMaxDepth(xml.readElementText().toInt());
            }
            else if (xml.name() == QString("max_nodes")) {
                problem.setMaxNodes(xml.readElementText().toInt());
            }
        }
    }
}

void XMLParser::parseSelectionMethod(QXmlStreamReader& xml, SymbolicRegressionProblem& problem) {
    QString type;
    int tournamentSize = 0;

    while (!(xml.isEndElement() && xml.name() == QString("selection_method"))) {
        xml.readNext();

        if (xml.isStartElement()) {
            if (xml.name() == QString("type")) {
                type = xml.readElementText();
            }
            else if (xml.name() == QString("tournament_size")) {
                tournamentSize = xml.readElementText().toInt();
            }
        }
    }

    if (type == "tournament" && tournamentSize > 0) {
        problem.setSelection(new TournamentSelection(tournamentSize));
    }
}

void XMLParser::parseCrossoverMethod(QXmlStreamReader& xml, SymbolicRegressionProblem& problem) {
    QString type;
    float reproductionRate = 0.0f;

    while (!(xml.isEndElement() && xml.name() == QString("crossover_method"))) {
        xml.readNext();

        if (xml.isStartElement()) {
            if (xml.name() == QString("type")) {
                type = xml.readElementText();
            }
            else if (xml.name() == QString("reproduction_rate")) {
                reproductionRate = xml.readElementText().toFloat();
            }
        }
    }

    if (type == "subtree_crossover") {
        auto crossover = new SubtreeCrossover();
        crossover->setReproductionRate(reproductionRate);
        problem.setCrossover(crossover);
    }
}

void XMLParser::parseMutationMethod(QXmlStreamReader& xml, SymbolicRegressionProblem& problem) {
    QString type;
    float mutationRate = 0.0f;

    while (!(xml.isEndElement() && xml.name() == QString("mutation_method"))) {
        xml.readNext();

        if (xml.isStartElement()) {
            if (xml.name() == QString("type")) {
                type = xml.readElementText();
            }
            else if (xml.name() == QString("mutation_rate")) {
                mutationRate = xml.readElementText().toFloat();
            }
        }
    }

    if (!type.isEmpty() && mutationRate > 0.0f) {
        problem.setMutation(new Mutation(mutationRate));
    }
}

std::vector<FunctionFactory> XMLParser::createFunctionSet(const QStringList& functions) {
    std::vector<FunctionFactory> functionSet;

    for (const QString& func : functions) {
        if (func == "add") {
            functionSet.push_back([]() { return new AddOperator(); });
        }
        else if (func == "multiply") {
            functionSet.push_back([]() { return new MultiplyOperator(); });
        }
        else if (func == "divide") {
            functionSet.push_back([]() { return new DivideOperator(); });
        }
        else if (func == "subtract") {
            functionSet.push_back([]() { return new SubtractOperator(); });
        }
    }

    return functionSet;
}

std::vector<TerminalFactory> XMLParser::createTerminalSet(const QStringList& variables, bool useConstants) {
    std::vector<TerminalFactory> terminalSet;

    for (const QString& var : variables) {
        terminalSet.push_back([var]() { return new VariableNode(var.toStdString()); });
    }

    if (useConstants) {
        terminalSet.push_back([]() { return new ConstNode(); });
    }

    return terminalSet;
}

void XMLParser::parseExecutionMode(QXmlStreamReader& xml, Task& task) {
    // Read the execution_mode element text directly
    QString mode = xml.readElementText().trimmed();
    if (mode == "gpu") {
        task.setExecutionMode(Task::ExecutionMode::GPU);
    } else {
        task.setExecutionMode(Task::ExecutionMode::CPU);
    }
}

void XMLParser::parseTargetData(QXmlStreamReader& xml, SymbolicRegressionProblem& problem) {
    QString delimiter = xml.attributes().value("delimiter").toString();
    QString targetColumn = xml.attributes().value("target_column").toString();
    QString csvData = xml.readElementText();

    if (delimiter.isEmpty() || targetColumn.isEmpty() || csvData.isEmpty()) {
        throw std::runtime_error("Invalid target data configuration");
    }

    char delimChar = delimiter.at(0).toLatin1();
    std::vector<Target> targets = parseCSVData(csvData, delimChar, targetColumn);
    problem.setTargets(targets);
}

// Helper method implementations
std::vector<Target> XMLParser::parseCSVData(const QString& csvData, char delimiter, const QString& targetColumn) {
    // Use your existing Target::readTargetsFromCSV logic but with string data instead of file
    QTextStream stream(csvData.toUtf8());
    QStringList lines;

    while (!stream.atEnd()) {
        QString line = stream.readLine().trimmed();
        if (!line.isEmpty()) {
            lines.append(line);
        }
    }

    if (lines.size() < 2) {
        throw std::runtime_error("Insufficient CSV data");
    }

    // For now, create a temporary file and use your existing CSV parser
    // In production, you might want to modify Target::readTargetsFromCSV to accept string data
    QTemporaryFile tempFile;
    if (tempFile.open()) {
        QTextStream fileStream(&tempFile);
        for (const QString& line : lines) {
            fileStream << line << "\n";
        }
        tempFile.close();

        return Target::readTargetsFromCSV(tempFile.fileName().toStdString(), delimiter, targetColumn.toStdString());
    }

    throw std::runtime_error("Failed to parse CSV data");
}

void XMLParser::parseFunctionSet(QXmlStreamReader& xml, SymbolicRegressionProblem& problem) {
    QStringList functions;

    while (!(xml.isEndElement() && xml.name() == QString("function_set"))) {
        xml.readNext();

        if (xml.isStartElement() && xml.name() == QString("function")) {
            functions.append(xml.readElementText());
        }
    }

    problem.setFunctionSet(createFunctionSet(functions));
}

void XMLParser::parseTerminalSet(QXmlStreamReader& xml, SymbolicRegressionProblem& problem) {
    QStringList variables;
    bool useConstants = false;

    while (!(xml.isEndElement() && xml.name() == QString("terminal_set"))) {
        xml.readNext();

        if (xml.isStartElement()) {
            if (xml.name() == QString("variable")) {
                variables.append(xml.readElementText());
            }
            else if (xml.name() == QString("constant")) {
                useConstants = true;
                xml.readElementText(); // Read empty text
            }
        }
    }

    problem.setTerminalSet(createTerminalSet(variables, useConstants));
}