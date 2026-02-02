//
// Created by aleks on 30. 11. 25.
//

#ifndef XMLPARSER_H
#define XMLPARSER_H

#include <QString>
#include <QXmlStreamReader>
#include <QFile>
#include <memory>
#include "../problems/SymbolicRegressionProblem.cuh"
#include "../selections/TournamentSelection.h"
#include "../crossover/SubtreeCrossover.h"
#include "../mutation/Mutation.h"
#include "../tasks/Task.h"

class XMLParser {
public:
    XMLParser();

    // Parse XML configuration and run the genetic algorithm
    bool parseAndRun(const QString& xmlContent);

    // Get the result after running
    std::string getBestSolution() const { return bestSolution; }
    double getBestFitness() const { return bestFitness; }
    int getGenerationsRun() const { return generationsRun; }

private:
    std::string bestSolution;
    double bestFitness;
    int generationsRun;

    // Parsing methods
    SymbolicRegressionProblem parseGeneticAlgorithm(QXmlStreamReader& xml);
    void parseStoppingCriteria(QXmlStreamReader& xml, SymbolicRegressionProblem& problem);
    void parseAlgorithmParameters(QXmlStreamReader& xml, SymbolicRegressionProblem& problem);
    void parseSelectionMethod(QXmlStreamReader& xml, SymbolicRegressionProblem& problem);
    void parseCrossoverMethod(QXmlStreamReader& xml, SymbolicRegressionProblem& problem);
    void parseMutationMethod(QXmlStreamReader& xml, SymbolicRegressionProblem& problem);
    void parseFunctionSet(QXmlStreamReader& xml, SymbolicRegressionProblem& problem);
    void parseTerminalSet(QXmlStreamReader& xml, SymbolicRegressionProblem& problem);
    void parseExecutionMode(QXmlStreamReader& xml, Task& task);
    void parseTargetData(QXmlStreamReader& xml, SymbolicRegressionProblem& problem);

    // Helper methods
    std::vector<Target> parseCSVData(const QString& csvData, char delimiter, const QString& targetColumn);
    std::vector<FunctionFactory> createFunctionSet(const QStringList& functions);
    std::vector<TerminalFactory> createTerminalSet(const QStringList& variables, bool useConstants);
};

#endif //GEEP_XMLPARSER_H