//
// Created by aleks on 26.2.2025.
//

#ifndef GEEPCONFIG_H
#define GEEPCONFIG_H

#include <QString>
#include <QStringList>
#include "selections/SelectionMethod.h"

class GEEPConfig {
private:
    double mutationProbability;
    double reproductionProbability;
    double crossoverProbability;
    size_t populationSize;
    size_t maxGenerations;
    SelectionMethod selectionMethod;
    size_t tournamentSize;
    size_t elitismSize;
    QStringList terminals;
    QStringList functions;

public:
    // Constructor
    GEEPConfig();

    // Methods to process XML file
    bool loadFromXml(const QString &filePath);

    // Method to send data to CUDA
    void sendToCuda();

    // Getters
    double getMutationProbability() const { return mutationProbability; }
    double getReproductionProbability() const { return reproductionProbability; }
    double getCrossoverProbability() const { return crossoverProbability; }
    size_t getPopulationSize() const { return populationSize; }
    size_t getMaxGenerations() const { return maxGenerations; }
    SelectionMethod getSelectionMethod() const { return selectionMethod; }
    size_t getTournamentSize() const { return tournamentSize; }
    size_t getElitismSize() const { return elitismSize; }
    QStringList getTerminals() const { return terminals; }
    QStringList getFunctions() const { return functions; }

    // Setters
    void setMutationProbability(double mutationProbability) { this->mutationProbability = mutationProbability; }
    void setReproductionProbability(double reproductionProbability) { this->reproductionProbability = reproductionProbability; }
    void setCrossoverProbability(double crossoverProbability) { this->crossoverProbability = crossoverProbability; }
    void setPopulationSize(size_t populationSize) { this->populationSize = populationSize; }
    void setMaxGenerations(size_t maxGenerations) { this->maxGenerations = maxGenerations; }
    void setSelectionMethod(SelectionMethod selectionMethod) { this->selectionMethod = selectionMethod; }
    void setTournamentSize(size_t tournamentSize) { this->tournamentSize = tournamentSize; }
    void setElitismSize(size_t elitismSize) { this->elitismSize = elitismSize; }
    void setTerminals(const QStringList &terminals) { this->terminals = terminals; }
    void setFunctions(const QStringList &functions) { this->functions = functions; }
};



#endif //GEEPCONFIG_H
