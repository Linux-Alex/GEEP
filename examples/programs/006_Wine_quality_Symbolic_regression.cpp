//
// Created by aleks on 31. 01. 26.
//


#include "../ExampleRunner.h"
#include "../LogHelper.h"
#include <iostream>

#include "../../src/nodes/functions/AddOperator.h"
#include "../../src/nodes/functions/MultiplyOperator.h"
#include "../../src/nodes/terminals/VariableNode.h"
#include "../../src/nodes/terminals/ConstNode.h"
#include "../../src/nodes/functions/DivideOperator.h"
#include "../../src/nodes/functions/SubtractOperator.h"
#include "../../src/problems/SymbolicRegressionProblem.cuh"
#include "../../src/selections/TournamentSelection.h"
#include "../../src/tasks/Task.h"
#include "../../src/crossover/SubtreeCrossover.h"
#include "../../src/mutation/Mutation.h"

/**
 * PROBLEM DEFINITION: Find the equation for wine quality assessment.
 * The function is defined in the SymbolicRegressionProblem class.
 */

REGISTER_PROGRAM(006_Wine_quality_Symbolic_regression) {
    // Create a new task
    Task symbolicRegressionTask("Symbolic regression");

    // Create a new problem
    SymbolicRegressionProblem problem = SymbolicRegressionProblem("Find equation for wine quality assessment prediction")
        .setStopCrit(StopCriterion().addCriterion(GENERATIONS, 100000)/*.addCriterion(GENERATIONS, 1000)*/)
        .setElitism(5)
        .setSelection(new TournamentSelection(9))
        .setCrossover(&(new SubtreeCrossover())->setReproductionRate(0.5f))
        .setMutation(new Mutation(0.5f))
        .setMaxDepth(5)
        .setMaxNodes(31)
        .setPopulationSize(1000);

    problem.setFunctionSet({
        []() { return new AddOperator(); },
        []() { return new MultiplyOperator(); },
        []() { return new DivideOperator(); },
        []() { return new SubtractOperator(); },
    });

    problem.setTerminalSet({
        []() { return new VariableNode("fixed acidity"); },
        []() { return new VariableNode("volatile acidity"); },
        []() { return new VariableNode("citric acid"); },
        []() { return new VariableNode("residual sugar"); },
        []() { return new VariableNode("chlorides"); },
        []() { return new VariableNode("free sulfur dioxide"); },
        []() { return new VariableNode("total sulfur dioxide"); },
        []() { return new VariableNode("density"); },
        []() { return new VariableNode("pH"); },
        []() { return new VariableNode("sulphates"); },
        []() { return new VariableNode("alcohol"); },
        []() { return new ConstNode(); },
    });

    problem.setTargets(Target::readTargetsFromCSV("/home/aleks/GEEP/GEEP/examples/data/Wine Quality Red.csv", ';', "quality"));

    // Add the problem to the program
    symbolicRegressionTask.setProblem(&problem);

    try {
        // Run the task on CPU
        // symbolicRegressionTask.setExecutionMode(Task::ExecutionMode::CPU).run();

        // Run the task on GPU
        symbolicRegressionTask.setExecutionMode(Task::ExecutionMode::GPU).run();
    } catch (std::exception &e) {
        LogHelper::logMessage("Error running symbolic regression program: " + std::string(e.what()), true);
        return;
    }

    LogHelper::logMessage("Symbolic regression program finished.");
}