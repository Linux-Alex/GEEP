//
// Created by aleks on 1.3.2025.
//

#include "../ExampleRunner.h"
#include "../LogHelper.h"
#include <iostream>

#include "../../src/nodes/functions/AddOperator.h"
#include "../../src/nodes/functions/MultiplyOperator.h"
#include "../../src/nodes/terminals/VariableNode.h"
#include "../../src/nodes/terminals/ConstNode.h"
#include "../../src/problems/SymbolicRegressionProblem.h"
#include "../../src/selections/TournamentSelection.h"
#include "../../src/tasks/Task.h"
#include "../../src/crossover/SubtreeCrossover.h"

/**
 * PROBLEM DEFINITION: Find the function f(x) = x^2 + 2x + 1 using symbolic regression.
 * The function is defined in the SymbolicRegressionProblem class.
 */

REGISTER_PROGRAM(001_Symbolic_regression_find_function) {
    LogHelper::logMessage("Running symbolic regression program...");

    // Create a new task
    Task symbolicRegressionTask("Symbolic regression");

    // Create a new problem
    SymbolicRegressionProblem problem = SymbolicRegressionProblem("Find function by target data")
        .setStopCrit(StopCriterion().addCriterion(GENERATIONS, 10000))
        .setElitism(5)
        .setSelection(new TournamentSelection(3))
        .setCrossover(new SubtreeCrossover())
        .setPopulationSize(100);

    problem.setFunctionSet({
        []() { return new AddOperator(); },
        []() { return new MultiplyOperator(); },
    });

    problem.setTerminalSet({
        []() { return new VariableNode("x"); },
        []() { return new ConstNode(); },
    });

    problem.setTargets({
        Target().setCondition("x", 1.0).setTargetValue(4.0),
        Target().setCondition("x", 2.0).setTargetValue(9.0),
        Target().setCondition("x", 3.0).setTargetValue(16.0),
        Target().setCondition("x", 4.0).setTargetValue(25.0),
        Target().setCondition("x", 5.0).setTargetValue(36.0),
    });

    // Add the problem to the program
    symbolicRegressionTask.setProblem(&problem);

    try {
        // Run the program
        symbolicRegressionTask.run();
    } catch (std::exception &e) {
        LogHelper::logMessage("Error running symbolic regression program: " + std::string(e.what()), true);
        return;
    }

    LogHelper::logMessage("Symbolic regression program finished.");
}