//
// Created by aleks on 1.3.2025.
//

// 001_Symbolic_regression.cpp
#include "../ExampleRunner.h"
#include "../LogHelper.h"
#include <iostream>

#include "../../src/nodes/functions/AddOperator.h"
#include "../../src/nodes/functions/MultiplyOperator.h"
#include "../../src/nodes/terminals/VariableNode.h"
#include "../../src/nodes/terminals/ConstNode.h"
#include "../../src/problems/SymbolicRegressionProblem.h"
#include "../../src/tasks/Task.h"

/**
 * PROBLEM DEFINITION: Find the function f(x) = x^2 + 2x + 1 using symbolic regression.
 * The function is defined in the SymbolicRegressionProblem class.
 */

REGISTER_PROGRAM(001_Symbolic_regression_find_function) {
    LogHelper::logMessage("Running symbolic regression program...");

    // Create a new task
    Task symbolicRegressionTask("Symbolic regression");

    // Create a new problem
    SymbolicRegressionProblem problem("Find function", StopCriterion::EVALUATIONS, 1, {10.0}, {-10.0}, ObjectiveType::MINIMIZE);
    problem.setStopCritMaxEvaluations(10000);

    problem.setFunctionSet({
        []() { return std::make_unique<AddOperator>(); },
        []() { return std::make_unique<MultiplyOperator>(); },
    });

    problem.setTerminalSet({
        new VariableNode("x"),
        new ConstNode(),
    });




    // Add the problem to the program
    symbolicRegressionTask.setProblem(&problem);

    // Run the program
    symbolicRegressionTask.run();

    LogHelper::logMessage("Symbolic regression program finished.");


}