//
// Created by aleks on 26.4.2025.
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

/**
 * PROBLEM DEFINITION: Find the function f(x) = x * 2 using symbolic regression.
 * The function is defined in the SymbolicRegressionProblem class.
 */

REGISTER_PROGRAM(003_Minimal_example_for_symbolic_regression) {
    // LogHelper::logMessage("Running symbolic regression program...");

    // Create a new task
    Task symbolicRegressionTask("Symbolic regression");

    // Create a new problem
    SymbolicRegressionProblem problem = SymbolicRegressionProblem("Find function by target data")
        .setStopCrit(StopCriterion().addCriterion(GENERATIONS, 2))
        // .setElitism(5)
        .setSelection(new TournamentSelection(3))
        .setCrossover(new SubtreeCrossover())
        .setMaxDepth(5)
        .setMaxNodes(16)
        .setPopulationSize(1);

    problem.setFunctionSet({
        []() { return new AddOperator(); },
        []() { return new MultiplyOperator(); },
        []() { return new DivideOperator(); },
        []() { return new SubtractOperator(); },
    });

    problem.setTerminalSet({
        []() { return new VariableNode("x"); },
        []() { return new ConstNode(); },
    });

    problem.setTargets({
        Target().setCondition("x", 1.0).setTargetValue(4.0),
        Target().setCondition("x", 2.0).setTargetValue(9.0)
    });

    // Add the problem to the program
    symbolicRegressionTask.setProblem(&problem);

    try {
        // Run the task on CPU
        symbolicRegressionTask.setExecutionMode(Task::ExecutionMode::CPU).run();

        // Run the task on GPU
        symbolicRegressionTask.setExecutionMode(Task::ExecutionMode::GPU).run();

        // Test GPU computing
        // symbolicRegressionTask.testGPUComputing();
    } catch (std::exception &e) {
        LogHelper::logMessage("Error running symbolic regression program: " + std::string(e.what()), true);
        return;
    }

    LogHelper::logMessage("Symbolic regression program finished.");
}