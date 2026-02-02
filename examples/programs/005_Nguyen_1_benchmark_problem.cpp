//
// Created by aleks on 1. 12. 25.
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
 * PROBLEM DEFINITION: Find the function Nguyen-1: f(x) = x^2 + x + 1 using symbolic regression.
 * The function is defined in the SymbolicRegressionProblem class.
 */

REGISTER_PROGRAM(005_Nguyen_1_benchmark_problem) {

    // Create a new task
    Task symbolicRegressionTask("Nguyen 2 Symbolic regression");

    // Create a new problem
    SymbolicRegressionProblem problem = SymbolicRegressionProblem("Find function by target data")
        // .setStopCrit(StopCriterion().addCriterion(GENERATIONS, 1000))
        .setStopCrit(StopCriterion().addCriterion(MSE, 0))
        .setElitism(0)
        .setSelection(new TournamentSelection(3))
        .setCrossover(&(new SubtreeCrossover())->setReproductionRate(0.02f))
        .setMutation(new Mutation(0.02f))
        .setMaxDepth(5)
        .setMaxNodes(31)
        .setPopulationSize(200);

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

    // Generate target data for Nguyen-1: f(x) = x^2 + x + 1 for x in [1 .. 20 by step 1]
    std::vector<Target> targets;
    for (int x = 1; x <= 20; x++) {
        double y =  (x * x * x * x * x) + (x * x * x * x) + (x * x * x) + (x * x) + x + 1;
        targets.push_back(Target().setCondition("x", static_cast<double>(x)).setTargetValue(y));
    }

    problem.setTargets(targets);


    // Add the problem to the program
    symbolicRegressionTask.setProblem(&problem);

    try {
        // Run the task on CPU
        // symbolicRegressionTask.setExecutionMode(Task::ExecutionMode::CPU).run();

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