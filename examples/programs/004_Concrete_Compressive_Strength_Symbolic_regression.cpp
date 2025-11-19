//
// Created by aleks on 16. 11. 25.
//

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
 * PROBLEM DEFINITION: Find the equation for Contrete Compressive Strength using symbolic regression.
 * The function is defined in the SymbolicRegressionProblem class.
 */

REGISTER_PROGRAM(004_Concrete_Compressive_Strength_Symbolic_regression) {
    // Create a new task
    Task symbolicRegressionTask("Symbolic regression");

    // Create a new problem
    SymbolicRegressionProblem problem = SymbolicRegressionProblem("Find equation for Concrete Compressive Strength")
        .setStopCrit(StopCriterion().addCriterion(GENERATIONS, 300))
        // .setElitism(5)
        .setSelection(new TournamentSelection(9))
        .setCrossover(&(new SubtreeCrossover())->setReproductionRate(0.02f))
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
        []() { return new VariableNode("Cement"); },
        []() { return new VariableNode("Blast Furnace Slag"); },
        []() { return new VariableNode("Fly Ash"); },
        []() { return new VariableNode("Water"); },
        []() { return new VariableNode("Superplasticizer"); },
        []() { return new VariableNode("Coarse Aggregate"); },
        []() { return new VariableNode("Fine Aggregate"); },
        []() { return new VariableNode("Age"); },
        []() { return new ConstNode(); },
    });

    problem.setTargets(Target::readTargetsFromCSV("/home/aleks/GEEP/GEEP/examples/data/Concrete Compressive Strength Data.csv", ',', "Concrete compressive strength"));

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