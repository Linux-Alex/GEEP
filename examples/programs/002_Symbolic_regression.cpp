//
// Created by aleks on 1.3.2025.
//

//
// Created by aleks on 1.3.2025.
//

// 001_Symbolic_regression.cpp
#include "../ExampleRunner.h"
#include "../LogHelper.h"
#include <iostream>
#include "../../src/tasks/Task.h"


REGISTER_PROGRAM(002_Symbolic_regression_minimize) {
    LogHelper::logMessage("Running symbolic regression program...");

    // Create a new task
    Task symbolicRegressionTask("Symbolic regression");

    // Create a new problem
    Problem problem("Symbolic regression", StopCriterion(), 1, {10.0}, {-10.0}, ObjectiveType::MINIMIZE);

    // Add the problem to the program
    symbolicRegressionTask.setProblem(&problem);

    // Run the program
    symbolicRegressionTask.run();

    LogHelper::logMessage("Symbolic regression program finished.");


}