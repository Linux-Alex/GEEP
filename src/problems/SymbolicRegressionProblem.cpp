//
// Created by aleks on 2.3.2025.
//

#include "SymbolicRegressionProblem.cuh"
#include <stdexcept>

#include "../nodes/functions/AddOperator.h"
#include "../nodes/functions/MultiplyOperator.h"
#include "../nodes/terminals/ConstNode.h"
#include "../nodes/terminals/VariableNode.h"
#include "../tasks/Task.h"

float SymbolicRegressionProblem::evaluate(Solution *solution) {
    const std::vector<double>& values = solution->getValues();

    float fitness = 0.0;

    for (const Target& target : this->targets) {
        // Target input state
        const std::map<std::string, double>& state = target.getState();

        if (solution->getRoot() == nullptr) {
            continue;
        }

        float result = 0.0;
        // Evaluate the solution
        try {
            result = solution->getRoot()->evaluate(state);
        }
        catch (const std::exception& e) {
            // If evaluation fails, assign a high error
            fitness += 1e6;
            continue;
        }
        // Calculate the difference (error)
        float error = result - target.getTargetValue();

        // Add the square of the error to the fitness
        fitness += error * error;
    }

    // Return the fitness (lower is better)
    return fitness / (float)this->targets.size();
}

void SymbolicRegressionProblem::setTargets(const std::vector<Target> &targets) { this->targets = targets; }



void SymbolicRegressionProblem::testGPUEvaluation() {
    Solution * s = new Solution();

    AddOperator a, b;
    MultiplyOperator m;

    m.addChild(new VariableNode("x"));
    m.addChild(new VariableNode("x"));

    b.addChild(new VariableNode("x"));
    b.addChild(new ConstNode(1.0f));

    a.addChild(m.clone());
    a.addChild(b.clone());

    s->setRoot(a.clone());

    GPUTree tree;
    tree.allocate(8, 1);

    tree.addSolution(0, s);

    std::cout << "Node count: " << tree.node_counts[0] << std::endl;


    // Print tree nodes
    std::cout << "Tree nodes: ";
    for (size_t i = 0; i < tree.node_counts[0]; i++) {
        std::cout << static_cast<int>(tree.nodes[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << "Tree values: ";
    for (size_t i = 0; i < tree.node_counts[0]; i++) {
        std::cout << tree.values[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Tree children: ";
    for (size_t i = 0; i < tree.node_counts[0] * 2; i++) {
        std::cout << tree.children[i] << " ";
    }
    std::cout << std::endl;

    std::vector<Target> targets;
    for (int x = 1; x <= 20; x++) {
        double y = x * x + x + 1;
        targets.push_back(Target().setCondition("x", static_cast<double>(x)).setTargetValue(y));
    }

    Task t = Task("Test task");
    SymbolicRegressionProblem p = SymbolicRegressionProblem("Test problem");
    p.setTargets(targets);

    p.prepareTargetData();
    p.gpuEvaluate(tree);

    LogHelper::logMessage("Fitness from GPU evaluation: " + std::to_string(tree.fitness_values[0]));
}
