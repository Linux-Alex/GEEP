cmake_minimum_required(VERSION 3.30)
project(GEEP LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# For CUDA files
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_61 --ptxas-options=-v")

# Find Qt
set(CMAKE_PREFIX_PATH "/opt/Qt/6.8.2/gcc_64/lib/cmake")
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Widgets)

add_executable(GEEP src/main.cpp
        src/GEEPConfig.cpp
        src/GEEPConfig.h
        src/nodes/Node.cpp
        src/nodes/Node.h
        src/nodes/FunctionNode.cpp
        src/nodes/FunctionNode.h
        src/nodes/functions/AddOperator.cpp
        src/nodes/functions/AddOperator.h
        src/solutions/Solution.cpp
        src/solutions/Solution.h
        src/criterions/StopCriterionType.h
        src/tasks/Task.cpp
        src/tasks/Task.h
        src/problems/Problem.cpp
        src/problems/Problem.h
        src/problems/ObjectiveType.h
        examples/programs/001_Symbolic_regression_find_function.cpp
        examples/ExampleRunner.cpp
        examples/ExampleRunner.h
        examples/LogHelper.cpp
        examples/LogHelper.h
        examples/programs/002_Symbolic_regression.cpp
        src/problems/SymbolicRegressionProblem.cpp
        src/problems/SymbolicRegressionProblem.cuh
        src/problems/SymbolicRegressionProblem.cu
        src/selections/SelectionMethod.h
        src/nodes/terminals/ConstNode.cpp
        src/nodes/terminals/ConstNode.h
        src/nodes/TerminalNode.cpp
        src/nodes/TerminalNode.h
        src/nodes/terminals/VariableNode.cpp
        src/nodes/terminals/VariableNode.h
        src/nodes/functions/MultiplyOperator.cpp
        src/nodes/functions/MultiplyOperator.h
        src/helpers/NumberConversions.cpp
        src/helpers/NumberConversions.h
        src/targets/Target.cpp
        src/targets/Target.h
        src/cuda/CudaUtils.cu
        src/cuda/CudaUtils.h
        src/nodes/FunctionFactory.h
        src/nodes/TerminalFactory.h
        src/criterions/StopCriterion.cpp
        src/criterions/StopCriterion.h
        src/selections/Selection.cpp
        src/selections/Selection.h
        src/selections/TournamentSelection.cpp
        src/selections/TournamentSelection.h
        src/selections/TournamentSelection.cu
        src/selections/RouletteWheelSelection.cpp
        src/selections/RouletteWheelSelection.h
        src/crossover/Crossover.cpp
        src/crossover/Crossover.h
        src/crossover/SubtreeCrossover.cpp
        src/crossover/SubtreeCrossover.h
        src/crossover/SubtreeCrossover.cu
        src/nodes/functions/DivideOperator.cpp
        src/nodes/functions/DivideOperator.h
        src/nodes/functions/SubtractOperator.cpp
        src/nodes/functions/SubtractOperator.h
        src/cuda/GPUTree.h
        src/cuda/GPUTree.cu
        examples/programs/003_Minimal_example_for_symbolic_regression.cpp
)

set_target_properties(GEEP PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# Google Test Integration
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)

# Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable CUDA separable compilation
enable_testing()

# Test executable
add_executable(GEEPTests src/tests/main_test.cpp
        src/tests/SymbolicRegressionGPUTest.cu
        src/tests/SymbolicRegressionGPUTest.h
        src/problems/SymbolicRegressionProblem.cuh
        src/problems/SymbolicRegressionProblem.cu
        src/problems/SymbolicRegressionProblem.cpp
        src/problems/Problem.h
        src/problems/Problem.cpp
        src/solutions/Solution.h
        src/solutions/Solution.cpp
        src/nodes/Node.h
        src/nodes/Node.cpp
        src/nodes/FunctionNode.h
        src/nodes/FunctionNode.cpp
        src/nodes/terminals/ConstNode.h
        src/nodes/terminals/ConstNode.cpp
        src/nodes/TerminalNode.h
        src/nodes/TerminalNode.cpp
        src/nodes/terminals/VariableNode.h
        src/nodes/terminals/VariableNode.cpp
        src/nodes/functions/AddOperator.h
        src/nodes/functions/AddOperator.cpp
        src/nodes/functions/MultiplyOperator.h
        src/nodes/functions/MultiplyOperator.cpp
        src/nodes/functions/DivideOperator.h
        src/nodes/functions/DivideOperator.cpp
        src/nodes/functions/SubtractOperator.h
        src/nodes/functions/SubtractOperator.cpp
        src/criterions/StopCriterionType.h
        src/criterions/StopCriterion.h
        src/criterions/StopCriterion.cpp
        src/targets/Target.h
        src/targets/Target.cpp
        src/cuda/CudaUtils.h
        src/cuda/CudaUtils.cu
        src/cuda/GPUTree.h
        src/cuda/GPUTree.cu
        src/selections/SelectionMethod.h
        src/selections/Selection.h
        src/selections/Selection.cpp
        src/selections/TournamentSelection.h
        src/selections/TournamentSelection.cpp
        src/selections/TournamentSelection.cu
        src/crossover/Crossover.h
        src/crossover/Crossover.cpp
        src/crossover/SubtreeCrossover.h
        src/crossover/SubtreeCrossover.cpp
        src/crossover/SubtreeCrossover.cu
        src/helpers/NumberConversions.h
        src/helpers/NumberConversions.cpp
        src/tests/SubtreeCrossoverTest.cu
        src/tests/SubtreeCrossoverTest.h
)

# Link GoogleTest and your app code as needed
target_link_libraries(GEEPTests
        PRIVATE
        GTest::gtest
        GTest::gtest_main
        pthread
)

# Include directories for GTest and the project
target_include_directories(GEEPTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Discover and run tests automatically
include(GoogleTest)
gtest_discover_tests(GEEPTests)

# Link Qt
target_link_libraries(GEEP PRIVATE Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(GEEPTests PRIVATE Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Widgets)

# Include Qt directories
target_include_directories(GEEP PRIVATE
        ${Qt${QT_VERSION_MAJOR}_INCLUDE_DIRS}
)
target_include_directories(GEEPTests PRIVATE
        ${Qt${QT_VERSION_MAJOR}_INCLUDE_DIRS}
)