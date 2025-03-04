cmake_minimum_required(VERSION 3.30)
project(GEEP LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

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
        src/criterions/StopCriterion.h
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
        src/problems/SymbolicRegressionProblem.h
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
)

set_target_properties(GEEP PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# Link Qt
target_link_libraries(GEEP PRIVATE Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Widgets)

# Include Qt directories
target_include_directories(GEEP PRIVATE
        ${Qt${QT_VERSION_MAJOR}_INCLUDE_DIRS}
)