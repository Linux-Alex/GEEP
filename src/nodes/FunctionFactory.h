//
// Created by aleks on 7.3.2025.
//

#ifndef FUNCTIONFACTORY_H
#define FUNCTIONFACTORY_H

#include <memory>
#include <functional>

class FunctionNode;

using FunctionFactory = std::function<std::unique_ptr<FunctionNode>()>;

#endif //FUNCTIONFACTORY_H
