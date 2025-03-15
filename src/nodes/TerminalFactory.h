//
// Created by aleks on 15.3.2025.
//

#ifndef TERMINALFACTORY_H
#define TERMINALFACTORY_H

#include <memory>
#include <functional>

class TerminalNode;

using TerminalFactory = std::function<std::unique_ptr<TerminalNode>()>;

#endif //TERMINALFACTORY_H
