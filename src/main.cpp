#include <iostream>
#include <QCoreApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QFile>
#include <QXmlStreamReader>

#include "../examples/ExampleRunner.h"
#include "cuda/CudaUtils.h"
#include "service/GEEPService.h"


int main(int argc, char *argv[]) {
    QCoreApplication a(argc, argv);
    QCoreApplication::setApplicationName("GEEP");
    QCoreApplication::setApplicationVersion("0.1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("GEEP Framework - Genetic Evolutionary Engineering Platform");
    parser.addHelpOption();
    parser.addVersionOption();

    // Options
    parser.addOption(QCommandLineOption("service", "Start as a service (runs forever)."));
    parser.addOption(QCommandLineOption("input", "Input XML file.", "file"));
    parser.addOption(QCommandLineOption("output-dir", "Specify the output directory.", "directory"));
    parser.addOption(QCommandLineOption("server-port", "Specify the port number.", "port", "8096"));
    parser.addOption(QCommandLineOption("example", "Run an example program.", "program"));
    parser.addOption(QCommandLineOption("example-list", "List examples by type: all|program|xml.", "type", "all"));
    parser.addOption(QCommandLineOption("check-cuda", "Check for CUDA support."));
    parser.addOption(QCommandLineOption("program", "Specify the program name.", "own-program"));

    parser.process(a);

    // Check for CUDA support with verbose output
    if (parser.isSet("check-cuda")) {
        if (!hasCudaSupport(true)) {
            std::cerr << "CUDA is NOT supported. Exiting." << std::endl;
            return 1;
        }
    }

    // Handle options
    if (parser.isSet("service")) {
        quint16 port = parser.value("server-port").toUShort();

        GEEPService service;
        if (!service.startService(port)) {
            return 1;
        }

        std::cout << "GEEP Service is running. Press Ctrl+C to stop." << std::endl;

        // Keep the application running
        return a.exec();
    } else if (parser.isSet("input")) {
        QString inputFile = parser.value("input");
        GEEPService::processXmlFile(inputFile);
    }

    QString outputDir = parser.value("output-dir");
    if (outputDir.isEmpty()) {
        outputDir = QDir::currentPath(); // Default to current directory
    } else {
        QDir dir(outputDir);
        if (!dir.exists()) {
            dir.mkpath("."); // Create the directory if it doesn't exist
        }
    }

    // If the example option is set, run the example program
    if (parser.isSet("example")) {
        try {
            ExampleRunner::run(parser.value("example").toStdString());
            return 0;
        }
        catch (const std::exception& e) {
            std::cerr << "Error running example: " << e.what() << std::endl;
            return 1;
        }
    }

    // If the example option is set, run the own program
    if (parser.isSet("program")) {
        try {
            ExampleRunner::run(parser.value("program").toStdString());
            return 0;
        }
        catch (const std::exception& e) {
            std::cerr << "Error running own program: " << e.what() << std::endl;
            return 1;
        }
    }

    // If the example-list option is set, list the examples
    if (parser.isSet("example-list")) {
        std::string type = parser.value("example-list").toStdString();
        if (type == "all") {
            ExampleRunner::listAll();
        } else if (type == "program") {
            ExampleRunner::listAll(true, false);
        } else if (type == "xml") {
            ExampleRunner::listAll(false, true);
        } else {
            std::cerr << "Invalid example list type: " << type << std::endl;
        }
        return 0;
    }


    return 0;
}
