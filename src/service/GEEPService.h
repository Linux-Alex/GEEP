//
// Created by aleks on 30. 11. 25.
//

#ifndef GEEP_GEEPSERVICE_H
#define GEEP_GEEPSERVICE_H

#include <iostream>
#include <QCoreApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QFile>
#include <QXmlStreamReader>
#include <QHttpServer>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDateTime>
#include <QTcpServer>
#include <QElapsedTimer>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <map>
#include <thread>
#include <QJsonArray>  // Add this line
#include <QUuid>       // Add this for UUID generation

#include "../cuda/CudaUtils.h"
#include "../solutions/Solution.h"
#include "XMLParser.h"

struct GARequest {
    QString id;
    QString status; // "pending", "running", "completed", "error"
    QDateTime submittedAt;
    QDateTime startedAt;
    QDateTime completedAt;
    std::string bestSolution;
    float bestFitness;
    size_t generationsRun;
    std::string errorMessage;

    // Default constructor
    GARequest() : status("pending"), bestFitness(0.0f), generationsRun(0) {
        submittedAt = QDateTime::currentDateTime();
    }

    // Parameterized constructor
    GARequest(const QString& requestId)
        : id(requestId), status("pending"), bestFitness(0.0f), generationsRun(0) {
        submittedAt = QDateTime::currentDateTime();
    }
};

class GEEPService : public QObject {
    Q_OBJECT

private:
    QHttpServer* httpServer;
    QTcpServer* tcpServer;
    bool isRunning;

    // Request management
    std::map<QString, GARequest> requests;
    QMutex requestsMutex;
    QWaitCondition requestCondition;
    bool stopWorkerThread = false;
    std::thread workerThread;

public:
    GEEPService(QObject* parent = nullptr) : QObject(parent), httpServer(nullptr), tcpServer(nullptr), isRunning(false) {
        // Start the worker thread for processing requests
        workerThread = std::thread(&GEEPService::processRequestsWorker, this);
    }

    ~GEEPService() {
        stopService();
    }

    bool startService(quint16 port = 8096) {
        if (isRunning) {
            std::cout << "Service is already running." << std::endl;
            return true;
        }

        httpServer = new QHttpServer(this);
        tcpServer = new QTcpServer(this);

        // Setup routes
        setupRoutes();

        // Start the TCP server
        if (!tcpServer->listen(QHostAddress::Any, port)) {
            std::cerr << "Failed to start server on port " << port << ": "
                      << tcpServer->errorString().toStdString() << std::endl;
            return false;
        }

        // Bind the HTTP server to the TCP server
        if (!httpServer->bind(tcpServer)) {
            std::cerr << "Failed to bind HTTP server to TCP server" << std::endl;
            return false;
        }

        isRunning = true;
        std::cout << "GEEP Service started on port " << tcpServer->serverPort() << std::endl;
        std::cout << "Available endpoints:" << std::endl;
        std::cout << "  POST /api/run-configuration - Submit genetic algorithm configuration" << std::endl;
        std::cout << "  GET  /api/requests - List all requests" << std::endl;
        std::cout << "  GET  /api/requests/{id} - Get request status and results" << std::endl;
        std::cout << "  GET  /api/status - Get service status" << std::endl;
        std::cout << "  GET  /api/stop - Stop the service" << std::endl;

        return true;
    }

    void stopService() {
        // Stop worker thread
        {
            QMutexLocker locker(&requestsMutex);
            stopWorkerThread = true;
            requestCondition.wakeAll();
        }

        if (workerThread.joinable()) {
            workerThread.join();
        }

        if (tcpServer) {
            tcpServer->close();
            tcpServer->deleteLater();
            tcpServer = nullptr;
        }
        if (httpServer) {
            httpServer->deleteLater();
            httpServer = nullptr;
        }
        isRunning = false;
        std::cout << "GEEP Service stopped." << std::endl;
    }

    static void processXmlFile(const QString& filePath) {
        QFile file(filePath);
        if (!file.open(QFile::ReadOnly | QFile::Text)) {
            std::cerr << "Cannot read file: " << filePath.toStdString() << std::endl;
            return;
        }

        QByteArray xmlData = file.readAll();
        file.close();

        XMLParser parser;
        try {
            if (parser.parseAndRun(QString::fromUtf8(xmlData))) {
                std::cout << "Result - Fitness: " << parser.getBestFitness()
                          << ", Solution: " << parser.getBestSolution()
                          << ", Generations: " << parser.getGenerationsRun() << std::endl;
            } else {
                std::cerr << "Failed to parse and run XML configuration" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing XML: " << e.what() << std::endl;
        }
    }

private:
    void setupRoutes() {
        // Health check endpoint
        httpServer->route("/api/status", QHttpServerRequest::Method::Get,
            [this](const QHttpServerRequest& request) {
                QJsonObject status;
                status["service"] = "GEEP Genetic Algorithm Service";
                status["status"] = "running";
                status["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
                status["cuda_supported"] = hasCudaSupport(false);
                status["port"] = tcpServer->serverPort();
                status["pending_requests"] = getPendingRequestCount();

                return QHttpServerResponse(QJsonDocument(status).toJson(), QHttpServerResponse::StatusCode::Ok);
            });

        // Submit new configuration
        httpServer->route("/api/run-configuration", QHttpServerRequest::Method::Post,
            [this](const QHttpServerRequest& request) {
                return handleSubmitConfiguration(request);
            });

        // List all requests
        httpServer->route("/api/requests", QHttpServerRequest::Method::Get,
            [this](const QHttpServerRequest& request) {
                return handleListRequests(request);
            });

        // Get specific request status
        httpServer->route("/api/requests/<arg>", QHttpServerRequest::Method::Get,
            [this](const QString& requestId, const QHttpServerRequest& request) {
                return handleGetRequest(requestId, request);
            });

        // Stop service endpoint
        httpServer->route("/api/stop", QHttpServerRequest::Method::Get,
            [this](const QHttpServerRequest& request) {
                QJsonObject response;
                response["message"] = "Service stopping...";
                QCoreApplication::quit();
                return QHttpServerResponse(QJsonDocument(response).toJson(), QHttpServerResponse::StatusCode::Ok);
            });

        // Default route
        httpServer->route("/", QHttpServerRequest::Method::Get,
            [](const QHttpServerRequest& request) {
                QJsonObject info;
                info["message"] = "GEEP Genetic Evolutionary Engineering Platform";
                info["version"] = "0.1.0";
                info["endpoints"] = "Use /api/run-configuration to submit XML configurations";

                return QHttpServerResponse(QJsonDocument(info).toJson(), QHttpServerResponse::StatusCode::Ok);
            });
    }

    QHttpServerResponse handleSubmitConfiguration(const QHttpServerRequest& request) {
        std::cout << "Received configuration submission" << std::endl;

        // Check content type
        auto contentType = request.value("content-type");
        if (!contentType.contains("application/xml") &&
            !contentType.contains("text/xml")) {
            QJsonObject error;
            error["error"] = "Content-Type must be application/xml or text/xml";
            return QHttpServerResponse(QJsonDocument(error).toJson(),
                                       QHttpServerResponse::StatusCode::BadRequest);
        }

        // Get XML data from request body
        QByteArray xmlData = request.body();
        if (xmlData.isEmpty()) {
            QJsonObject error;
            error["error"] = "Empty XML configuration";
            return QHttpServerResponse(QJsonDocument(error).toJson(),
                                       QHttpServerResponse::StatusCode::BadRequest);
        }

        // Generate unique request ID
        QString requestId = QUuid::createUuid().toString(QUuid::WithoutBraces);

        {
            QMutexLocker locker(&requestsMutex);
            // Store the request
            GARequest gaRequest(requestId);
            gaRequest.status = "pending";
            requests[requestId] = gaRequest;

            // Store XML data temporarily (you might want to store this differently)
            // For now, we'll process it immediately in the worker thread
        }

        // Notify worker thread
        requestCondition.wakeOne();

        // Return immediate response with request ID
        QJsonObject response;
        response["status"] = "submitted";
        response["request_id"] = requestId;
        response["message"] = "Request submitted and queued for processing";

        std::cout << "Request submitted with ID: " << requestId.toStdString() << std::endl;

        return QHttpServerResponse(QJsonDocument(response).toJson(),
                                   QHttpServerResponse::StatusCode::Accepted);
    }

    QHttpServerResponse handleListRequests(const QHttpServerRequest& request) {
        QMutexLocker locker(&requestsMutex);

        QJsonArray requestsArray;
        for (const auto& [id, gaRequest] : requests) {
            QJsonObject requestObj;
            requestObj["id"] = gaRequest.id;
            requestObj["status"] = gaRequest.status;
            requestObj["submitted_at"] = gaRequest.submittedAt.toString(Qt::ISODate);

            if (!gaRequest.startedAt.isNull()) {
                requestObj["started_at"] = gaRequest.startedAt.toString(Qt::ISODate);
            }
            if (!gaRequest.completedAt.isNull()) {
                requestObj["completed_at"] = gaRequest.completedAt.toString(Qt::ISODate);
            }

            requestsArray.append(requestObj);
        }

        QJsonObject response;
        response["requests"] = requestsArray;
        response["total"] = static_cast<int>(requests.size());

        return QHttpServerResponse(QJsonDocument(response).toJson(),
                                   QHttpServerResponse::StatusCode::Ok);
    }

    QHttpServerResponse handleGetRequest(const QString& requestId, const QHttpServerRequest& request) {
        QMutexLocker locker(&requestsMutex);

        auto it = requests.find(requestId);
        if (it == requests.end()) {
            QJsonObject error;
            error["error"] = "Request not found";
            return QHttpServerResponse(QJsonDocument(error).toJson(),
                                       QHttpServerResponse::StatusCode::NotFound);
        }

        const GARequest& gaRequest = it->second;
        QJsonObject response;
        response["id"] = gaRequest.id;
        response["status"] = gaRequest.status;
        response["submitted_at"] = gaRequest.submittedAt.toString(Qt::ISODate);

        if (!gaRequest.startedAt.isNull()) {
            response["started_at"] = gaRequest.startedAt.toString(Qt::ISODate);
        }
        if (!gaRequest.completedAt.isNull()) {
            response["completed_at"] = gaRequest.completedAt.toString(Qt::ISODate);
        }

        if (gaRequest.status == "completed") {
            response["best_fitness"] = gaRequest.bestFitness;
            response["best_solution"] = QString::fromStdString(gaRequest.bestSolution);
            response["generations_run"] = static_cast<int>(gaRequest.generationsRun);
        } else if (gaRequest.status == "error") {
            response["error"] = QString::fromStdString(gaRequest.errorMessage);
        }

        return QHttpServerResponse(QJsonDocument(response).toJson(),
                                   QHttpServerResponse::StatusCode::Ok);
    }

    void processRequestsWorker() {
        while (true) {
            QString requestId;

            {
                QMutexLocker locker(&requestsMutex);

                // Check if we should stop
                if (stopWorkerThread) {
                    break;
                }

                // Find a pending request
                for (auto& [id, request] : requests) {
                    if (request.status == "pending") {
                        requestId = id;
                        request.status = "running";
                        request.startedAt = QDateTime::currentDateTime();
                        break;
                    }
                }

                // If no pending requests, wait
                if (requestId.isEmpty()) {
                    requestCondition.wait(&requestsMutex);
                    continue;
                }
            }

            // Process the request (outside the lock)
            if (!requestId.isEmpty()) {
                processSingleRequest(requestId);
            }
        }
    }

    void processSingleRequest(const QString& requestId) {
        std::cout << "Processing request: " << requestId.toStdString() << std::endl;

        // For now, we'll use a simple XML string
        // In a real implementation, you'd retrieve the stored XML data
        std::string dummyXml = R"(<?xml version="1.0"?>
<genetic_algorithm>
    <problem_type>symbolic_regression</problem_type>
    <problem_name>Test Request</problem_name>
    <stopping_criteria>
        <criterion type="generations">50</criterion>
    </stopping_criteria>
    <algorithm_parameters>
        <population_size>50</population_size>
        <elitism_count>2</elitism_count>
        <max_depth>4</max_depth>
        <max_nodes>15</max_nodes>
    </algorithm_parameters>
    <!-- ... other parameters ... -->
</genetic_algorithm>)";

        try {
            XMLParser parser;
            if (parser.parseAndRun(QString::fromStdString(dummyXml))) {
                // Update request with results
                QMutexLocker locker(&requestsMutex);
                auto it = requests.find(requestId);
                if (it != requests.end()) {
                    it->second.status = "completed";
                    it->second.completedAt = QDateTime::currentDateTime();
                    it->second.bestFitness = parser.getBestFitness();
                    it->second.bestSolution = parser.getBestSolution();
                    it->second.generationsRun = parser.getGenerationsRun();
                }

                std::cout << "Request completed: " << requestId.toStdString() << std::endl;
            } else {
                throw std::runtime_error("XMLParser failed to run");
            }
        } catch (const std::exception& e) {
            QMutexLocker locker(&requestsMutex);
            auto it = requests.find(requestId);
            if (it != requests.end()) {
                it->second.status = "error";
                it->second.completedAt = QDateTime::currentDateTime();
                it->second.errorMessage = e.what();
            }

            std::cerr << "Request failed: " << requestId.toStdString() << " - " << e.what() << std::endl;
        }
    }

    int getPendingRequestCount() {
        QMutexLocker locker(&requestsMutex);
        int count = 0;
        for (const auto& [id, request] : requests) {
            if (request.status == "pending") {
                count++;
            }
        }
        return count;
    }
};

#endif //GEEP_GEEPSERVICE_H