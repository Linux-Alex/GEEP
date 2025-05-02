#include <cuda_runtime.h>

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

// ANSI color codes for terminal output
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

// Example CUDA kernel
__global__ void kernel() {
    printf("Hello from CUDA kernel!\n");
}

// extern "C" __device__ void cudaPrintf(const char* format, ...) {
// #ifdef DEBUG_CUDA
//     va_list args;
//     va_start(args, format);
//     vprintf(format, args);
//     va_end(args);
// #endif
// }


extern "C" void run_cuda_code() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();  // Ensure the kernel has finished executing
}

// Function to log messages with style
void logMessage(const std::string &message, bool isError = false) {
    if (isError) {
        // For errors, use red for the message
        std::cerr << RED << BOLD << "ERROR: " << RESET << RED << message << RESET << std::endl;
    } else {
        // For info, use bold cyan for the message
        std::cout << CYAN << BOLD << "INFO: " << RESET << GREEN << message << RESET << std::endl;
    }
}

// Function to check for CUDA support and display device information
extern "C" bool hasCudaSupport(bool verbose = true) {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    // Check if CUDA is available and at least one GPU is present
    if (error != cudaSuccess || deviceCount == 0) {
        if (verbose) {
            logMessage("CUDA error: " + std::string(cudaGetErrorString(error)), true);
        }
        return false;
    }

    if (verbose) {
        logMessage("Found " + std::to_string(deviceCount) + " CUDA-capable device(s):");

        // Display information for each device
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);

            std::cout << CYAN << BOLD << "  Device " << i << ": " << RESET
                      << GREEN << deviceProp.name << RESET
                      << " (Compute Capability: " << deviceProp.major << "." << deviceProp.minor << ")\n";
        }
    }

    return true;
}