//
// Created by aleks on 4.3.2025.
//

#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#ifdef __cplusplus
extern "C" {
#endif

    // Function to check for CUDA support
    bool hasCudaSupport(bool verbose = true);

    // void cudaPrintf(const char* format, ...);

#ifdef __cplusplus
}
#endif

#endif //CUDAUTILS_H
