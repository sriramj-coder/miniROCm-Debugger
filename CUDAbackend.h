#pragma once
#include <vector>
#include <string>
#include <cstddef>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <iomanip>

// === Logging Helper ===
enum class LogLevel { INFO, WARN, ERROR };

inline void log(LogLevel level, const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::string levelStr;
    switch (level) {
    case LogLevel::INFO:  levelStr = "[INFO]"; break;
    case LogLevel::WARN:  levelStr = "[WARN]"; break;
    case LogLevel::ERROR: levelStr = "[ERROR]"; break;
    }

    std::cout << std::put_time(&tm, "%H:%M:%S") << " "
        << levelStr << " " << message << "\n";
}

// CUDA error-checking wrapper
#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        log(LogLevel::ERROR, std::string("CUDA error at ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + " -> " + cudaGetErrorString(err));     \
    }                                                                         \
} while(0)

// === Data Structures ===
struct MemoryBlock {
    size_t id;
    size_t size;
    void* ptr;
};

struct KernelDispatch {
    std::string name;
    size_t memId;
    dim3 gridDim;
    dim3 blockDim;
    std::chrono::system_clock::time_point timestamp;
};

// === CUDA Backend Class ===
class CUDABackend {
public:
    void init();
    void alloc(size_t numBytes);
    void freeAll();
    void freeById(size_t id);

    void launchKernel(const std::string& kernelName);
    void dumpMemory(size_t id, size_t count = 16);

    void listAllocations() const;
    void listDispatches() const;

    void setGrid(dim3 grid);
    void setBlock(dim3 block);

private:
    std::vector<MemoryBlock> allocations;
    std::vector<KernelDispatch> dispatches;

    size_t nextId = 1;

    dim3 currentGrid = dim3(1, 1, 1);
    dim3 currentBlock = dim3(128, 1, 1);
};
