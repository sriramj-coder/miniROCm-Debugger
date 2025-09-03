#include "CUDABackend.h"
#include <iostream>
#include <vector>
#include <string>

// === CUDA Kernels ===
// Simple kernel: fills an array with a constant value
__global__ void fill_array(int* data, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// Simple kernel: multiplies each element by a factor
__global__ void scale_array(int* data, int factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}

// === Backend Class Implementation ===

void CUDABackend::init() {
    cudaDeviceProp deviceProps{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, 0));
    log(LogLevel::INFO, "[CUDA] Initialized on device: " + std::string(deviceProps.name));
}

void CUDABackend::alloc(size_t numBytes) {
    void* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, numBytes));
    if (!devicePtr) return;

    MemoryBlock memBlock{ nextId++, numBytes, devicePtr };
    allocations.push_back(memBlock);

    log(LogLevel::INFO, "[CUDA] Allocated " + std::to_string(numBytes) +
                        " bytes at ID=" + std::to_string(memBlock.id));
}

void CUDABackend::freeAll() {
    for (auto& memBlock : allocations) {
        CUDA_CHECK(cudaFree(memBlock.ptr));
        log(LogLevel::INFO, "[CUDA] Freed " + std::to_string(memBlock.size) +
                            " bytes (ID=" + std::to_string(memBlock.id) + ")");
    }
    allocations.clear();
}

void CUDABackend::freeById(size_t id) {
    for (auto it = allocations.begin(); it != allocations.end(); ++it) {
        if (it->id == id) {
            CUDA_CHECK(cudaFree(it->ptr));
            log(LogLevel::INFO, "[CUDA] Freed " + std::to_string(it->size) +
                                " bytes (ID=" + std::to_string(it->id) + ")");
            allocations.erase(it);
            return;
        }
    }
    log(LogLevel::ERROR, "[CUDA] No allocation found with ID=" + std::to_string(id));
}

void CUDABackend::launchKernel(const std::string& kernelName) {
    if (allocations.empty()) {
        log(LogLevel::ERROR, "[CUDA] No memory allocated, cannot launch kernel");
        return;
    }

    // always use the first allocation
    MemoryBlock& target = allocations.front();
    int n = static_cast<int>(target.size / sizeof(int));

    if (kernelName == "fill_array") {
        fill_array<<<currentGrid, currentBlock>>>((int*)target.ptr, 42, n);
    }
    else if (kernelName == "scale_array") {
        scale_array<<<currentGrid, currentBlock>>>((int*)target.ptr, 2, n);
    }
    else {
        log(LogLevel::ERROR, "[CUDA] Unknown kernel name: " + kernelName);
        return;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    log(LogLevel::INFO, "[CUDA] Kernel '" + kernelName + "' dispatched on ID=" +
                        std::to_string(target.id) + " (" + std::to_string(n) + " ints), grid=(" +
                        std::to_string(currentGrid.x) + "," + std::to_string(currentGrid.y) + "," +
                        std::to_string(currentGrid.z) + "), block=(" +
                        std::to_string(currentBlock.x) + "," + std::to_string(currentBlock.y) + "," +
                        std::to_string(currentBlock.z) + ")");

    dispatches.push_back({ kernelName, target.id, currentGrid, currentBlock,
                           std::chrono::system_clock::now() });
}

void CUDABackend::listAllocations() const {
    if (allocations.empty()) {
        log(LogLevel::INFO, "[CUDA] No active allocations");
        return;
    }

    log(LogLevel::INFO, "[CUDA] Active allocations:");
    for (const auto& memBlock : allocations) {
        std::cout << "  ID=" << memBlock.id
                  << " | Size=" << memBlock.size << " bytes\n";
    }
}

void CUDABackend::listDispatches() const {
    if (dispatches.empty()) {
        log(LogLevel::INFO, "[CUDA] No kernel dispatches");
        return;
    }

    log(LogLevel::INFO, "[CUDA] Kernel Dispatches:");
    for (const auto& d : dispatches) {
        auto t = std::chrono::system_clock::to_time_t(d.timestamp);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::cout << "  [" << std::put_time(&tm, "%H:%M:%S") << "] Kernel: " << d.name
                  << " | MemID: " << d.memId
                  << " | Grid: (" << d.gridDim.x << "," << d.gridDim.y << "," << d.gridDim.z << ")"
                  << " | Block: (" << d.blockDim.x << "," << d.blockDim.y << "," << d.blockDim.z << ")\n";
    }
}

void CUDABackend::dumpMemory(size_t id, size_t count) {
    for (const auto& memBlock : allocations) {
        if (memBlock.id == id) {
            int n = static_cast<int>(memBlock.size / sizeof(int));
            count = std::min<int>(count, n);

            std::vector<int> hostBuffer(count);
            CUDA_CHECK(cudaMemcpy(hostBuffer.data(), memBlock.ptr,
                                  count * sizeof(int), cudaMemcpyDeviceToHost));

            log(LogLevel::INFO, "Memory dump (first " + std::to_string(count) +
                                " ints) for ID=" + std::to_string(id));

            for (int val : hostBuffer) {
                std::cout << val << " ";
            }
            std::cout << "\n";
            return;
        }
    }
    log(LogLevel::ERROR, "No allocation found with ID=" + std::to_string(id));
}

void CUDABackend::setGrid(dim3 grid) {
    currentGrid = grid;
}

void CUDABackend::setBlock(dim3 block) {
    currentBlock = block;
}
