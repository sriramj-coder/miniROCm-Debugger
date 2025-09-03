#include "CUDABackend.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    CUDABackend backend;
    backend.init();

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--alloc" && i + 1 < argc) {
            size_t numBytes = std::stoull(argv[++i]);
            backend.alloc(numBytes);
        }
        else if (arg == "--free" && i + 1 < argc) {
            size_t id = std::stoull(argv[++i]);
            backend.freeById(id);
        }
        else if (arg == "--freeall") {
            backend.freeAll();
        }
        else if (arg == "--list") {
            backend.listAllocations();
        }
        else if (arg == "--kernel" && i + 1 < argc) {
            std::string kernelName(argv[++i]);
            backend.launchKernel(kernelName);
        }
        else if (arg == "--grid" && i + 3 < argc) {
            int gx = std::stoi(argv[++i]);
            int gy = std::stoi(argv[++i]);
            int gz = std::stoi(argv[++i]);
            backend.setGrid(dim3(gx, gy, gz));
        }
        else if (arg == "--block" && i + 3 < argc) {
            int bx = std::stoi(argv[++i]);
            int by = std::stoi(argv[++i]);
            int bz = std::stoi(argv[++i]);
            backend.setBlock(dim3(bx, by, bz));
        }
        else if (arg == "--dispatches") {
            backend.listDispatches();
        }
        else if (arg == "--dump" && i + 1 < argc) {
            size_t id = std::stoull(argv[++i]);
            backend.dumpMemory(id);
        }
        else if (arg == "--help") {
            std::cout << "Usage:\n"
                << "  --alloc <bytes>\n"
                << "  --free <id>\n"
                << "  --freeall\n"
                << "  --list\n"
                << "  --kernel <fill_array|scale_array>\n"
                << "  --grid <x> <y> <z>\n"
                << "  --block <x> <y> <z>\n"
                << "  --dispatches\n"
                << "  --dump <id>\n";
        }
        else {
            log(LogLevel::ERROR, "Unknown or incomplete argument: " + arg);
        }
    }

    return 0;
}
