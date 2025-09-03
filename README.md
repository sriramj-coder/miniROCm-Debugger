# miniROCm-Debugger  

A lightweight **CUDA-based GPU debugger and memory manager**, inspired by ROCm tools.  
This project provides a command-line interface (CLI) for managing GPU memory, launching kernels, inspecting memory, and logging execution details.  

---

## 🚀 Features  

### GPU Initialization  
- Detects and initializes available CUDA devices  

### Memory Management  
- `--alloc <bytes>` → Allocate GPU memory  
- `--free <id>` → Free memory by ID  
- `--freeall` → Free all GPU memory  
- `--list` → List active memory allocations  

### Kernel Dispatch  
- Supports example kernels: `fill_array`, `scale_array`  
- `--kernel <name>` → Launch kernel on allocated memory  
- `--grid <x> <y> <z>` and `--block <x> <y> <z>` → Configure execution dimensions  
- `--dispatches` → List kernel dispatch history with timestamps  

### Memory Inspection  
- `--dump <id>` → Dump contents of a memory allocation  

### Logging System  
- Structured logs with timestamps and levels (`INFO`, `WARN`, `ERROR`)  
- Integrated CUDA error handling via `CUDA_CHECK` macro  

---

## 🛠️ Example Usage  

```bash
miniROCm-Debugger.exe --alloc 64 --alloc 128 --kernel fill_array --kernel scale_array --dump 1 --dump 2 --grid 2 2 1 --block 8 1 1 --dispatches --freeall







