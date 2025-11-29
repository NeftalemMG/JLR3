// ============================================================================
// PROFILING UTILITIES
// ============================================================================
// This file provides utilities for measuring and comparing performance of
// CUDA kernels, specifically for comparing non-fused vs fused implementations.
//
// METRICS MEASURED:
// 1. Kernel execution time (milliseconds)
// 2. Memory throughput (GB/s)
// 3. GPU occupancy (theoretical)
// 4. End-to-end inference time per batch
//
// USAGE:
// 1. Create a Profiler object
// 2. Start timing with start_timer()
// 3. Launch your kernel
// 4. Stop timing with stop_timer()
// 5. Print results with print_kernel_stats()
// ============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

// ============================================================================
// CUDA EVENT-BASED TIMER
// ============================================================================
// CUDA events provide accurate GPU timing by recording timestamps directly
// on the GPU timeline. This is more accurate than CPU-based timing because:
// 1. No CPU-GPU synchronization overhead
// 2. Measures actual kernel execution time (not including launch overhead)
// 3. Sub-millisecond precision
//
// How it works:
// - cudaEventRecord() inserts a timestamp into the GPU's command queue
// - cudaEventElapsedTime() computes the difference between two timestamps
// - cudaEventSynchronize() blocks CPU until GPU reaches that point
// ============================================================================

class CudaTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float elapsed_ms;
    bool is_timing;

public:
    // Constructor: Create CUDA events
    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        elapsed_ms = 0.0f;
        is_timing = false;
    }

    // Destructor: Clean up CUDA events
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    // Start timing
    void start() {
        // Record event in the default stream (stream 0)
        // This creates a timestamp right before the next kernel launch
        cudaEventRecord(start_event, 0);
        is_timing = true;
    }

    // Stop timing
    void stop() {
        if (!is_timing) {
            printf("Warning: Timer was not started\n");
            return;
        }

        // Record end event
        cudaEventRecord(stop_event, 0);
        
        // Wait for the event to complete
        // This ensures the kernel has finished executing
        cudaEventSynchronize(stop_event);
        
        // Compute elapsed time in milliseconds
        cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
        
        is_timing = false;
    }

    // Get elapsed time in milliseconds
    float get_elapsed_ms() const {
        return elapsed_ms;
    }

    // Get elapsed time in seconds
    float get_elapsed_sec() const {
        return elapsed_ms / 1000.0f;
    }
};

// ============================================================================
// CPU TIMER (for end-to-end timing including kernel launches)
// ============================================================================
// Sometimes we want to measure total time including:
// - Kernel launch overhead
// - Memory transfers
// - CPU-GPU synchronization
//
// For this, we use CPU-side timing with gettimeofday()
// ============================================================================

class CPUTimer {
private:
    struct timeval start_time;
    struct timeval stop_time;
    double elapsed_ms;

public:
    CPUTimer() : elapsed_ms(0.0) {}

    void start() {
        gettimeofday(&start_time, NULL);
    }

    void stop() {
        gettimeofday(&stop_time, NULL);
        
        // Compute elapsed time in milliseconds
        long seconds = stop_time.tv_sec - start_time.tv_sec;
        long microseconds = stop_time.tv_usec - start_time.tv_usec;
        elapsed_ms = (seconds * 1000.0) + (microseconds / 1000.0);
    }

    double get_elapsed_ms() const {
        return elapsed_ms;
    }

    double get_elapsed_sec() const {
        return elapsed_ms / 1000.0;
    }
};

// ============================================================================
// KERNEL STATISTICS STRUCTURE
// ============================================================================
// This structure holds all the metrics we want to track for a kernel
// ============================================================================

struct KernelStats {
    const char* kernel_name;        // Name of the kernel
    float execution_time_ms;        // Kernel execution time (ms)
    size_t memory_bytes_read;       // Bytes read from global memory
    size_t memory_bytes_written;    // Bytes written to global memory
    size_t total_threads;           // Total number of threads launched
    int blocks;                     // Number of blocks
    int threads_per_block;          // Threads per block
    
    // Computed metrics
    float memory_throughput_gbs;    // Memory bandwidth (GB/s)
    float compute_time_per_element; // Time per element (microseconds)
    
    // Initialize all fields
    KernelStats(const char* name) 
        : kernel_name(name),
          execution_time_ms(0.0f),
          memory_bytes_read(0),
          memory_bytes_written(0),
          total_threads(0),
          blocks(0),
          threads_per_block(0),
          memory_throughput_gbs(0.0f),
          compute_time_per_element(0.0f) {}
    
    // Compute derived metrics
    void compute_metrics() {
        // Total memory traffic (read + write)
        size_t total_memory_bytes = memory_bytes_read + memory_bytes_written;
        
        // Memory throughput in GB/s
        // Formula: (bytes / 1e9) / (time_ms / 1000)
        if (execution_time_ms > 0) {
            memory_throughput_gbs = (total_memory_bytes / 1e9f) / (execution_time_ms / 1000.0f);
            
            // Time per element in microseconds
            if (total_threads > 0) {
                compute_time_per_element = (execution_time_ms * 1000.0f) / total_threads;
            }
        }
    }
    
    // Print statistics in a nice format
    void print() const {
        printf("\n");
        printf("========================================\n");
        printf("Kernel: %s\n", kernel_name);
        printf("========================================\n");
        printf("Execution time:     %.4f ms\n", execution_time_ms);
        printf("Memory read:        %.2f MB\n", memory_bytes_read / 1e6f);
        printf("Memory written:     %.2f MB\n", memory_bytes_written / 1e6f);
        printf("Total memory:       %.2f MB\n", (memory_bytes_read + memory_bytes_written) / 1e6f);
        printf("Memory throughput:  %.2f GB/s\n", memory_throughput_gbs);
        printf("Blocks:             %d\n", blocks);
        printf("Threads per block:  %d\n", threads_per_block);
        printf("Total threads:      %zu\n", total_threads);
        printf("Time per element:   %.4f μs\n", compute_time_per_element);
        printf("========================================\n");
    }
};

// ============================================================================
// PROFILER CLASS
// ============================================================================
// High-level interface for profiling kernels
// ============================================================================

class KernelProfiler {
private:
    CudaTimer cuda_timer;
    CPUTimer cpu_timer;

public:
    KernelProfiler() {}

    // Profile a kernel launch and return statistics
    // This is a generic profiling function that measures timing
    KernelStats profile_kernel(
        const char* kernel_name,
        void (*kernel_func)(void*),  // Function pointer to kernel wrapper
        void* kernel_args,            // Kernel arguments
        size_t memory_read,           // Bytes read from memory
        size_t memory_written,        // Bytes written to memory
        int blocks,                   // Number of blocks
        int threads_per_block         // Threads per block
    ) {
        KernelStats stats(kernel_name);
        
        // Set launch configuration
        stats.blocks = blocks;
        stats.threads_per_block = threads_per_block;
        stats.total_threads = blocks * threads_per_block;
        stats.memory_bytes_read = memory_read;
        stats.memory_bytes_written = memory_written;
        
        // Warm-up run (important for accurate timing)
        // First kernel launch may be slower due to:
        // - Driver initialization
        // - Code cache misses
        // - Memory allocation overhead
        kernel_func(kernel_args);
        cudaDeviceSynchronize();
        
        // Actual timed run
        cuda_timer.start();
        kernel_func(kernel_args);
        cuda_timer.stop();
        
        // Record timing
        stats.execution_time_ms = cuda_timer.get_elapsed_ms();
        
        // Compute derived metrics
        stats.compute_metrics();
        
        return stats;
    }
    
    // Measure end-to-end inference time for a batch
    double profile_batch_inference(
        void (*forward_pass)(void*),
        void* args,
        int num_iterations
    ) {
        // Warm-up
        forward_pass(args);
        cudaDeviceSynchronize();
        
        // Time multiple iterations
        cpu_timer.start();
        for (int i = 0; i < num_iterations; i++) {
            forward_pass(args);
        }
        cudaDeviceSynchronize();
        cpu_timer.stop();
        
        // Return average time per iteration
        return cpu_timer.get_elapsed_ms() / num_iterations;
    }
};

// ============================================================================
// COMPARISON UTILITIES
// ============================================================================
// Functions to compare non-fused vs fused implementations
// ============================================================================

// Compare two kernel implementations and print speedup
void compare_kernels(const KernelStats& baseline, const KernelStats& optimized) {
    printf("\n");
    printf("============================================\n");
    printf("PERFORMANCE COMPARISON\n");
    printf("============================================\n");
    printf("Baseline:  %s\n", baseline.kernel_name);
    printf("Optimized: %s\n", optimized.kernel_name);
    printf("--------------------------------------------\n");
    
    // Execution time comparison
    float time_speedup = baseline.execution_time_ms / optimized.execution_time_ms;
    printf("Execution time:\n");
    printf("  Baseline:  %.4f ms\n", baseline.execution_time_ms);
    printf("  Optimized: %.4f ms\n", optimized.execution_time_ms);
    printf("  Speedup:   %.2fx\n", time_speedup);
    printf("\n");
    
    // Memory throughput comparison
    float bw_improvement = (optimized.memory_throughput_gbs / baseline.memory_throughput_gbs) * 100.0f - 100.0f;
    printf("Memory throughput:\n");
    printf("  Baseline:  %.2f GB/s\n", baseline.memory_throughput_gbs);
    printf("  Optimized: %.2f GB/s\n", optimized.memory_throughput_gbs);
    printf("  Improvement: %.1f%%\n", bw_improvement);
    printf("\n");
    
    // Memory traffic reduction
    size_t baseline_mem = baseline.memory_bytes_read + baseline.memory_bytes_written;
    size_t optimized_mem = optimized.memory_bytes_read + optimized.memory_bytes_written;
    float mem_reduction = (1.0f - (float)optimized_mem / baseline_mem) * 100.0f;
    printf("Memory traffic:\n");
    printf("  Baseline:  %.2f MB\n", baseline_mem / 1e6f);
    printf("  Optimized: %.2f MB\n", optimized_mem / 1e6f);
    printf("  Reduction: %.1f%%\n", mem_reduction);
    printf("============================================\n");
}

// Print a table comparing multiple implementations
void print_comparison_table(KernelStats* stats, int num_kernels) {
    printf("\n");
    printf("=================================================================================\n");
    printf("%-40s %12s %12s %12s\n", "Kernel", "Time (ms)", "BW (GB/s)", "Speedup");
    printf("=================================================================================\n");
    
    // Use first kernel as baseline
    float baseline_time = stats[0].execution_time_ms;
    
    for (int i = 0; i < num_kernels; i++) {
        float speedup = baseline_time / stats[i].execution_time_ms;
        printf("%-40s %12.4f %12.2f %12.2fx\n",
               stats[i].kernel_name,
               stats[i].execution_time_ms,
               stats[i].memory_throughput_gbs,
               speedup);
    }
    printf("=================================================================================\n");
}

// ============================================================================
// TENSOR SHAPE IMPACT ANALYSIS
// ============================================================================
// Study how tensor dimensions affect kernel performance
// ============================================================================

void analyze_shape_impact(
    void (*kernel_func)(int, int, int, int),  // Kernel that takes N, C, H, W
    const char* kernel_name
) {
    printf("\n");
    printf("========================================\n");
    printf("Tensor Shape Impact Analysis: %s\n", kernel_name);
    printf("========================================\n");
    
    // Test different batch sizes
    printf("\nBatch size impact (C=64, H=28, W=28):\n");
    printf("%-15s %12s\n", "Batch Size", "Time (ms)");
    printf("----------------------------------------\n");
    
    int batch_sizes[] = {1, 4, 16, 32, 64, 128};
    for (int i = 0; i < 6; i++) {
        int N = batch_sizes[i];
        // Note: actual implementation would call kernel_func(N, 64, 28, 28)
        // and measure timing - this is a simplified template
        printf("%-15d %12s\n", N, "[measure]");
    }
    
    // Test different channel sizes
    printf("\nChannel size impact (N=32, H=28, W=28):\n");
    printf("%-15s %12s\n", "Channels", "Time (ms)");
    printf("----------------------------------------\n");
    
    int channel_sizes[] = {16, 32, 64, 128, 256};
    for (int i = 0; i < 5; i++) {
        int C = channel_sizes[i];
        printf("%-15d %12s\n", C, "[measure]");
    }
    
    // Test different spatial sizes
    printf("\nSpatial size impact (N=32, C=64):\n");
    printf("%-15s %12s\n", "Spatial Size", "Time (ms)");
    printf("----------------------------------------\n");
    
    int spatial_sizes[] = {14, 28, 56, 112};
    for (int i = 0; i < 4; i++) {
        int H = spatial_sizes[i];
        printf("%-15dx%d %12s\n", H, H, "[measure]");
    }
    
    printf("========================================\n");
}

// ============================================================================
// UTILITY: Check CUDA errors
// ============================================================================
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// UTILITY: Print GPU device properties
// ============================================================================
void print_device_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("\n");
    printf("========================================\n");
    printf("GPU Device Information\n");
    printf("========================================\n");
    printf("Device name:           %s\n", prop.name);
    printf("Compute capability:    %d.%d\n", prop.major, prop.minor);
    printf("Total global memory:   %.2f GB\n", prop.totalGlobalMem / 1e9f);
    printf("Shared memory per block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);
    printf("Registers per block:   %d\n", prop.regsPerBlock);
    printf("Warp size:             %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM:    %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs:         %d\n", prop.multiProcessorCount);
    printf("Memory clock rate:     %.2f MHz\n", prop.memoryClockRate / 1000.0f);
    printf("Memory bus width:      %d-bit\n", prop.memoryBusWidth);
    printf("Peak memory bandwidth: %.2f GB/s\n", 
           2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6f);
    printf("========================================\n");
}

// ============================================================================
// END OF PROFILING_UTILS.CU
// ============================================================================