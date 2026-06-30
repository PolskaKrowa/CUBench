/**
 * CUBench - The Definitive Open-Source GPU Benchmarking Utility
 * -------------------------------------------------------------
 * A comprehensive GPU performance benchmarking tool designed to evaluate CUDA kernel execution,
 * inefficient memory handling, thread spilling, and much more.
 * 
 * Author: Stevenson Parker
 * Created: 24/07/2025 (DD/MM/YYYY)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#pragma comment(lib, "cusparse")
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/reverse.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <vector>
#ifdef _WIN32
    #include <conio.h>
#endif
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp16.hpp>
#include <sstream>
#if defined(_WIN32) || defined(_WIN64)
    #include <io.h>
    #define fileno _fileno
    #define dup _dup
    #define dup2 _dup2
    #define close _close
    #define isatty _isatty
#else
    #include <unistd.h>
#endif
#include <algorithm>
#include <random>
#include <cstdarg>
#include <iomanip>
#include <string>
#include <cooperative_groups.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

// ----------------------------------------------------------------------------
// ProgressBar configuration — compile-time overrides.
//
// These can be overridden at compile time via -D flags, e.g.:
//
//   nvcc -DPROGRESS_BAR_FPS=30 ...          // 30 fps redraw
//   nvcc -DPROGRESS_BAR_FPS=5  ...          // 5  fps redraw (low CPU)
//
// PROGRESS_BAR_FPS controls how often the spinner thread redraws the bar
// (both the spinner animation and the sixel animation).  Higher values =
// smoother animation but more CPU usage.  Default: 10 fps.
//
// Animation timing is REAL-TIME based, not frame based.  The shimmer sweep
// and brightness pulse are driven by elapsed wall-clock seconds (computed
// from anim_phase_ / PROGRESS_BAR_FPS), so they run at the same speed
// regardless of framerate — smoother at high FPS, choppier but the same
// pace at low FPS.
//
// The animation periods can also be overridden at compile time (in seconds):
//
//   nvcc -DPROGRESS_BAR_SHIMMER_PERIOD=1.0 ...   // shimmer cycle (default 2.0 s)
//   nvcc -DPROGRESS_BAR_PULSE_PERIOD=1.0   ...   // pulse cycle   (default 1.8 s)
// ----------------------------------------------------------------------------
#ifndef PROGRESS_BAR_FPS
#define PROGRESS_BAR_FPS 10
#endif
#ifndef PROGRESS_BAR_SHIMMER_PERIOD
#define PROGRESS_BAR_SHIMMER_PERIOD 2.0
#endif
#ifndef PROGRESS_BAR_PULSE_PERIOD
#define PROGRESS_BAR_PULSE_PERIOD 1.8
#endif
#include <cstring>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>
#include <fstream>
#include <map>
#include <numeric>
#include <chrono>
#if defined(__linux__)
    #include <dlfcn.h>
    #include <pthread.h>
    #include <sched.h>
    #include <dirent.h>
    #include <sys/stat.h>
    #include <sys/types.h>
#elif defined(_WIN32)
    #include <windows.h>
#endif

namespace cg = cooperative_groups;
using namespace nvcuda::wmma;

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSPARSE error %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult res = call; \
        if (res != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, res); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#ifndef _WIN32
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/select.h>

int getch(void) {
    struct termios oldattr, newattr;
    int ch;

    tcgetattr(STDIN_FILENO, &oldattr);             // Get current terminal attributes
    newattr = oldattr;
    newattr.c_lflag &= ~(ICANON | ECHO);           // Disable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &newattr);    // Set new attributes

    ch = getchar();                                // Read one character

    tcsetattr(STDIN_FILENO, TCSANOW, &oldattr);    // Restore old attributes
    return ch;
}

#define _getch() getch()

#endif

// Helper: check CUDA errors
static inline void CHECK_CUDA(cudaError_t e, const char* msg = nullptr) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error%s: %s\n", msg ? msg : "", cudaGetErrorString(e));
        exit(1);
    }
}

// Helper: simple timing wrapper using cuda events
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop)); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void startEvent() { CHECK_CUDA(cudaEventRecord(start)); }
    float stopMs() {
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ============================================================================
// NVML dynamic loader - resolves NVML symbols at runtime so the binary does
// not require NVML headers or a hard link dependency.  If NVML is not present
// at runtime the related benchmarks degrade gracefully.
// ============================================================================
struct NvmlApi {
    bool loaded = false;
    bool initialized = false;
    void* handle = nullptr;

    // Function pointer typedefs (use raw void* for the opaque nvmlDevice_t)
    typedef int (*fn_init_t)(void);
    typedef int (*fn_shutdown_t)(void);
    typedef int (*fn_get_count_t)(unsigned int*);
    typedef int (*fn_get_handle_by_index_t)(unsigned int, void**);
    typedef int (*fn_get_handle_by_pci_bus_id_t)(const char*, void**);
    typedef int (*fn_get_power_usage_t)(void*, unsigned int*);
    typedef int (*fn_get_power_limit_t)(void*, unsigned int*);
    typedef int (*fn_get_temperature_t)(void*, int, unsigned int*);
    typedef int (*fn_get_clock_info_t)(void*, int, unsigned int*);
    typedef int (*fn_get_max_clock_info_t)(void*, int, unsigned int*);
    typedef int (*fn_get_app_clocks_t)(void*, int type, unsigned int* clock);
    typedef int (*fn_get_throttle_reasons_t)(void*, unsigned long long*);
    typedef int (*fn_get_nvlink_state_t)(void*, unsigned int, unsigned int*);
    typedef int (*fn_get_nvlink_version_t)(void*, unsigned int, unsigned int*);
    typedef int (*fn_get_nvlink_remote_pcie_info_t)(void*, unsigned int, void*);
    typedef const char* (*fn_error_string_t)(int);

    fn_init_t                       Init = nullptr;
    fn_shutdown_t                   Shutdown = nullptr;
    fn_get_count_t                  DeviceGetCount = nullptr;
    fn_get_handle_by_index_t        DeviceGetHandleByIndex = nullptr;
    fn_get_handle_by_pci_bus_id_t   DeviceGetHandleByPciBusId = nullptr;
    fn_get_power_usage_t            DeviceGetPowerUsage = nullptr;
    fn_get_power_limit_t            DeviceGetPowerManagementLimit = nullptr;
    fn_get_temperature_t            DeviceGetTemperature = nullptr;
    fn_get_clock_info_t             DeviceGetClockInfo = nullptr;
    fn_get_max_clock_info_t         DeviceGetMaxClockInfo = nullptr;
    fn_get_app_clocks_t             DeviceGetApplicationClocks = nullptr;
    fn_get_throttle_reasons_t       DeviceGetThrottleReasons = nullptr;
    fn_get_nvlink_state_t           DeviceGetNvLinkState = nullptr;
    fn_get_nvlink_version_t         DeviceGetNvLinkVersion = nullptr;
    fn_get_nvlink_remote_pcie_info_t DeviceGetNvLinkRemotePciInfo = nullptr;
    fn_error_string_t               ErrorString = nullptr;

    static NvmlApi& instance() {
        static NvmlApi inst;
        return inst;
    }

    bool load() {
        if (loaded) return initialized;
        loaded = true;
#if defined(__linux__)
        handle = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!handle) handle = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
#elif defined(_WIN32)
        handle = (void*)LoadLibraryA("nvml.dll");
#endif
        if (!handle) return false;

        #define RESOLVE(name, sym) \
            name = reinterpret_cast<decltype(name)>(sym_lookup(handle, sym)); \
            if (!name) { /* try with _v2 suffix */ \
                name = reinterpret_cast<decltype(name)>(sym_lookup(handle, sym "_v2")); \
            }

        RESOLVE(Init, "nvmlInit");
        RESOLVE(Shutdown, "nvmlShutdown");
        RESOLVE(DeviceGetCount, "nvmlDeviceGetCount");
        RESOLVE(DeviceGetHandleByIndex, "nvmlDeviceGetHandleByIndex");
        RESOLVE(DeviceGetHandleByPciBusId, "nvmlDeviceGetHandleByPciBusId");
        RESOLVE(DeviceGetPowerUsage, "nvmlDeviceGetPowerUsage");
        RESOLVE(DeviceGetPowerManagementLimit, "nvmlDeviceGetPowerManagementLimit");
        RESOLVE(DeviceGetTemperature, "nvmlDeviceGetTemperature");
        RESOLVE(DeviceGetClockInfo, "nvmlDeviceGetClockInfo");
        RESOLVE(DeviceGetMaxClockInfo, "nvmlDeviceGetMaxClockInfo");
        RESOLVE(DeviceGetApplicationClocks, "nvmlDeviceGetApplicationClocks");
        RESOLVE(DeviceGetThrottleReasons, "nvmlDeviceGetThrottleReasons");
        RESOLVE(DeviceGetNvLinkState, "nvmlDeviceGetNvLinkState");
        RESOLVE(DeviceGetNvLinkVersion, "nvmlDeviceGetNvLinkVersion");
        RESOLVE(DeviceGetNvLinkRemotePciInfo, "nvmlDeviceGetNvLinkRemotePciInfo");
        ErrorString = reinterpret_cast<fn_error_string_t>(sym_lookup(handle, "nvmlErrorString"));
        #undef RESOLVE

        if (!Init || !DeviceGetHandleByIndex) return false;
        if (Init() != 0 /* NVML_SUCCESS */) return false;
        initialized = true;
        return true;
    }

    ~NvmlApi() {
        if (initialized && Shutdown) Shutdown();
        // Note: do not dlclose on Linux as some NVML versions crash on dlopen/dlclose cycles.
        // The library handle leaks but this is process-lifetime only.
    }

private:
    static void* sym_lookup(void* h, const char* name) {
        if (!h) return nullptr;
#if defined(__linux__)
        return dlsym(h, name);
#elif defined(_WIN32)
        return (void*)GetProcAddress((HMODULE)h, name);
#else
        return nullptr;
#endif
    }
};

// ----------------------------------------------------------------------------
// Throttle reason flag decoder (NVML clocks throttle reason bitmask)
// ----------------------------------------------------------------------------
static std::string decodeThrottleReasons(unsigned long long reasons) {
    if (reasons == 0) return "none";
    std::string out;
    struct Bit { unsigned long long mask; const char* name; };
    static const Bit bits[] = {
        {0x01, "gpu_idle"},
        {0x02, "app_clocks"},
        {0x04, "sw_power_cap"},
        {0x08, "hw_slowdown"},
        {0x10, "sync_boost"},
        {0x20, "sw_thermal"},
        {0x40, "hw_thermal"},
        {0x80, "hw_power_brake"},
        {0x100, "display_clocks"},
    };
    for (const auto& b : bits) {
        if (reasons & b.mask) {
            if (!out.empty()) out += "|";
            out += b.name;
        }
    }
    return out;
}

// ============================================================================
// GpuMetricSampler - background sampler that polls NVML for power, clocks,
// temperature and throttle reasons while a benchmark workload runs.
// ============================================================================
class GpuMetricSampler {
public:
    struct Sample {
        std::chrono::steady_clock::time_point t;
        unsigned int sm_clock_mhz = 0;
        unsigned int mem_clock_mhz = 0;
        unsigned int power_mw = 0;
        unsigned int temp_c = 0;
        unsigned long long throttle_reasons = 0;
    };

    struct Stats {
        bool valid = false;
        double avg_power_w = 0;
        double peak_power_w = 0;
        double avg_sm_clock_mhz = 0;
        double peak_sm_clock_mhz = 0;
        double min_sm_clock_mhz = 0;
        double avg_mem_clock_mhz = 0;
        double avg_temp_c = 0;
        double peak_temp_c = 0;
        double clock_drop_pct = 0;   // (peak - sustained_avg) / peak * 100
        size_t throttle_samples = 0;
        size_t total_samples = 0;
        unsigned long long last_throttle_reasons = 0;
    };

    GpuMetricSampler() = default;
    ~GpuMetricSampler() { stop(); }

    // Initialise NVML handle for the given CUDA device index.  Returns false
    // if NVML is unavailable (sampler becomes a no-op).
    bool init(int device_index) {
        NvmlApi& nvml = NvmlApi::instance();
        if (!nvml.load()) return false;
        unsigned int count = 0;
        if (nvml.DeviceGetCount(&count) != 0) return false;
        if ((unsigned int)device_index >= count) return false;
        void* dev = nullptr;
        if (nvml.DeviceGetHandleByIndex((unsigned int)device_index, &dev) != 0) return false;
        nvml_device_ = dev;
        return true;
    }

    bool available() const { return nvml_device_ != nullptr; }

    unsigned int getPowerLimitMw() const {
        if (!available()) return 0;
        NvmlApi& nvml = NvmlApi::instance();
        if (!nvml.DeviceGetPowerManagementLimit) return 0;
        unsigned int limit_mw = 0;
        if (nvml.DeviceGetPowerManagementLimit(nvml_device_, &limit_mw) != 0) return 0;
        return limit_mw;
    }

    void start(int sample_interval_ms = 50) {
        if (!available()) return;
        sample_interval_ms_ = sample_interval_ms;
        {
            std::lock_guard<std::mutex> lock(samples_mutex_);
            samples_.clear();
        }
        running_.store(true);
        thread_ = std::thread([this]() {
            while (running_.load()) {
                sampleOnce();
                std::this_thread::sleep_for(std::chrono::milliseconds(sample_interval_ms_));
            }
        });
    }

    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        if (thread_.joinable()) thread_.join();
        sampleOnce(); // final sample
    }

    Stats computeStats(size_t skip_initial = 2) const {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        Stats st;
        if (samples_.size() <= skip_initial) return st;
        st.valid = true;
        st.total_samples = samples_.size() - skip_initial;

        // Peak across ALL samples (peak clock typically at start)
        for (size_t i = 0; i < samples_.size(); i++) {
            const auto& s = samples_[i];
            if (s.sm_clock_mhz > 0) st.peak_sm_clock_mhz = std::max(st.peak_sm_clock_mhz, (double)s.sm_clock_mhz);
            if (s.power_mw > 0)     st.peak_power_w = std::max(st.peak_power_w, s.power_mw / 1000.0);
            if (s.temp_c > 0)       st.peak_temp_c = std::max(st.peak_temp_c, (double)s.temp_c);
        }

        // Averages over sustained region (skip warm-up samples)
        double sum_p = 0, sum_sm = 0, sum_mem = 0, sum_t = 0;
        size_t n_p = 0, n_sm = 0, n_mem = 0, n_t = 0, throttle_count = 0;
        double min_sm = 1e9;
        for (size_t i = skip_initial; i < samples_.size(); i++) {
            const auto& s = samples_[i];
            if (s.power_mw > 0)     { sum_p += s.power_mw / 1000.0; n_p++; }
            if (s.sm_clock_mhz > 0) { sum_sm += s.sm_clock_mhz; n_sm++; min_sm = std::min(min_sm, (double)s.sm_clock_mhz); }
            if (s.mem_clock_mhz > 0){ sum_mem += s.mem_clock_mhz; n_mem++; }
            if (s.temp_c > 0)       { sum_t += s.temp_c; n_t++; }
            if (s.throttle_reasons != 0) { throttle_count++; st.last_throttle_reasons |= s.throttle_reasons; }
        }
        if (n_p > 0) st.avg_power_w = sum_p / n_p;
        if (n_sm > 0) st.avg_sm_clock_mhz = sum_sm / n_sm;
        if (n_mem > 0) st.avg_mem_clock_mhz = sum_mem / n_mem;
        if (n_t > 0) st.avg_temp_c = sum_t / n_t;
        if (min_sm < 1e8) st.min_sm_clock_mhz = min_sm;
        st.throttle_samples = throttle_count;
        if (st.peak_sm_clock_mhz > 0)
            st.clock_drop_pct = (st.peak_sm_clock_mhz - st.avg_sm_clock_mhz) / st.peak_sm_clock_mhz * 100.0;
        return st;
    }

private:
    void sampleOnce() {
        if (!available()) return;
        NvmlApi& nvml = NvmlApi::instance();
        Sample s;
        s.t = std::chrono::steady_clock::now();
        unsigned int val = 0;
        if (nvml.DeviceGetClockInfo && nvml.DeviceGetClockInfo(nvml_device_, 0 /*NVML_CLOCK_SM*/, &val) == 0)
            s.sm_clock_mhz = val;
        val = 0;
        if (nvml.DeviceGetClockInfo && nvml.DeviceGetClockInfo(nvml_device_, 1 /*NVML_CLOCK_MEM*/, &val) == 0)
            s.mem_clock_mhz = val;
        val = 0;
        if (nvml.DeviceGetPowerUsage && nvml.DeviceGetPowerUsage(nvml_device_, &val) == 0)
            s.power_mw = val;
        val = 0;
        if (nvml.DeviceGetTemperature && nvml.DeviceGetTemperature(nvml_device_, 0 /*NVML_TEMPERATURE_GPU*/, &val) == 0)
            s.temp_c = val;
        unsigned long long reasons = 0;
        if (nvml.DeviceGetThrottleReasons && nvml.DeviceGetThrottleReasons(nvml_device_, &reasons) == 0)
            s.throttle_reasons = reasons;
        std::lock_guard<std::mutex> lock(samples_mutex_);
        samples_.push_back(s);
    }

    void* nvml_device_ = nullptr;
    std::vector<Sample> samples_;
    mutable std::mutex samples_mutex_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    int sample_interval_ms_ = 50;
};

// ============================================================================
// NUMA topology helpers (Linux only; graceful no-op elsewhere)
// ============================================================================
struct NumaNode {
    int node_id = 0;
    std::vector<int> cpus;
};

static std::vector<int> parseCpuList(const std::string& s) {
    std::vector<int> cpus;
    size_t i = 0;
    while (i < s.size()) {
        // Skip non-digit chars
        while (i < s.size() && (s[i] < '0' || s[i] > '9')) i++;
        if (i >= s.size()) break;
        int start = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            start = start * 10 + (s[i] - '0'); i++;
        }
        int end = start;
        if (i < s.size() && s[i] == '-') {
            i++;
            end = 0;
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                end = end * 10 + (s[i] - '0'); i++;
            }
        }
        for (int c = start; c <= end; c++) cpus.push_back(c);
        // Skip until comma or end
        while (i < s.size() && s[i] != ',') i++;
        if (i < s.size() && s[i] == ',') i++;
    }
    return cpus;
}

static std::vector<NumaNode> detectNumaNodes() {
    std::vector<NumaNode> nodes;
#if defined(__linux__)
    DIR* dir = opendir("/sys/devices/system/node");
    if (!dir) return nodes;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() < 5 || name.substr(0, 4) != "node") continue;
        int node_id = atoi(name.c_str() + 4);
        NumaNode node;
        node.node_id = node_id;
        std::ifstream f("/sys/devices/system/node/" + name + "/cpulist");
        std::string cpus_str;
        if (std::getline(f, cpus_str)) {
            node.cpus = parseCpuList(cpus_str);
        }
        if (node.cpus.empty()) {
            // Fallback: try /proc/cpuinfo-style cpumap
            std::ifstream fm("/sys/devices/system/node/" + name + "/cpumap");
            std::string cpumap;
            if (std::getline(fm, cpumap)) {
                // Parse hex bitmap - rare; just pick a synthetic CPU 0
                node.cpus.push_back(node_id * 8);
            }
        }
        nodes.push_back(node);
    }
    closedir(dir);
#endif
    return nodes;
}

// Read the NUMA node for a GPU given its PCI coordinates.  Returns -1 if
// unknown (single-NUMA systems, Windows, etc.).
static int getGpuNumaNode(int pci_domain, int pci_bus, int pci_device, int pci_func = 0) {
#if defined(__linux__)
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%d/numa_node",
             pci_domain, pci_bus, pci_device, pci_func);
    std::ifstream f(path);
    int node = -1;
    if (f >> node) return node;
#endif
    return -1;
}

// Try to discover NVLink connections between two CUDA devices via NVML.
// Returns the NVLink version (1, 2, 3, 4) if any active link exists, else 0.
static int detectNvLinkBetween(int dev_a, int dev_b) {
    NvmlApi& nvml = NvmlApi::instance();
    if (!nvml.load()) return 0;
    if (!nvml.DeviceGetNvLinkState) return 0;
    void* h_a = nullptr;
    if (nvml.DeviceGetHandleByIndex((unsigned int)dev_a, &h_a) != 0) return 0;
    unsigned int state = 0;
    int best_version = 0;
    // NVML_MAX_NVLINK_LINKS is typically 6 on Pascal/Volta, 12 on newer.
    for (unsigned int link = 0; link < 18; link++) {
        if (nvml.DeviceGetNvLinkState(h_a, link, &state) != 0) break;
        if (state == 0) continue;
        unsigned int version = 0;
        if (nvml.DeviceGetNvLinkVersion) {
            if (nvml.DeviceGetNvLinkVersion(h_a, link, &version) == 0) {
                best_version = std::max(best_version, (int)version);
            }
        } else {
            best_version = std::max(best_version, 1); // unknown version but active
        }
    }
    return best_version;
}

// Vector math utility functions
__device__ __host__ float3 normalize(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 1e-6f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ __host__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __host__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __host__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// ============================================================================
// ProgressBar — single-bar progress indicator with optional sixel graphics.
//
// Shows ONE progress bar (no scrolling task history) that updates in place.
// The bar spans the full terminal width and auto-adjusts on resize.
//
// Three rendering modes:
//
//   1. Sixel mode  — a 12-pixel-tall animated sixel graphics bar (2 sixel
//                    rows, full terminal width in pixels) above a single text
//                    line.  Three animation effects run simultaneously:
//                      • Gradient pulse  — brightness breathes (sine wave)
//                      • Shimmer sweep   — bright band sweeps L→R across fill
//                      • Leading-edge glow — bright cyan glow at fill boundary
//                    Re-drawn in place via DECSC/DECRC cursor save/restore +
//                    ED erase-to-end-of-screen.  Robust to whether sixel
//                    actually rendered: we always restore to the saved start
//                    position and erase everything below it, so no row-
//                    counting is needed.
//
//   2. Text mode   — a single ASCII-art bar inline with the text line.
//
//   3. Non-TTY     — one line per task, no escape codes (for piped output).
//
// Sixel detection: env-var heuristics first (instant), then a DA1 terminal
// query (ESC[c) as fallback — the definitive method.  Attribute 4 in the
// DA1 response means sixel is supported.  The CUBENCH_SIXEL env var can
// override the detection ("1"/"0").
// ============================================================================
class ProgressBar {
public:
    ProgressBar(int total, const char* label = "Benchmark")
        : total_(total), label_(label ? label : "Benchmark") {
        is_tty_ = isatty(fileno(stderr));
        use_unicode_ = is_tty_ && detectUnicodeSupport();
        use_sixel_ = is_tty_ && detectSixelSupport();
        // To force sixel on/off regardless of detection, set the env var:
        //     CUBENCH_SIXEL=1   (force on)    CUBENCH_SIXEL=0   (force off)
        start_time_ = std::chrono::steady_clock::now();
        task_start_ = start_time_;

        if (!is_tty_) {
            fprintf(stderr, "Starting %d %s tasks...\n", total_, label_.c_str());
            fflush(stderr);
        } else {
            emit_progress();
        }
    }

    ~ProgressBar() { stop_spinner(); }

    void start_task(int idx, const std::string& name) {
        stop_spinner();
        current_ = idx;
        current_name_ = name;
        task_start_ = std::chrono::steady_clock::now();

        if (!is_tty_) {
            fprintf(stderr, "  [%d/%d] %s ... ", current_, total_, name.c_str());
            fflush(stderr);
            return;
        }

        spinner_running_ = true;
        spinner_thread_ = std::thread([this]() {
            // Braille spinner (UTF-8) or ASCII fallback
            const char* spinner_chars = use_unicode_
                ? "\xe2\xa0\x8b\xe2\xa0\x99\xe2\xa0\xb9\xe2\xa0\xb8\xe2\xa0\xbc"
                  "\xe2\xa0\xb4\xe2\xa0\xa6\xe2\xa0\xa7\xe2\xa0\x87\xe2\xa0\x8f"
                : "|/-\\";
            int spin_len = use_unicode_ ? 10 : 4;
            int spin_bytes = use_unicode_ ? 3 : 1;
            int i = 0;
            while (spinner_running_) {
                {
                    std::lock_guard<std::mutex> lock(render_mutex_);
                    const char* p = spinner_chars + (i % spin_len) * spin_bytes;
                    current_spinner_ = use_unicode_
                        ? std::string(p, 3)
                        : std::string(p, 1);
                    emit_progress();
                }
                i++;
                // Frame interval derived from PROGRESS_BAR_FPS (default 10
                // → 100 ms).  1000 / FPS, rounded to the nearest millisecond.
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(1000 / PROGRESS_BAR_FPS));
            }
        });
    }

    void finish_task() {
        stop_spinner();
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - task_start_).count();
        completed_count_++;

        if (!is_tty_) {
            fprintf(stderr, "done (%.2fs)\n", elapsed);
            fflush(stderr);
            return;
        }

        std::lock_guard<std::mutex> lock(render_mutex_);
        current_ = 0;
        current_name_ = "";
        current_spinner_ = " ";
        emit_progress();
    }

    void finish_all() {
        stop_spinner();
        auto now = std::chrono::steady_clock::now();
        double total_elapsed =
            std::chrono::duration<double>(now - start_time_).count();

        if (!is_tty_) {
            fprintf(stderr, "All %d %s tasks complete in %.2fs\n",
                    total_, label_.c_str(), total_elapsed);
            fflush(stderr);
            return;
        }

        std::lock_guard<std::mutex> lock(render_mutex_);
        current_ = total_;
        current_name_ = "Complete";
        current_spinner_ = use_unicode_ ? "\xe2\x9c\x93" : "*";  // ✓ or *
        completed_count_ = total_;
        emit_progress();
        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s: %d tasks in %.2fs\n",
                label_.c_str(), total_, total_elapsed);
        fflush(stderr);
    }

private:
    void stop_spinner() {
        if (spinner_running_) {
            spinner_running_ = false;
            if (spinner_thread_.joinable()) spinner_thread_.join();
        }
    }

    // ------------------------------------------------------------------
    // Terminal size detection (queried on every render → auto-resize)
    // ------------------------------------------------------------------
    int getTerminalWidth() {
#if defined(_WIN32)
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(
                GetStdHandle(STD_ERROR_HANDLE), &csbi)) {
            return csbi.srWindow.Right - csbi.srWindow.Left + 1;
        }
        return 80;
#else
        struct winsize ws;
        if (ioctl(fileno(stderr), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
            return ws.ws_col;
        return 80;
#endif
    }

    int getTerminalPixelWidth() {
#if !defined(_WIN32)
        struct winsize ws;
        if (ioctl(fileno(stderr), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
            if (ws.ws_xpixel > 0) {
                // Derive the cell width from the reported pixel dimensions,
                // then compute the image width as cell_w × cols.  Integer
                // division (floor) guarantees the image width is an exact
                // multiple of the cell width and never exceeds the
                // terminal's actual pixel width — preventing the sixel
                // image from wrapping to extra rows.
                int cell_w = ws.ws_xpixel / ws.ws_col;
                if (cell_w > 0) return cell_w * ws.ws_col;
            }
            // Fallback when ws_xpixel is unavailable: assume 9 px/column.
            // This is conservative (most monospace fonts at common sizes
            // are 7-10 px wide) and prevents overflow on narrower cells.
            // The previous × 10 caused overflow on 8-9 px cells.
            return ws.ws_col * 9;
        }
#endif
        return getTerminalWidth() * 9;
    }

    // ------------------------------------------------------------------
    // Sixel detection: env vars → DA1 terminal query (definitive)
    // ------------------------------------------------------------------
    bool detectSixelSupport() {
        if (!isatty(fileno(stderr))) return false;

        // User override
        const char* force = getenv("CUBENCH_SIXEL");
        if (force) {
            std::string f(force);
            if (f == "1" || f == "yes" || f == "on" || f == "true") return true;
            if (f == "0" || f == "no" || f == "off" || f == "false") return false;
        }

        // Quick env-var heuristics
        const char* term = getenv("TERM");
        if (term) {
            std::string t(term);
            if (t == "dumb" || t == "linux" || t.empty()) return false;
            if (t.find("screen") != std::string::npos) return false;
            if (t.find("tmux") != std::string::npos) return false;
            if (t.find("sixel") != std::string::npos) return true;
            if (t.find("mlterm") != std::string::npos) return true;
            if (t.find("foot") != std::string::npos) return true;
        }

        const char* tp = getenv("TERM_PROGRAM");
        if (tp) {
            std::string s(tp);
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (s == "wezterm") return true;
            if (s == "mintty") return true;
            if (s == "contour") return true;
        }

        const char* te = getenv("TERMINAL_EMULATOR");
        if (te) {
            std::string s(te);
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (s.find("contour") != std::string::npos) return true;
        }

        if (getenv("CONTOUR_VERSION")) return true;

        // DA1 query as last-resort fallback (most reliable)
        return querySixelViaDA1();
    }

    // Send ESC[c (Primary Device Attributes) to the terminal and check if
    // the response contains attribute 4 (= sixel graphics support).
    //
    // The query is WRITTEN to stderr (terminal output) but the response
    // arrives on STDIN (terminal input).  We must set STDIN to raw mode
    // and read the response from there, not from stderr.
    bool querySixelViaDA1() {
#if !defined(_WIN32)
        int in_fd = STDIN_FILENO;   // response arrives on stdin
        int out_fd = fileno(stderr); // query is sent via stderr

        if (!isatty(in_fd) || !isatty(out_fd)) return false;

        struct termios oldtio, newtio;
        if (tcgetattr(in_fd, &oldtio) != 0) return false;

        newtio = oldtio;
        newtio.c_lflag &= ~(ICANON | ECHO | ECHONL);
        newtio.c_iflag &= ~(IXON | IXOFF | ICRNL);
        newtio.c_cc[VMIN] = 0;
        newtio.c_cc[VTIME] = 0;    // pure non-blocking, use select for timeout

        if (tcsetattr(in_fd, TCSANOW, &newtio) != 0) return false;

        // Flush any pending input so we don't read stale data
        tcflush(in_fd, TCIFLUSH);

        // Send DA1 query via stderr (terminal output)
        if (write(out_fd, "\033[c", 3) != 3) {
            tcsetattr(in_fd, TCSANOW, &oldtio);
            return false;
        }

        char buf[256];
        int total = 0;
        // Try for up to ~1 second (20 × 50ms)
        for (int attempt = 0; attempt < 20; attempt++) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(in_fd, &fds);
            struct timeval tv = {0, 50000};  // 50ms per attempt
            int ret = select(in_fd + 1, &fds, nullptr, nullptr, &tv);
            if (ret <= 0) break;
            int n = read(in_fd, buf + total, sizeof(buf) - total - 1);
            if (n <= 0) break;
            total += n;
            if (buf[total - 1] == 'c') break;  // DA response ends with 'c'
        }

        tcsetattr(in_fd, TCSANOW, &oldtio);
        buf[total] = '\0';

        if (total == 0) return false;

        // Check for attribute 4 (sixel) in the DA1 response.
        // Response format: ESC [ ? <attr> ; <attr> ; ... c
        // Attribute 4 = sixel graphics support.
        std::string resp(buf, total);
        // Match "4" as a complete attribute value (bounded by ? ; or c)
        if (resp.find("?4;") != std::string::npos) return true;
        if (resp.find(";4;") != std::string::npos) return true;
        if (resp.find(";4c") != std::string::npos) return true;
        if (resp.find("?4c") != std::string::npos) return true;
        return false;
#else
        return false;
#endif
    }

    bool detectUnicodeSupport() {
        const char* checks[] = {"LANG", "LC_ALL", "LC_CTYPE", nullptr};
        for (int i = 0; checks[i]; i++) {
            const char* val = getenv(checks[i]);
            if (val) {
                std::string v(val);
                if (v.find("UTF-8") != std::string::npos ||
                    v.find("utf8") != std::string::npos ||
                    v.find("UTF8") != std::string::npos)
                    return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------
    // UTF-8 display-width helpers
    // ------------------------------------------------------------------
    static int displayWidth(const std::string& s) {
        int width = 0;
        for (size_t i = 0; i < s.size(); ) {
            unsigned char c = (unsigned char)s[i];
            if ((c & 0x80) == 0)        { i += 1; }       // ASCII
            else if ((c & 0xE0) == 0xC0) { i += 2; }       // 2-byte
            else if ((c & 0xF0) == 0xE0) { i += 3; }       // 3-byte (braille, CJK)
            else if ((c & 0xF8) == 0xF0) { i += 4; }       // 4-byte (emoji)
            else                          { i += 1; }       // invalid
            width++;
        }
        return width;
    }

    // Truncate a UTF-8 string to at most max_width display columns.
    static std::string truncateToWidth(const std::string& s, int max_width) {
        int width = 0;
        size_t i = 0;
        while (i < s.size() && width < max_width) {
            unsigned char c = (unsigned char)s[i];
            size_t next;
            if      ((c & 0x80) == 0)    next = i + 1;
            else if ((c & 0xE0) == 0xC0) next = i + 2;
            else if ((c & 0xF0) == 0xE0) next = i + 3;
            else if ((c & 0xF8) == 0xF0) next = i + 4;
            else                          next = i + 1;
            if (next > s.size()) break;
            i = next;
            width++;
        }
        return s.substr(0, i);
    }

    // ------------------------------------------------------------------
    // Rendering — always exactly 1 line (text mode) or 2 lines (sixel +
    // text), fully overwritten each time.
    //
    // CRITICAL: the text line must NEVER wrap.  If it wraps to 2 visual
    // lines, \r and \033[K only affect the wrapped portion, leaving stale
    // content on the first visual line.  We prevent wrapping by:
    //   1. Building the text content first (spinner, percent, task, timing)
    //   2. Calculating its display width
    //   3. Filling the remaining width with the ASCII bar
    //   4. Truncating the final line to term_width (display columns, not bytes)
    // ------------------------------------------------------------------
    void emit_progress() {
        if (!is_tty_) return;

        int term_width = getTerminalWidth();

        if (first_emit_) {
            // Reserve 3 rows below the cursor (2 for the sixel image + 1
            // for the text line) so that writing the bar doesn't scroll
            // the terminal on each redraw.  The newlines expose 3 blank
            // rows; we then move back up, reset to column 0, and save.
            // Sixel height = 2 rows (12 px ÷ 6 px/row).
            fprintf(stderr, "\n\n\n\033[3A\r\0337");  // 3 LF + up 3 + CR + DECSC
            first_emit_ = false;
        } else {
            // Restore to the saved start position, reset to column 0,
            // and erase from the cursor to the end of the screen.  This
            // clears both the old sixel rows and the old text row.
            fprintf(stderr, "\0338\r\033[J");  // DECRC + CR + ED-0
        }

        if (use_sixel_) {
            anim_phase_++;  // advance animation frame (PROGRESS_BAR_FPS via spinner)
            int pixel_width = getTerminalPixelWidth();
            double frac = total_ > 0
                ? (double)completed_count_ / (double)total_ : 0.0;
            emit_sixel_bar(frac, pixel_width);
            // After the sixel image, cursor position is terminal-dependent:
            // some terminals move it down by the image height, others leave
            // it where it was.  We explicitly restore to the saved position
            // (P, column 0) and move down by the image height (2 rows) so
            // the text line always lands on the row immediately below the
            // image, regardless of terminal behaviour.
            fprintf(stderr, "\0338\r\033[2B");  // DECRC + CR + cursor-down 2
        }

        emit_text_line(term_width);
        fflush(stderr);
    }

    void emit_text_line(int term_width) {
        double frac = total_ > 0
            ? (double)completed_count_ / (double)total_ : 0.0;
        int percent = (int)(frac * 100);

        auto now = std::chrono::steady_clock::now();
        double total_elapsed =
            std::chrono::duration<double>(now - start_time_).count();
        double task_elapsed = current_ > 0
            ? std::chrono::duration<double>(now - task_start_).count() : 0.0;

        // --- Build the text portion (everything except the ASCII bar) ---
        std::string text;
        text += current_spinner_;

        char buf[256];
        int display_idx = (current_ > 0) ? current_ : completed_count_;
        snprintf(buf, sizeof(buf), " %3d%% [%d/%d]", percent, display_idx, total_);
        text += buf;

        if (!current_name_.empty()) {
            text += " " + current_name_;
        }
        if (task_elapsed > 0) {
            snprintf(buf, sizeof(buf), " | %.1fs", task_elapsed);
            text += buf;
        }

        std::string line;
        if (!use_sixel_) {
            // Insert an ASCII bar between the spinner and the text info.
            // Layout: <spinner> [<bar>] <percent> [task] [timing]
            // Bar width is calculated to fill exactly the remaining space.
            std::string prefix = current_spinner_;
            prefix += " [";

            std::string suffix;
            snprintf(buf, sizeof(buf), "] %3d%% [%d/%d]", percent, display_idx, total_);
            suffix += buf;

            if (!current_name_.empty()) {
                suffix += " " + current_name_;
            }
            if (task_elapsed > 0) {
                snprintf(buf, sizeof(buf), " | %.1fs", task_elapsed);
                suffix += buf;
            }

            int prefix_width = displayWidth(prefix);
            int suffix_width = displayWidth(suffix);
            int bar_width = term_width - prefix_width - suffix_width;
            if (bar_width < 5) bar_width = 5;

            int filled = (int)(frac * bar_width);
            if (filled > bar_width) filled = bar_width;
            if (filled < 0) filled = 0;

            line = prefix;
            for (int i = 0; i < bar_width; i++) {
                if (i < filled)       line += '=';
                else if (i == filled) line += '>';
                else                  line += ' ';
            }
            line += suffix;
        } else {
            // Sixel mode: no ASCII bar, just the text
            line = text;
        }

        // Safety: truncate to terminal width (display columns, not bytes)
        // to prevent wrapping if our width calculation was slightly off.
        if (displayWidth(line) > term_width) {
            line = truncateToWidth(line, term_width);
        }

        fprintf(stderr, "%s", line.c_str());
    }

    // ------------------------------------------------------------------
    // Sixel bar — full terminal pixel width, 12px tall (2 sixel rows),
    // with an animated blue→cyan gradient.
    //
    // Animation effects (driven by anim_phase_, incremented per emit at
    // PROGRESS_BAR_FPS via the spinner thread; default 10 fps):
    //
    //   • Gradient pulse   — the filled gradient "breathes": brightness is
    //                         modulated by a sine wave with period
    //                         PROGRESS_BAR_PULSE_PERIOD (default 1.8 s).
    //   • Shimmer sweep    — a bright white-cyan band (~40 px wide) sweeps
    //                         left→right across the filled portion and
    //                         loops with period PROGRESS_BAR_SHIMMER_PERIOD
    //                         (default 2.0 s), like a moving reflection.
    //   • Leading-edge glow — the last 3 px of the fill are bright cyan,
    //                         creating a glowing "front" that advances as
    //                         the bar fills.
    //
    // TIMING IS REAL-TIME BASED: anim_phase_ (a frame counter) is divided
    // by PROGRESS_BAR_FPS to yield elapsed seconds, and the animation
    // phases are computed from that.  This decouples animation speed from
    // framerate — the shimmer and pulse complete in the same wall-clock
    // time at any FPS, just with more samples (smoother) at high FPS and
    // fewer (choppier) at low FPS.  The animation looks identical at the
    // default 10 fps regardless of what FPS is compiled in.
    //
    // The gradient is position-based (not progress-based): the left side
    // is always deep blue and the right side is always bright cyan,
    // regardless of fill level.  Only the fill cutoff moves.
    //
    // All sixel RGB values are 0–100 (per the VT240/VT340 spec).
    // Run-length encoded for compactness.
    // ------------------------------------------------------------------
    void emit_sixel_bar(double frac, int pixel_width) {
        if (pixel_width < 10) pixel_width = 10;
        int filled = (int)(frac * pixel_width);
        if (filled > pixel_width) filled = pixel_width;
        if (filled < 0) filled = 0;

        const int num_bands     = 8;     // gradient colour bands
        const int shimmer_width = 40;    // px — width of the sweeping highlight
        const int edge_glow     = 3;     // px — bright glow at fill boundary
        // Periods are in SECONDS (real time), configurable via -D flags.
        const double shimmer_period = PROGRESS_BAR_SHIMMER_PERIOD;
        const double pulse_period   = PROGRESS_BAR_PULSE_PERIOD;

        // Convert the frame counter to elapsed seconds.  This is the key
        // to framerate-independent animation: at 30 fps we get 3× as many
        // samples per second, but each sample advances the phase by 1/30
        // of a second instead of 1/10, so the real-time speed is identical.
        double t = (double)anim_phase_ / (double)PROGRESS_BAR_FPS;

        // --- Shimmer position: sweeps off-screen-left → off-screen-right ---
        int shimmer_start = -1, shimmer_end = -1;
        if (filled > shimmer_width) {
            int sweep = filled + shimmer_width;           // total travel
            double phase_frac = fmod(t, shimmer_period) / shimmer_period;  // 0..1
            shimmer_start = (int)(phase_frac * (double)sweep) - shimmer_width;
            shimmer_end   = shimmer_start + shimmer_width;
        }

        // --- Pulse: brightness modulation 0.82 .. 1.0 (sine wave) ---
        float pulse = 0.82f + 0.18f *
            (0.5f + 0.5f * sinf((float)(t * 2.0 * 3.14159265 / pulse_period)));

        std::string s = "\033Pq";
        char buf[64];

        // --- Colour palette (values MUST be 0–100 per sixel spec) ---
        // 0: unfilled portion (dark blue-gray)
        s += "#0;2;9;9;17";
        // 1..8: gradient (pulse-modulated brightness)
        for (int i = 0; i < num_bands; i++) {
            float t = (float)i / (float)(num_bands - 1);
            int r = (int)(0             * pulse);
            int g = (int)((31 + t * 69) * pulse);
            int b = (int)((67 + t * 33) * pulse);
            snprintf(buf, sizeof(buf), "#%d;2;%d;%d;%d", i + 1, r, g, b);
            s += buf;
        }
        // 9:  shimmer highlight (bright white-cyan)
        s += "#9;2;85;100;100";
        // 10: leading-edge glow (bright cyan)
        s += "#10;2;55;100;100";

        // --- Build one sixel row, RLE-encoded ---
        // Per-pixel colour priority (highest first):
        //   unfilled  >  edge-glow  >  shimmer  >  gradient
        int band_width = std::max(1, pixel_width / num_bands);

        auto colour_at = [&](int x) -> int {
            if (x >= filled) return 0;                          // unfilled
            if (x >= filled - edge_glow) return 10;             // leading edge
            if (shimmer_end >= 0 &&                              // shimmer active
                x >= shimmer_start && x < shimmer_end) return 9;
            int band = x / band_width;                          // gradient
            if (band >= num_bands) band = num_bands - 1;
            return band + 1;
        };

        std::string row;
        int x = 0;
        while (x < pixel_width) {
            int c = colour_at(x);
            int run = 1;
            while (x + run < pixel_width && colour_at(x + run) == c) run++;
            snprintf(buf, sizeof(buf), "#%d", c);
            row += buf;
            if (run > 1) {
                snprintf(buf, sizeof(buf), "!%d", run);
                row += buf;
            }
            row += "~";     // all 6 vertical pixels on
            x += run;
        }

        // Emit 2 sixel rows (12 px total) — both identical for a solid bar.
        // The '-' character advances the sixel cursor to the next 6-px row.
        s += row;
        s += "-";
        s += row;
        s += "\033\\";    // ST — end DCS
        fprintf(stderr, "%s", s.c_str());
    }

    // ------------------------------------------------------------------
    int total_;
    int current_ = 0;
    int completed_count_ = 0;
    std::string current_name_;
    std::string current_spinner_ = " ";
    std::string label_;
    bool is_tty_ = false;
    bool use_sixel_ = false;
    bool use_unicode_ = false;
    bool first_emit_ = true;
    int anim_phase_ = 0;     // animation frame counter (incremented per emit)
    std::chrono::steady_clock::time_point start_time_, task_start_;
    std::thread spinner_thread_;
    std::atomic<bool> spinner_running_{false};
    std::mutex render_mutex_;
};

// Benchmark configuration
struct BenchmarkConfig {
    int width = 1920;
    int height = 1080;
    int iterations = 100;
    int num_triangles = 1000;
    int num_particles = 100000;
    int texture_size = 1024;
    int size = 1048576;
    int atomic_test_size = 1000000;
    int memory_test_size = 1048576;
    int instruction_test_size = 1048576;
    int occupancy_test_size = 1048576;
    int tensor_size = 1024;
    int context_switch_iterations = 1000;
    int ilp_test_size = 1048576;
    int divergence_test_size = 1048576;
    int dynamic_kernel_depth = 3;
    int dp_test_size = 10;
    int memory_latency_test_size = 1024;
    int pcie_transfer_size = 1048576;
    int p2p_transfer_size = 1048576;
    int async_test_size = 1048576;
    int async_streams = 4;
    int instruction_mix_size = 1048576;
    int cache_thrash_size = 1048576;
    int cache_conflict_sets = 16;
    int numa_transfer_size = 4 * 1048576;     // 4M floats = 16 MB per transfer
    int numa_transfer_warmup = 5;
    int numa_transfer_iterations = 30;
    int nvlink_transfer_size = 4 * 1048576;   // 4M floats = 16 MB per transfer
    int nvlink_transfer_warmup = 5;
    int nvlink_transfer_iterations = 50;
    int power_efficiency_workload_size = 1048576; // #threads in the workload
    int power_efficiency_light_ops = 2000;       // FMAs per thread (light)
    int power_efficiency_heavy_ops = 10000;      // FMAs per thread (heavy)
    int power_efficiency_duration_ms = 1500;     // target run length per workload
    int power_efficiency_sample_ms = 50;         // NVML sampling interval
    int power_efficiency_warmup_ms = 200;        // warm-up before measuring

    // Softmax + Flash-Attention benchmark configuration.
    // The attention benchmark treats inputs as a [batch*heads, seq, head_dim]
    // tensor; for the standalone softmax microbenchmark we use the same M
    // (batch*heads) rows of length N (seq_len).
    int softmax_seq_len       = 1024;   // N — sequence length / row length
    int softmax_head_dim      = 64;     // D — head dimension (must match FA_D)
    int softmax_batch_heads   = 1024;   // M — total rows (batch * num_heads)
    int softmax_iters         = 50;     // iterations per timed region
};

// Vertex structure
struct Vertex {
    float x, y, z;
    float r, g, b;
    float u, v;
};

// Triangle structure
struct Triangle {
    Vertex v0, v1, v2;
};

// Particle structure
struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float r, g, b, a;
    float size;
};

// Ray structure
struct Ray {
    float ox, oy, oz;  // origin
    float dx, dy, dz;  // direction
};

// Sphere structure
struct Sphere {
    float x, y, z;
    float radius;
    float r, g, b;
};

// Matrix structure
struct Matrix4x4 {
    float m[16];
};

// Mesh structure
struct Mesh {
    int vertex_count;
    int triangle_count;
    Vertex* vertices;
    int* indices;
};

// Light structure
struct Light {
    float x, y, z;
    float r, g, b;
    float intensity;
    int type;         // 0=point, 1=directional, 2=spot
};

// BVH Node structure for ray tracing acceleration
struct BVHNode {
    float3 min_bound;
    float3 max_bound;
    int left_child;   // -1 if leaf
    int right_child;  // -1 if leaf
    int triangle_start; // for leaf nodes
    int triangle_count; // for leaf nodes
};

// Basic rasterisation kernel
__global__ void rasteriseTriangles(Triangle* triangles, int num_triangles, 
                                 float* framebuffer, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    float x = (float)idx / width * 2.0f - 1.0f;
    float y = (float)idy / height * 2.0f - 1.0f;
    
    float closest_z = -1000.0f;
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    
    // Simple point-in-triangle test for all triangles
    for (int i = 0; i < num_triangles; i++) {
        Triangle tri = triangles[i];
        
        // Barycentric coordinates
        float denom = (tri.v1.y - tri.v2.y) * (tri.v0.x - tri.v2.x) + 
                      (tri.v2.x - tri.v1.x) * (tri.v0.y - tri.v2.y);
        
        if (fabsf(denom) < 1e-6f) continue;
        
        float a = ((tri.v1.y - tri.v2.y) * (x - tri.v2.x) + 
                   (tri.v2.x - tri.v1.x) * (y - tri.v2.y)) / denom;
        float b = ((tri.v2.y - tri.v0.y) * (x - tri.v2.x) + 
                   (tri.v0.x - tri.v2.x) * (y - tri.v2.y)) / denom;
        float c = 1.0f - a - b;
        
        if (a >= 0.0f && b >= 0.0f && c >= 0.0f) {
            float z = a * tri.v0.z + b * tri.v1.z + c * tri.v2.z;
            if (z > closest_z) {
                closest_z = z;
                color.x = a * tri.v0.r + b * tri.v1.r + c * tri.v2.r;
                color.y = a * tri.v0.g + b * tri.v1.g + c * tri.v2.g;
                color.z = a * tri.v0.b + b * tri.v1.b + c * tri.v2.b;
            }
        }
    }
    
    int pixel_idx = (idy * width + idx) * 3;
    framebuffer[pixel_idx] = color.x;
    framebuffer[pixel_idx + 1] = color.y;
    framebuffer[pixel_idx + 2] = color.z;
}

// Particle simulation and rendering kernel
__global__ void renderParticles(Particle* particles, int num_particles,
                               float* framebuffer, int width, int height, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_particles) return;
    
    Particle* p = &particles[idx];
    
    // Update particle position
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    p->z += p->vz * dt;
    
    // Apply gravity
    p->vy -= 9.81f * dt;
    
    // Bounce off boundaries
    if (p->x < -1.0f || p->x > 1.0f) p->vx *= -0.8f;
    if (p->y < -1.0f || p->y > 1.0f) p->vy *= -0.8f;
    
    // Project to screen space
    int screen_x = (int)((p->x + 1.0f) * 0.5f * width);
    int screen_y = (int)((p->y + 1.0f) * 0.5f * height);
    
    if (screen_x >= 0 && screen_x < width && screen_y >= 0 && screen_y < height) {
        int pixel_idx = (screen_y * width + screen_x) * 3;
        
        // Alpha blending
        float alpha = p->a;
        atomicAdd(&framebuffer[pixel_idx], p->r * alpha);
        atomicAdd(&framebuffer[pixel_idx + 1], p->g * alpha);
        atomicAdd(&framebuffer[pixel_idx + 2], p->b * alpha);
    }
}

// Simple ray-sphere intersection
__device__ bool intersectSphere(Ray ray, Sphere sphere, float* t) {
    float dx = ray.ox - sphere.x;
    float dy = ray.oy - sphere.y;
    float dz = ray.oz - sphere.z;
    
    float a = ray.dx * ray.dx + ray.dy * ray.dy + ray.dz * ray.dz;
    float b = 2.0f * (dx * ray.dx + dy * ray.dy + dz * ray.dz);
    float c = dx * dx + dy * dy + dz * dz - sphere.radius * sphere.radius;
    
    float discriminant = b * b - 4.0f * a * c;
    
    if (discriminant < 0) return false;
    
    float sqrt_disc = sqrtf(discriminant);
    float t1 = (-b - sqrt_disc) / (2.0f * a);
    float t2 = (-b + sqrt_disc) / (2.0f * a);
    
    *t = (t1 > 0) ? t1 : t2;
    return *t > 0;
}

// Ray tracing kernel
__global__ void rayTrace(Sphere* spheres, int num_spheres,
                        float* framebuffer, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    // Generate ray
    Ray ray;
    ray.ox = 0.0f;
    ray.oy = 0.0f;
    ray.oz = -5.0f;
    
    float screen_x = (float)idx / width * 2.0f - 1.0f;
    float screen_y = (float)idy / height * 2.0f - 1.0f;
    
    ray.dx = screen_x;
    ray.dy = screen_y;
    ray.dz = 1.0f;
    
    // Normalize direction
    float len = sqrtf(ray.dx * ray.dx + ray.dy * ray.dy + ray.dz * ray.dz);
    ray.dx /= len;
    ray.dy /= len;
    ray.dz /= len;
    
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float closest_t = 1e30f;
    
    // Test intersection with all spheres
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (intersectSphere(ray, spheres[i], &t) && t < closest_t) {
            closest_t = t;
            color.x = spheres[i].r;
            color.y = spheres[i].g;
            color.z = spheres[i].b;
        }
    }
    
    int pixel_idx = (idy * width + idx) * 3;
    framebuffer[pixel_idx] = color.x;
    framebuffer[pixel_idx + 1] = color.y;
    framebuffer[pixel_idx + 2] = color.z;
}

// Texture filtering kernel
__global__ void textureFilter(float* input_texture, float* output_texture,
                             int width, int height, int filter_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    float sum = 0.0f;
    int count = 0;
    int half_filter = filter_size / 2;
    
    for (int dy = -half_filter; dy <= half_filter; dy++) {
        for (int dx = -half_filter; dx <= half_filter; dx++) {
            int nx = idx + dx;
            int ny = idy + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input_texture[ny * width + nx];
                count++;
            }
        }
    }
    
    output_texture[idy * width + idx] = sum / count;
}

// Compute shader simulation kernel
__global__ void computeShaderSimulation(float* data, int size, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    float x = (float)idx / size;
    data[idx] = sinf(x * 10.0f + time) * cosf(x * 20.0f + time * 2.0f);
}

// Matrix multiplication benchmark
__global__ void matrixMultiply(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Geometry shader simulation (tessellation)
__global__ void tessellateTriangles(Triangle* input_triangles, Triangle* output_triangles, 
                                  int input_count, int tessellation_level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_count) return;
    
    Triangle tri = input_triangles[idx];
    int output_idx = idx * tessellation_level * tessellation_level;
    
    // Simple subdivision
    for (int i = 0; i < tessellation_level; i++) {
        for (int j = 0; j < tessellation_level - i; j++) {
            float u = (float)i / tessellation_level;
            float v = (float)j / tessellation_level;
            float w = 1.0f - u - v;
            
            if (output_idx < input_count * tessellation_level * tessellation_level) {
                Triangle& out_tri = output_triangles[output_idx++];
                
                // Interpolate vertices
                out_tri.v0.x = u * tri.v0.x + v * tri.v1.x + w * tri.v2.x;
                out_tri.v0.y = u * tri.v0.y + v * tri.v1.y + w * tri.v2.y;
                out_tri.v0.z = u * tri.v0.z + v * tri.v1.z + w * tri.v2.z;
                // ... (similar for v1, v2 and colors)
            }
        }
    }
}

// Deferred shading G-Buffer generation
__global__ void generateGBuffer(Triangle* triangles, int num_triangles,
                               float* position_buffer, float* normal_buffer, float* color_buffer,
                               int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    float x = (float)idx / width * 2.0f - 1.0f;
    float y = (float)idy / height * 2.0f - 1.0f;
    
    // Similar to rasterisation but store G-Buffer data
    float closest_z = -1000.0f;
    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float3 normal = make_float3(0.0f, 0.0f, 1.0f);
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int i = 0; i < num_triangles; i++) {
        Triangle tri = triangles[i];
        
        // Point-in-triangle test (same as before)
        float denom = (tri.v1.y - tri.v2.y) * (tri.v0.x - tri.v2.x) + 
                      (tri.v2.x - tri.v1.x) * (tri.v0.y - tri.v2.y);
        
        if (fabsf(denom) < 1e-6f) continue;
        
        float a = ((tri.v1.y - tri.v2.y) * (x - tri.v2.x) + 
                   (tri.v2.x - tri.v1.x) * (y - tri.v2.y)) / denom;
        float b = ((tri.v2.y - tri.v0.y) * (x - tri.v2.x) + 
                   (tri.v0.x - tri.v2.x) * (y - tri.v2.y)) / denom;
        float c = 1.0f - a - b;
        
        if (a >= 0.0f && b >= 0.0f && c >= 0.0f) {
            float z = a * tri.v0.z + b * tri.v1.z + c * tri.v2.z;
            if (z > closest_z) {
                closest_z = z;
                position = make_float3(x, y, z);
                
                // Calculate normal (cross product of edges)
                float3 edge1 = make_float3(tri.v1.x - tri.v0.x, tri.v1.y - tri.v0.y, tri.v1.z - tri.v0.z);
                float3 edge2 = make_float3(tri.v2.x - tri.v0.x, tri.v2.y - tri.v0.y, tri.v2.z - tri.v0.z);
                normal = normalize(cross(edge1, edge2));
                
                color.x = a * tri.v0.r + b * tri.v1.r + c * tri.v2.r;
                color.y = a * tri.v0.g + b * tri.v1.g + c * tri.v2.g;
                color.z = a * tri.v0.b + b * tri.v1.b + c * tri.v2.b;
            }
        }
    }
    
    int pixel_idx = (idy * width + idx) * 3;
    position_buffer[pixel_idx] = position.x;
    position_buffer[pixel_idx + 1] = position.y;
    position_buffer[pixel_idx + 2] = position.z;
    
    normal_buffer[pixel_idx] = normal.x;
    normal_buffer[pixel_idx + 1] = normal.y;
    normal_buffer[pixel_idx + 2] = normal.z;
    
    color_buffer[pixel_idx] = color.x;
    color_buffer[pixel_idx + 1] = color.y;
    color_buffer[pixel_idx + 2] = color.z;
}

// Deferred lighting pass
__global__ void deferredLighting(float* position_buffer, float* normal_buffer, float* color_buffer,
                               Light* lights, int num_lights, float* output_buffer,
                               int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    int pixel_idx = (idy * width + idx) * 3;
    
    float3 position = make_float3(position_buffer[pixel_idx], position_buffer[pixel_idx + 1], position_buffer[pixel_idx + 2]);
    float3 normal = make_float3(normal_buffer[pixel_idx], normal_buffer[pixel_idx + 1], normal_buffer[pixel_idx + 2]);
    float3 albedo = make_float3(color_buffer[pixel_idx], color_buffer[pixel_idx + 1], color_buffer[pixel_idx + 2]);
    
    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);
    
    // Accumulate lighting from all lights
    for (int i = 0; i < num_lights; i++) {
        Light light = lights[i];
        
        float3 light_dir = normalize(make_float3(light.x - position.x, light.y - position.y, light.z - position.z));
        float distance = length(make_float3(light.x - position.x, light.y - position.y, light.z - position.z));
        
        float attenuation = 1.0f / (1.0f + 0.1f * distance + 0.01f * distance * distance);
        float ndotl = fmaxf(0.0f, dot(normal, light_dir));
        
        final_color.x += albedo.x * light.r * light.intensity * ndotl * attenuation;
        final_color.y += albedo.y * light.g * light.intensity * ndotl * attenuation;
        final_color.z += albedo.z * light.b * light.intensity * ndotl * attenuation;
    }
    
    output_buffer[pixel_idx] = final_color.x;
    output_buffer[pixel_idx + 1] = final_color.y;
    output_buffer[pixel_idx + 2] = final_color.z;
}

__constant__ float c_conv_kernel[25]; // Max 5x5 kernel
__constant__ float c_memory_test[1024];

// Atomic operations benchmark kernels
__global__ void atomicAddTest(int* counters, float* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    for (int i = 0; i < iterations; i++) {
        atomicAdd(&counters[idx % 1000], 1);
        atomicAdd(&data[idx], 1.0f);
    }
}

__global__ void atomicMinMaxTest(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    int value = idx;
    atomicMin(&data[0], value);
    atomicMax(&data[1], value);
}

__global__ void atomicCASTest(int* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    for (int i = 0; i < iterations; i++) {
        int old_val, new_val;
        do {
            old_val = data[idx];
            new_val = old_val + 1;
        } while (atomicCAS(&data[idx], old_val, new_val) != old_val);
    }
}

// Memory access pattern benchmarks
__global__ void globalMemoryTest(float* input, float* output, int size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Test different access patterns
    int access_idx = (idx * stride) % size;
    output[idx] = input[access_idx] * 2.0f + 1.0f;
}

__global__ void sharedMemoryBandwidthTest(float* input, float* output, int size) {
    extern __shared__ float s_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load to shared memory
    if (idx < size) {
        s_data[tid] = input[idx];
    }
    
    __syncthreads();
    
    // Perform operations on shared memory
    float sum = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        sum += s_data[i];
    }
    
    if (idx < size) {
        output[idx] = sum / blockDim.x;
    }
}

__global__ void constantMemoryTest(float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float sum = 0.0f;
    for (int i = 0; i < 1024; i++) {
        sum += c_memory_test[i];
    }
    
    output[idx] = sum;
}

// Memory bandwidth test
__global__ void memoryBandwidthTest(float* input, float* output, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float value = input[idx];
    for (int i = 0; i < iterations; i++) {
        value = value * 1.001f + 0.001f;  // Simple computation
    }
    output[idx] = value;
}

// Shared memory test
__global__ void sharedMemoryTest(float* input, float* output, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    if (idx < size) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0 && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0 && blockIdx.x < size) {
        output[blockIdx.x] = sdata[0];
    }
}

// Post-processing effects kernel (bloom/blur)
__global__ void bloomEffect(float* input, float* output, int width, int height, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    int pixel_idx = (idy * width + idx) * 3;
    
    float r = input[pixel_idx];
    float g = input[pixel_idx + 1];
    float b = input[pixel_idx + 2];
    
    // Extract bright pixels
    float brightness = 0.299f * r + 0.587f * g + 0.114f * b;
    
    if (brightness > threshold) {
        output[pixel_idx] = r;
        output[pixel_idx + 1] = g;
        output[pixel_idx + 2] = b;
    } else {
        output[pixel_idx] = 0.0f;
        output[pixel_idx + 1] = 0.0f;
        output[pixel_idx + 2] = 0.0f;
    }
}

__global__ void multiplyKernel(cufftComplex* A,
                            const cufftComplex* B,
                            int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cufftComplex a = A[i];
        cufftComplex b = B[i];
        A[i] = make_cuComplex(a.x * b.x - a.y * b.y,
                            a.x * b.y + a.y * b.x);
    }
}

// Add these kernel implementations:
__global__ void asyncComputeKernel(float* input, float* output, int size, int workload_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float value = input[idx];
    // Intensive computation to stress ALU
    for (int i = 0; i < workload_factor; i++) {
        value = sinf(value) * cosf(value) + sqrtf(fabsf(value));
        value = logf(value + 1.0f) * expf(value * 0.001f);
    }
    output[idx] = value;
}

__global__ void asyncMemoryKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Memory-intensive pattern with strided access
    int stride = 32; // Create memory access conflicts
    int new_idx = (idx * stride) % size;
    output[idx] = input[new_idx] * 2.0f + input[(new_idx + 1) % size];
}

__global__ void instructionMixALUHeavy(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float a = input[idx];
    float result = a;
    
    // 80% ALU operations, 20% memory
    #pragma unroll 16
    for (int i = 0; i < 16; i++) {
        result = fmaf(result, 1.1f, 0.1f);  // FMA
        result = sinf(result) * cosf(result);  // Transcendental
        result = sqrtf(fabsf(result) + 1.0f);  // More ALU
    }
    
    output[idx] = result;
}

__global__ void instructionMixMemoryHeavy(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // 80% memory operations, 20% ALU
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        int access_idx = (idx + i * 131) % size;  // Prime number for conflict
        sum += input[access_idx];  // Memory load
        sum *= 1.01f;  // Minimal ALU
    }
    
    output[idx] = sum;
}

__global__ void instructionMixControlHeavy(int* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    int value = input[idx];
    float result = 1.0f;
    
    // Heavy branching and control flow
    for (int i = 0; i < 32; i++) {
        if (value & (1 << (i % 32))) {
            if (i % 3 == 0) {
                result += sinf((float)i);
            } else if (i % 3 == 1) {
                result *= cosf((float)i);
            } else {
                result = sqrtf(fabsf(result));
            }
        }
        
        // More branching
        switch (value % 7) {
            case 0: result += 1.0f; break;
            case 1: result -= 0.5f; break;
            case 2: result *= 1.1f; break;
            case 3: result /= 1.05f; break;
            case 4: result = fabsf(result); break;
            case 5: result = fmaxf(result, 0.1f); break;
            default: result = fminf(result, 100.0f); break;
        }
        
        value = (value >> 1) ^ (value & 1 ? 0x80200003 : 0);
    }
    
    output[idx] = result;
}

__global__ void cacheThrashingKernel(float* data, int* indices, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float sum = 0.0f;
    // Random memory access pattern to thrash cache
    for (int i = 0; i < 16; i++) {
        int access_idx = indices[(idx + i * size) % (size * 4)] % size;
        sum += data[access_idx];
        
        // Write back to create more cache pressure
        data[access_idx] = sum * 0.9f;
    }
}

__global__ void cacheConflictKernel(float* data, int size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size / stride) return;
    
    // Access memory in a pattern that creates cache conflicts
    int base_idx = idx * stride;
    float sum = 0.0f;
    
    // Create cache line conflicts by accessing specific patterns
    for (int i = 0; i < 32; i++) {
        int conflict_idx = (base_idx + i * 1024) % size;  // 1KB stride for L1 conflicts
        sum += data[conflict_idx];
    }
    
    data[base_idx] = sum;
}

__global__ void instructionThroughputINT(int* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    int val = data[idx];
    for (int i = 0; i < iterations; i++) {
        val = val * 3 + 7;
        val = val ^ (val << 13);
        val = val ^ (val >> 17);
        val = val ^ (val << 5);
    }
    data[idx] = val;
}

__global__ void instructionThroughputFP16(half* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    half val = data[idx];
    for (int i = 0; i < iterations; i++) {
        val = __hadd(__float2half(__half2float(val) * 1.1f), __float2half(0.1f));
        val = __float2half(__half2float(val) * (2.0f - __half2float(val)));
    }
    data[idx] = val;
}

__global__ void instructionThroughputFP32(float* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float val = data[idx];
    for (int i = 0; i < iterations; i++) {
        val = val * 1.1f + 0.1f;
        val = val * (2.0f - val);
        val = sinf(val * 0.1f);
        val = sqrtf(fabsf(val));
    }
    data[idx] = val;
}

__global__ void instructionThroughputFP64(double* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    double val = data[idx];
    for (int i = 0; i < iterations; i++) {
        val = val * 1.1 + 0.1;
        val = val * (2.0 - val);
        val = sin(val * 0.1);
        val = sqrt(fabs(val));
    }
    data[idx] = val;
}

__global__ void occupancyTestKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Light computation to test occupancy
    float val = input[idx];
    for (int i = 0; i < 100; i++) {
        val = val * 1.01f + 0.001f;
    }
    output[idx] = val;
}

__global__ void occupancyTestKernelLowOccupancy(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Use lots of registers to reduce occupancy
    float regs[32];
    for (int i = 0; i < 32; i++) {
        regs[i] = input[idx] + i * 0.1f;
    }
    
    float val = 0;
    for (int i = 0; i < 32; i++) {
        val += regs[i] * regs[(i + 1) % 32];
    }
    
    output[idx] = val;
}

__global__ void contextSwitchKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Minimal work to test context switch overhead
    data[idx] = data[idx] * 1.001f;
}

__global__ void ilpTestKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Multiple independent operations to test ILP
    float val1 = input[idx];
    float val2 = input[idx] * 2.0f;
    float val3 = input[idx] * 3.0f;
    float val4 = input[idx] * 4.0f;
    
    // Independent arithmetic chains
    val1 = val1 * 1.1f + 0.1f;
    val2 = val2 * 1.2f + 0.2f;
    val3 = val3 * 1.3f + 0.3f;
    val4 = val4 * 1.4f + 0.4f;
    
    val1 = sinf(val1);
    val2 = cosf(val2);
    val3 = sqrtf(fabsf(val3));
    val4 = expf(val4 * 0.1f);
    
    output[idx] = val1 + val2 + val3 + val4;
}

__global__ void divergenceTestKernel(int* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float result = 0.0f;
    int val = input[idx];
    
    // Create divergent control flow
    if (val % 32 < 16) {
        // Half of warp takes this path
        for (int i = 0; i < 100; i++) {
            result += sinf(val * i * 0.01f);
        }
    } else {
        // Other half takes this path
        for (int i = 0; i < 100; i++) {
            result += cosf(val * i * 0.01f);
        }
    }
    
    // Additional divergence within branches
    if (val % 8 < 4) {
        result *= 2.0f;
    } else {
        result *= 0.5f;
    }
    
    output[idx] = result;
}

__global__ void dynamicParallelismChild(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    data[idx] = data[idx] + 0.1f;
    
    if (depth > 0) {
        // Recursive launch
        dim3 childGrid((size + 255) / 256);
        dim3 childBlock(256);
        dynamicParallelismChild<<<childGrid, childBlock>>>(data, size, depth - 1);
    }
}

__global__ void dynamicParallelismParent(float* data, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    data[idx] = data[idx] * 1.1f;
    
    if (depth > 0) {
        // Launch child kernel
        dim3 childGrid((size + 255) / 256);
        dim3 childBlock(256);
        dynamicParallelismChild<<<childGrid, childBlock>>>(data, size, depth - 1);
    }
}

__global__ void flatKernelEquivalent(float* data, int size, int operations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    data[idx] = data[idx] * 1.1f; // Parent operation
    for (int i = 0; i < operations; i++) {
        data[idx] = data[idx] + 0.1f; // Child operations
    }
}

// Traditional kernel - launched many times
__global__ void traditional_work_kernel(float* input, float* output, int n, int work_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float result = input[idx];
    // Simulate work
    for (int i = 0; i < work_per_thread; i++) {
        result = sinf(result * 1.1f) + cosf(result * 0.9f);
    }
    output[idx] = result;
}

// Persistent kernel - launched once, processes multiple work items
__global__ void persistent_work_kernel(float* input, float* output, int* work_queue, 
                                     int* work_count, int total_work, int work_per_thread) {
    
    while (true) {
        // Atomically get next work item
        int work_idx = atomicAdd(work_count, 1);
        if (work_idx >= total_work) break;
        
        float result = input[work_idx];
        // Simulate work
        for (int i = 0; i < work_per_thread; i++) {
            result = sinf(result * 1.1f) + cosf(result * 0.9f);
        }
        output[work_idx] = result;
    }
}

// Coherent texture access pattern (good cache efficiency)
__global__ void texture_coherent_access(cudaTextureObject_t tex, float* output, 
                                       int width, int height, int* indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx >= total_pixels) return;
    
    // Sequential access pattern - good for cache
    int x = idx % width;
    int y = idx / width;
    
    float sum = 0.0f;
    // Sample texture with neighboring pixels (cache-friendly)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float u = (float)(x + dx) / width;
            float v = (float)(y + dy) / height;
            sum += tex2D<float>(tex, u, v);
        }
    }
    output[idx] = sum;
}

// Random texture access pattern (poor cache efficiency)
__global__ void texture_random_access(cudaTextureObject_t tex, float* output, 
                                     int width, int height, int* indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx >= total_pixels) return;
    
    float sum = 0.0f;
    // Random access pattern - poor for cache
    for (int i = 0; i < 9; i++) {
        int random_idx = (indices[idx] + i * 1237) % total_pixels;
        int x = random_idx % width;
        int y = random_idx / width;
        float u = (float)x / width;
        float v = (float)y / height;
        sum += tex2D<float>(tex, u, v);
    }
    output[idx] = sum;
}

// High register usage kernel to force spilling
__global__ void high_register_pressure_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Use many local variables to increase register pressure
    float r0 = input[idx];
    float r1 = r0 * 1.1f;
    float r2 = r1 + 2.2f;
    float r3 = r2 * 3.3f;
    float r4 = r3 + 4.4f;
    float r5 = r4 * 5.5f;
    float r6 = r5 + 6.6f;
    float r7 = r6 * 7.7f;
    float r8 = r7 + 8.8f;
    float r9 = r8 * 9.9f;
    float r10 = r9 + 10.1f;
    float r11 = r10 * 11.11f;
    float r12 = r11 + 12.12f;
    float r13 = r12 * 13.13f;
    float r14 = r13 + 14.14f;
    float r15 = r14 * 15.15f;
    float r16 = r15 + 16.16f;
    float r17 = r16 * 17.17f;
    float r18 = r17 + 18.18f;
    float r19 = r18 * 19.19f;
    float r20 = r19 + 20.20f;
    
    // More registers to force spilling
    float r21 = sinf(r0 + r1);
    float r22 = cosf(r2 + r3);
    float r23 = expf(r4 * 0.01f);
    float r24 = logf(fabsf(r5) + 1.0f);
    float r25 = sqrtf(fabsf(r6) + 1.0f);
    float r26 = r7 * r8;
    float r27 = r9 + r10;
    float r28 = r11 - r12;
    float r29 = r13 / (r14 + 1.0f);
    float r30 = r15 * r16 + r17;
    
    // Complex computation using all registers
    float result = 0.0f;
    result += r0 + r1 + r2 + r3 + r4;
    result += r5 + r6 + r7 + r8 + r9;
    result += r10 + r11 + r12 + r13 + r14;
    result += r15 + r16 + r17 + r18 + r19;
    result += r20 + r21 + r22 + r23 + r24;
    result += r25 + r26 + r27 + r28 + r29 + r30;
    
    // More computation to prevent compiler optimization
    for (int i = 0; i < 10; i++) {
        result = sinf(result * 0.1f) + cosf(result * 0.2f);
    }
    
    output[idx] = result;
}

// Low register usage kernel for comparison
__global__ void low_register_pressure_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float result = input[idx];
    
    // Simple computation with minimal registers
    for (int i = 0; i < 100; i++) {
        result = sinf(result * 0.1f) + cosf(result * 0.2f);
    }
    
    output[idx] = result;
}

// L1 Cache Latency Test Kernel
__global__ void l1CacheLatencyKernel(float* data, int* indices, float* output, int size, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    float sum = 0.0f;
    int idx = tid;
    
    // Chain of dependent loads to measure L1 latency
    for (int i = 0; i < iterations; i++) {
        sum += data[idx % size];
        idx = indices[idx % size];
    }
    
    output[tid] = sum;
}

// L2 Cache Latency Test Kernel
__global__ void l2CacheLatencyKernel(float* data, float* output, int size, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    float sum = 0.0f;
    int idx = tid;
    
    // Large stride access pattern to bypass L1 and hit L2
    for (int i = 0; i < 1000; i++) {
        idx = (idx + stride) % size;
        sum += data[idx];
    }
    
    output[tid] = sum;
}

// Shared Memory Latency Test Kernel
__global__ void sharedMemoryLatencyKernel(float* output, int iterations) {
    __shared__ float shared_data[1024];
    int tid = threadIdx.x;
    
    // Initialize shared memory
    if (tid < 1024) {
        shared_data[tid] = tid * 1.0f;
    }
    __syncthreads();
    
    float sum = 0.0f;
    int idx = tid;
    
    // Chain of dependent shared memory accesses
    for (int i = 0; i < iterations; i++) {
        sum += shared_data[idx % 1024];
        idx = (idx + 1) % 1024;
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

// Bank conflict test kernels
__global__ void bankConflictKernel(float* input, float* output, int n, int stride) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    if (idx < n) {
        // Load with potential bank conflicts based on stride
        int shared_idx = (tid * stride) % blockDim.x;
        sdata[shared_idx] = input[idx];
        
        __syncthreads();
        
        // Perform some computation
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            int read_idx = (shared_idx + i) % blockDim.x;
            sum += sdata[read_idx] * 0.1f;
        }
        
        __syncthreads();
        
        // Store back with potential conflicts
        sdata[shared_idx] = sum;
        
        __syncthreads();
        
        output[idx] = sdata[shared_idx];
    }
}

// Persistent occupancy kernel that keeps running
__global__ void persistentOccupancyKernel(float* input, float* output, int* work_flags, int n, int iterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int sm_id = blockIdx.x % 32; // Assuming max 32 SMs
    
    // Each block works persistently until signaled to stop
    while (work_flags[sm_id] > 0) {
        for (int iter = 0; iter < iterations; iter++) {
            if (tid < n) {
                float val = input[tid];
                
                // Simulate compute work
                for (int i = 0; i < 100; i++) {
                    val = sinf(val * 1.1f) + cosf(val * 0.9f);
                }
                
                output[tid] = val;
            }
            
            // Decrement work counter
            if (threadIdx.x == 0) {
                atomicSub(&work_flags[sm_id], 1);
            }
            __syncthreads();
        }
    }
}

// Simple kernels for CUDA graph testing
__global__ void graphKernel1(float* input, float* temp, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        temp[idx] = sinf(input[idx]) * 2.0f;
    }
}

__global__ void graphKernel2(float* temp, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = cosf(temp[idx]) + 1.0f;
    }
}

__global__ void graphKernel3(float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = sqrtf(fabsf(output[idx]));
    }
}

__global__ void thermalStressKernel(float* data, int size, int intensity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float value = data[idx];
    
    // High-intensity compute workload to stress SMs
    for (int i = 0; i < intensity; i++) {
        value = sinf(value) * cosf(value) + sqrtf(fabsf(value));
        value = expf(value * 0.001f) + logf(fabsf(value) + 1.0f);
        value = powf(fabsf(value), 0.7f) + tanhf(value);
        
        // Add some transcendental functions
        value = atanf(value) + sinhf(value * 0.1f);
        value = coshf(value * 0.1f) + erfcf(value * 0.01f);
        
        // Memory operations to increase power draw
        if (i % 10 == 0) {
            data[idx] = value;
            value = data[idx] + 0.001f;
        }
    }
    
    data[idx] = value;
}

// BVH Traversal kernel
__device__ bool rayAABBIntersect(const Ray& ray, const float3& min_bound, const float3& max_bound) {
    float tmin = (min_bound.x - ray.ox) / ray.dx;
    float tmax = (max_bound.x - ray.ox) / ray.dx;
    if (tmin > tmax) { float temp = tmin; tmin = tmax; tmax = temp; }
    
    float tymin = (min_bound.y - ray.oy) / ray.dy;
    float tymax = (max_bound.y - ray.oy) / ray.dy;
    if (tymin > tymax) { float temp = tymin; tymin = tymax; tymax = temp; }
    
    if (tmin > tymax || tymin > tmax) return false;
    
    tmin = fmaxf(tmin, tymin);
    tmax = fminf(tmax, tymax);
    
    float tzmin = (min_bound.z - ray.oz) / ray.dz;
    float tzmax = (max_bound.z - ray.oz) / ray.dz;
    if (tzmin > tzmax) { float temp = tzmin; tzmin = tzmax; tzmax = temp; }
    
    if (tmin > tzmax || tzmin > tmax) return false;
    
    return tmax >= 0;
}

__global__ void bvhTraversalKernel(const BVHNode* nodes, const Ray* rays, int* hit_results, 
                                   float* hit_distances, int num_rays, int num_nodes) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;
    
    Ray ray = rays[ray_idx];
    int stack[64]; // Stack for traversal
    int stack_ptr = 0;
    stack[0] = 0; // Start with root node
    
    float closest_hit = FLT_MAX;
    int hit_triangle = -1;
    
    while (stack_ptr >= 0) {
        int node_idx = stack[stack_ptr--];
        if (node_idx >= num_nodes || node_idx < 0) continue;
        
        const BVHNode& node = nodes[node_idx];
        
        if (rayAABBIntersect(ray, node.min_bound, node.max_bound)) {
            if (node.left_child == -1) {
                // Leaf node - simplified triangle intersection
                float t = length(node.min_bound) + (float)node.triangle_count * 0.1f;
                if (t < closest_hit) {
                    closest_hit = t;
                    hit_triangle = node.triangle_start;
                }
            } else {
                // Internal node - add children to stack
                if (stack_ptr < 62) {
                    if (node.right_child < num_nodes) stack[++stack_ptr] = node.right_child;
                    if (node.left_child < num_nodes) stack[++stack_ptr] = node.left_child;
                }
            }
        }
    }
    
    hit_results[ray_idx] = hit_triangle;
    hit_distances[ray_idx] = closest_hit;
}

// Kernel with low register usage (should achieve high occupancy)
__global__ void lowRegisterKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val * 2.0f + 1.0f;
    }
}

// Kernel with high register usage (should limit occupancy)
__global__ void highRegisterKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Use many registers to force spilling and reduce occupancy
        float r0 = input[idx];
        float r1 = r0 * 1.1f + 0.1f;
        float r2 = r1 * 1.2f + 0.2f;
        float r3 = r2 * 1.3f + 0.3f;
        float r4 = r3 * 1.4f + 0.4f;
        float r5 = r4 * 1.5f + 0.5f;
        float r6 = r5 * 1.6f + 0.6f;
        float r7 = r6 * 1.7f + 0.7f;
        float r8 = r7 * 1.8f + 0.8f;
        float r9 = r8 * 1.9f + 0.9f;
        float r10 = r9 * 2.0f + 1.0f;
        float r11 = r10 * 2.1f + 1.1f;
        float r12 = r11 * 2.2f + 1.2f;
        float r13 = r12 * 2.3f + 1.3f;
        float r14 = r13 * 2.4f + 1.4f;
        float r15 = r14 * 2.5f + 1.5f;
        float r16 = r15 * 2.6f + 1.6f;
        float r17 = r16 * 2.7f + 1.7f;
        float r18 = r17 * 2.8f + 1.8f;
        float r19 = r18 * 2.9f + 1.9f;
        float r20 = r19 * 3.0f + 2.0f;
        
        // Complex computation to prevent optimization
        float result = sinf(r0) + cosf(r1) + sqrtf(r2) + logf(r3 + 1.0f) + 
                      expf(r4 * 0.1f) + tanf(r5 * 0.1f) + r6 * r7 + r8 / (r9 + 1.0f) +
                      r10 * r11 + r12 - r13 + r14 * r15 - r16 + r17 * r18 + r19 * r20;
        
        output[idx] = result;
    }
}

// Kernel with minimal shared memory usage
__global__ void lowSharedMemoryKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

// Kernel with high shared memory usage (should limit occupancy)
__global__ void highSharedMemoryKernel(float* input, float* output, int size) {
    extern __shared__ float shared_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Use large amounts of shared memory
    if (idx < size) {
        shared_data[tid] = input[idx];
        shared_data[tid + blockDim.x] = input[idx] * 2.0f;
        shared_data[tid + 2 * blockDim.x] = input[idx] * 3.0f;
        shared_data[tid + 3 * blockDim.x] = input[idx] * 4.0f;
    }
    
    __syncthreads();
    
    if (idx < size) {
        float result = 0.0f;
        // Access shared memory in a pattern that prevents optimization
        for (int i = 0; i < 4; i++) {
            result += shared_data[tid + i * blockDim.x] * (i + 1);
        }
        output[idx] = result;
    }
}

// Kernel with both high register and shared memory usage
__global__ void highRegisterAndSharedMemoryKernel(float* input, float* output, int size) {
    extern __shared__ float shared_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < size) {
        // High register usage
        float r0 = input[idx];
        float r1 = r0 * 1.1f + 0.1f;
        float r2 = r1 * 1.2f + 0.2f;
        float r3 = r2 * 1.3f + 0.3f;
        float r4 = r3 * 1.4f + 0.4f;
        float r5 = r4 * 1.5f + 0.5f;
        float r6 = r5 * 1.6f + 0.6f;
        float r7 = r6 * 1.7f + 0.7f;
        float r8 = r7 * 1.8f + 0.8f;
        float r9 = r8 * 1.9f + 0.9f;
        float r10 = r9 * 2.0f + 1.0f;
        
        // High shared memory usage
        shared_data[tid] = r0;
        shared_data[tid + blockDim.x] = r1;
        shared_data[tid + 2 * blockDim.x] = r2;
        shared_data[tid + 3 * blockDim.x] = r3;
        shared_data[tid + 4 * blockDim.x] = r4;
        shared_data[tid + 5 * blockDim.x] = r5;
    }
    
    __syncthreads();
    
    if (idx < size) {
        float result = 0.0f;
        for (int i = 0; i < 6; i++) {
            result += shared_data[tid + i * blockDim.x] * sinf((float)i);
        }
        output[idx] = result;
    }
}


__global__ void cg_sync_kernel(int rounds, int* out) {
    cg::grid_group grid = cg::this_grid();
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // each round perform a small amount of work and sync
    for (int r = 0; r < rounds; ++r) {
        // trivial work: increment an atomic counter per block
        if (threadIdx.x == 0) {
            atomicAdd(out, 1);
        }
        // synchronise across the whole grid
        grid.sync();
        // trivial per-thread op
        if (gid % 64 == 0) { /* no-op to avoid being optimised away */ }
    }
}

__global__ void wmma_gemm_fp16_kernel(half const* A, half const* B, float* C, int M, int N, int K) {
    // This kernel computes C = A * B using WMMA 16x16x16 tiles.
    // Each block computes one tile for simplicity.
    int tile_m = (blockIdx.x);
    int tile_n = (blockIdx.y);

    int row = tile_m * 16;
    int col = tile_n * 16;

    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        // load A tile
        load_matrix_sync(a_frag, A + row * K + k, K);
        load_matrix_sync(b_frag, B + k * N + col, N);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // store results
    // write the 16x16 tile out to C
    for (int i = 0; i < 16; ++i) {
        int r = row + i;
        if (r >= M) continue;
        for (int j = 0; j < 16; ++j) {
            int c = col + j;
            if (c >= N) continue;
            int idx = r * N + c;
            // convert fragment element index to value
            // the simplest: write acc_frag directly (frag layout abstract) by reading via accessor
            C[idx] = acc_frag.x[i*16 + j]; // note: frag layout is implementation defined; works on modern toolchains
        }
    }
}

__global__ void managed_touch_kernel(char* base, size_t total, size_t stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = blockDim.x * gridDim.x;
    for (size_t off = idx * stride; off < total; off += step * stride) {
        volatile char v = base[off];
        (void)v;
    }
}

// Multi-Dimensional Convolution Kernels
__global__ void conv2dKernel(const float* input, float* output, const float* kernel,
                             int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half_kernel = kernel_size / 2;
    float sum = 0.0f;
    
    for (int ky = -half_kernel; ky <= half_kernel; ky++) {
        for (int kx = -half_kernel; kx <= half_kernel; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                int input_idx = iy * width + ix;
                int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                sum += input[input_idx] * kernel[kernel_idx];
            }
        }
    }
    
    output[y * width + x] = sum;
}

__global__ void conv3dKernel(const float* input, float* output, const float* kernel,
                             int width, int height, int depth, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    int half_kernel = kernel_size / 2;
    float sum = 0.0f;
    
    for (int kz = -half_kernel; kz <= half_kernel; kz++) {
        for (int ky = -half_kernel; ky <= half_kernel; ky++) {
            for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                int iz = z + kz;
                
                if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth) {
                    int input_idx = iz * width * height + iy * width + ix;
                    int kernel_idx = (kz + half_kernel) * kernel_size * kernel_size + 
                                   (ky + half_kernel) * kernel_size + (kx + half_kernel);
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }
    
    output[z * width * height + y * width + x] = sum;
}

// BFS/SSSP Kernels
__global__ void bfsKernel(const int* offsets, const int* edges, int* levels, 
                         int* visited, int num_nodes, int current_level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    if (levels[tid] == current_level) {
        int start = offsets[tid];
        int end = (tid + 1 < num_nodes) ? offsets[tid + 1] : 0;
        
        for (int i = start; i < end; i++) {
            int neighbor = edges[i];
            if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
                levels[neighbor] = current_level + 1;
            }
        }
    }
}

__global__ void ssspKernel(const int* offsets, const int* edges, const int* weights,
                          int* distances, int* updated, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    int current_dist = distances[tid];
    if (current_dist == INT_MAX) return;
    
    int start = offsets[tid];
    int end = (tid + 1 < num_nodes) ? offsets[tid + 1] : 0;
    
    for (int i = start; i < end; i++) {
        int neighbor = edges[i];
        int new_dist = current_dist + weights[i];
        
        int old_dist = atomicMin(&distances[neighbor], new_dist);
        if (new_dist < old_dist) {
            *updated = 1;
        }
    }
}

// SIMT Performance Kernels
__global__ void simtUniformKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // All threads in warp execute same instructions - good SIMT
    float val = input[idx];
    val = val * 2.0f + 1.0f;
    val = sinf(val) + cosf(val);
    val = sqrtf(fabsf(val));
    val = expf(val * 0.01f);
    
    output[idx] = val;
}

__global__ void simtDivergentKernel(const float* input, float* output, 
                                   const int* branch_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float val = input[idx];
    int branch = branch_data[idx] % 32; // Force divergence within warps
    
    // Heavy divergence - poor SIMT utilization
    if (branch < 8) {
        for (int i = 0; i < 100; i++) {
            val = sinf(val * 1.1f);
        }
    } else if (branch < 16) {
        for (int i = 0; i < 100; i++) {
            val = cosf(val * 0.9f);
        }
    } else if (branch < 24) {
        for (int i = 0; i < 100; i++) {
            val = expf(val * 0.01f);
        }
    } else {
        for (int i = 0; i < 100; i++) {
            val = logf(fabsf(val) + 1.0f);
        }
    }
    
    output[idx] = val;
}

__global__ void simtBalancedKernel(const float* input, float* output, 
                                  const int* branch_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float val = input[idx];
    int branch = branch_data[idx];
    
    // All threads execute all branches but use results selectively - better SIMT
    float result1 = 0.0f, result2 = 0.0f, result3 = 0.0f, result4 = 0.0f;
    
    for (int i = 0; i < 100; i++) {
        result1 = sinf(val * 1.1f);
        result2 = cosf(val * 0.9f);
        result3 = expf(val * 0.01f);
        result4 = logf(fabsf(val) + 1.0f);
    }
    
    // Select result based on branch condition
    int sel = branch % 4;
    val = (sel == 0) ? result1 : (sel == 1) ? result2 : (sel == 2) ? result3 : result4;
    
    output[idx] = val;
}

// ============================================================================
// Power-efficiency workload kernels
//
// Two distinct compute patterns are exposed so the power-efficiency benchmark
// can compare a low-power "light" workload against a maximum-throughput
// "heavy" workload, in order to measure clock drop-off and GFLOPS/Watt.
//
//   lightWorkloadKernel : one FMA per iteration -> 2 FLOPs/iter, low ILP,
//                         intended to be launched on a partial grid (~1/4 of
//                         the SMs) so the GPU stays only partially occupied.
//
//   heavyWorkloadKernel : four FMAs per iteration -> 8 FLOPs/iter, high ILP,
//                         intended to be launched on a full grid to maximise
//                         power draw and trigger thermal / power-limit
//                         throttling.
//
// Both kernels periodically write back to memory to prevent the compiler from
// eliminating the computation as dead code.
// ============================================================================
__global__ void lightWorkloadKernel(float* data, int size, int ops_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float v = data[idx];
    #pragma unroll 8
    for (int i = 0; i < ops_per_thread; i++) {
        // 1 FMA per iteration -> 2 FLOPs/iter, intentionally low ILP
        v = fmaf(v, 1.0001f, 0.0001f);
    }
    // Volatile write-back to prevent DCE
    if ((idx & 0x3FF) == 0) data[idx] = v;
    data[idx] = v;
}

__global__ void heavyWorkloadKernel(float* data, int size, int ops_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float a = data[idx];
    float b = fmaf(a, 0.5f, 1.0f);
    float c = fmaf(a, 0.25f, 2.0f);
    float d = fmaf(a, 0.125f, 3.0f);

    #pragma unroll 8
    for (int i = 0; i < ops_per_thread; i++) {
        // 4 FMAs per iteration -> 8 FLOPs/iter, high ILP dependency chain
        a = fmaf(a, b, c);
        b = fmaf(b, c, d);
        c = fmaf(c, d, a);
        d = fmaf(d, a, b);
    }
    // Write-back so the compiler cannot eliminate the work
    data[idx] = a + b + c + d;
}

// A second heavy variant that combines FP32 FMAs with transcendental ops.
// This is the worst-case for power draw (mixed FPU + Special Function Unit
// usage) and is intended to push the GPU into its thermal/power envelope
// faster than pure FMAs.
__global__ void heavyMixedWorkloadKernel(float* data, int size, int ops_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float a = data[idx];
    float b = fmaf(a, 0.5f, 1.0f);
    float c = fmaf(a, 0.25f, 2.0f);
    float d = fmaf(a, 0.125f, 3.0f);

    #pragma unroll 4
    for (int i = 0; i < ops_per_thread; i++) {
        a = fmaf(a, b, c);
        b = fmaf(b, c, d);
        c = fmaf(c, d, a);
        d = fmaf(d, a, b);
        // Periodic transcendental to engage SFU (high power)
        if ((i & 0x7) == 0) {
            a = sinf(a) + cosf(b);
            b = tanhf(c * 0.1f);
        }
    }
    data[idx] = a + b + c + d;
}

// ============================================================================
// Softmax kernels — naive 3-pass vs online 1-pass (with warp shuffles).
// Both compute softmax(input, -1) row-wise; one warp (32 threads) per row.
// ============================================================================

// Naive 3-pass softmax:
//   pass 1 = row max
//   pass 2 = sum(exp(x - max))
//   pass 3 = normalize
__global__ void softmax_naive_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int row_length) {
    int row = blockIdx.x;
    int tid = threadIdx.x;            // 0..31
    const float* in_row  = input  + (size_t)row * row_length;
    float*       out_row = output + (size_t)row * row_length;

    // --- Pass 1: row max ---
    float local_max = -INFINITY;
    for (int i = tid; i < row_length; i += 32) {
        local_max = fmaxf(local_max, in_row[i]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
    }
    float row_max = local_max;

    // --- Pass 2: sum(exp(x - max)) ---
    float local_sum = 0.0f;
    for (int i = tid; i < row_length; i += 32) {
        local_sum += expf(in_row[i] - row_max);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);
    }
    float inv_sum = 1.0f / local_sum;

    // --- Pass 3: normalize ---
    for (int i = tid; i < row_length; i += 32) {
        out_row[i] = expf(in_row[i] - row_max) * inv_sum;
    }
}

// Online softmax (1-pass for (m, l); 2-pass over data because the output
// write requires a second sweep).  Each thread maintains its own running
// (m, l) over its strided elements; a single warp reduction merges the 32
// per-thread (m, l) pairs into the row (m, l) using the online merge rule:
//     m = max(m_a, m_b)
//     l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
// This is the same primitive that flash-attention uses internally, and
// cuts the global-memory traffic for the reduction phase in half vs. naive.
__global__ void softmax_online_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int row_length) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* in_row  = input  + (size_t)row * row_length;
    float*       out_row = output + (size_t)row * row_length;

    // --- Phase 1: per-thread running (m, l) ---
    float m_local = -INFINITY;
    float l_local = 0.0f;
    for (int i = tid; i < row_length; i += 32) {
        float x = in_row[i];
        float m_new = fmaxf(m_local, x);
        l_local = l_local * expf(m_local - m_new) + expf(x - m_new);
        m_local = m_new;
    }

    // --- Warp reduce (m, l) pairs ---
    for (int off = 16; off > 0; off >>= 1) {
        float m_other = __shfl_xor_sync(0xffffffff, m_local, off);
        float l_other = __shfl_xor_sync(0xffffffff, l_local, off);
        float m_new = fmaxf(m_local, m_other);
        l_local = l_local * expf(m_local - m_new) + l_other * expf(m_other - m_new);
        m_local = m_new;
    }
    float row_max = m_local;
    float inv_sum = 1.0f / l_local;

    // --- Phase 2: write normalized output ---
    for (int i = tid; i < row_length; i += 32) {
        out_row[i] = expf(in_row[i] - row_max) * inv_sum;
    }
}

// ============================================================================
// Naive attention — materializes the full M×N score matrix in HBM.
// Three kernel launches: QK^T → row-wise softmax → P@V.
// The intermediate S (and the in-place softmaxed P) each cost M*N floats of
// HBM bandwidth — this is exactly the overhead flash-attention eliminates.
// ============================================================================

__global__ void attention_qk_kernel(const float* __restrict__ Q,  // [M, D]
                                    const float* __restrict__ K,  // [N, D]
                                    float* __restrict__ S,         // [M, N]
                                    int M, int N, int D, float scale) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float dot = 0.0f;
    for (int d = 0; d < D; d++) {
        dot += Q[(size_t)row * D + d] * K[(size_t)col * D + d];
    }
    S[(size_t)row * N + col] = dot * scale;
}

__global__ void attention_pv_kernel(const float* __restrict__ P,  // [M, N]
                                    const float* __restrict__ V,  // [N, D]
                                    float* __restrict__ O,         // [M, D]
                                    int M, int N, int D) {
    int row = blockIdx.y;
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || d >= D) return;
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += P[(size_t)row * N + n] * V[(size_t)n * D + d];
    }
    O[(size_t)row * D + d] = sum;
}

// ============================================================================
// Flash attention (tiled, online softmax) — single kernel, no S materialized.
//
//   Computes  O = softmax(scale * Q @ K^T) @ V
//
// One block per Br-row tile of Q.  Each block iterates over Bc-wide tiles
// of K/V, maintains per-row running (m, l, O) in shared memory, and uses
// the online softmax merge rule to accumulate tile contributions without
// ever writing the full M×N score matrix to HBM.
//
// Tile sizes are compile-time #defines so the inner D-loop unrolls.
// ============================================================================

#define FA_BR 16    // rows of Q per block
#define FA_BC 64    // columns of K/V per tile
#define FA_D  64    // head dim (must match config.softmax_head_dim)

__global__ void flash_attention_kernel(const float* __restrict__ Q,  // [M, D]
                                       const float* __restrict__ K,  // [N, D]
                                       const float* __restrict__ V,  // [N, D]
                                       float* __restrict__ O,         // [M, D]
                                       int M, int N, float scale) {
    constexpr int Br = FA_BR;
    constexpr int Bc = FA_BC;
    constexpr int D  = FA_D;

    int q_row_start = blockIdx.x * Br;

    extern __shared__ float smem[];
    float* sK = smem;                       // [Bc, D]
    float* sV = sK + (size_t)Bc * D;        // [Bc, D]
    float* sS = sV + (size_t)Bc * D;        // [Br, Bc]
    float* sO = sS + (size_t)Br * Bc;       // [Br, D]
    float* sM = sO + (size_t)Br * D;        // [Br]
    float* sL = sM + Br;                    // [Br]

    int tid = threadIdx.x;
    int nt  = blockDim.x;

    // Init per-row running state
    for (int r = tid; r < Br; r += nt) {
        sM[r] = -INFINITY;
        sL[r] = 0.0f;
        for (int d = 0; d < D; d++) sO[r * D + d] = 0.0f;
    }
    __syncthreads();

    int valid_Br = std::min(Br, M - q_row_start);
    if (valid_Br <= 0) return;

    // Loop over K/V tiles
    for (int kv_start = 0; kv_start < N; kv_start += Bc) {
        int actual_Bc = std::min(Bc, N - kv_start);

        // --- Load K, V tiles into shared memory ---
        for (int i = tid; i < actual_Bc * D; i += nt) {
            int lr = i / D;
            int lc = i % D;
            sK[lr * D + lc] = K[(size_t)(kv_start + lr) * D + lc];
            sV[lr * D + lc] = V[(size_t)(kv_start + lr) * D + lc];
        }
        __syncthreads();

        // --- Compute scores sS[r, c] = scale * sum_d Q[r,d] * K[c,d] ---
        for (int idx = tid; idx < valid_Br * actual_Bc; idx += nt) {
            int r = idx / actual_Bc;
            int c = idx % actual_Bc;
            int qrow = q_row_start + r;
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                dot += Q[(size_t)qrow * D + d] * sK[c * D + d];
            }
            sS[r * Bc + c] = dot * scale;
        }
        __syncthreads();

        // --- Online softmax update per row ---
        for (int r = tid; r < valid_Br; r += nt) {
            float m_old = sM[r];
            // Find new max (running max + tile max)
            float m_new = m_old;
            for (int c = 0; c < actual_Bc; c++) {
                m_new = fmaxf(m_new, sS[r * Bc + c]);
            }
            float rescale = expf(m_old - m_new);
            // Rescale existing O and l
            #pragma unroll
            for (int d = 0; d < D; d++) sO[r * D + d] *= rescale;
            float l_new = sL[r] * rescale;
            // Accumulate new contributions
            for (int c = 0; c < actual_Bc; c++) {
                float p = expf(sS[r * Bc + c] - m_new);
                l_new += p;
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    sO[r * D + d] += p * sV[c * D + d];
                }
            }
            sM[r] = m_new;
            sL[r] = l_new;
        }
        __syncthreads();
    }

    // --- Final: O = O / l ---
    for (int r = tid; r < valid_Br; r += nt) {
        float inv_l = 1.0f / sL[r];
        #pragma unroll
        for (int d = 0; d < D; d++) {
            O[(size_t)(q_row_start + r) * D + d] = sO[r * D + d] * inv_l;
        }
    }
}

#undef FA_BR
#undef FA_BC
#undef FA_D

class RenderBenchmark {
private:
    BenchmarkConfig config;
    
    // Device memory
    float* d_framebuffer;
    Triangle* d_triangles;
    Particle* d_particles;
    Sphere* d_spheres;
    float* d_texture_input;
    float* d_texture_output;
    float* d_compute_data;
    
    // Host memory
    Triangle* h_triangles;
    Particle* h_particles;
    Sphere* h_spheres;
    float* h_dp_data;
    Light* h_lights;
    
    // CUDA events for timing
    cudaEvent_t start_event, stop_event;

    // Additional device memory
    float* d_position_buffer;
    float* d_normal_buffer;
    float* d_color_buffer;
    float* d_matrix_a;
    float* d_matrix_b;
    float* d_matrix_c;
    Light* d_lights;
    Triangle* d_tessellated_triangles;
    // Convolution memory
    float* d_conv_input;
    float* d_conv_output;
    float* d_conv_kernel;
    
    // Atomic operations memory
    int* d_atomic_counters;
    float* d_atomic_data;
    
    // Memory access test arrays
    float* d_global_memory;
    float* d_shared_test_input;
    float* d_shared_test_output;
    size_t size = 1 << 20;
    int iterations = 50;
    
    int num_lights = 50;
    int matrix_size = 512;

    // Instruction throughput test memory
    int* d_int_data;
    half* d_fp16_data;
    float* d_fp32_data;
    double* d_fp64_data;
    
    // Occupancy test memory
    float* d_occupancy_input;
    float* d_occupancy_output;
    
    // Tensor core test memory
    half* d_tensor_a;
    half* d_tensor_b;
    float* d_tensor_c;
    
    // ILP test memory
    float* d_ilp_input;
    float* d_ilp_output;
    
    // Divergence test memory
    int* d_divergence_input;
    float* d_divergence_output;

    // Dynamic parallelism memory
    float* d_dp_data;

    // Persistent threads test memory
    float* d_persistent_input;
    float* d_persistent_output;
    int* d_work_queue;
    int* d_work_count;

    // Texture cache test memory
    cudaArray* d_texture_array_coherent;
    cudaArray* d_texture_array_random;
    cudaTextureObject_t tex_coherent;
    cudaTextureObject_t tex_random;
    float* d_texture_cache_output;
    int* d_texture_indices_coherent;
    int* d_texture_indices_random;

    // Register pressure test memory
    float* d_register_input;
    float* d_register_output;

    // Memory latency test data
    float* d_latency_test_data;
    int* d_latency_indices;
    float* d_latency_output;
    
    // PCIe bandwidth test data
    float* h_pcie_pinned;
    float* h_pcie_pageable;
    float* d_pcie_data;
    
    // P2P communication data (if multi-GPU available)
    float* d_p2p_src;
    float* d_p2p_dst;
    int num_gpus;

    // Async execution test memory
    cudaStream_t* async_streams;
    float** d_async_inputs;
    float** d_async_outputs;
    cudaEvent_t* async_events;
    
    // Instruction mix test memory
    float* d_instmix_input;
    float* d_instmix_output;
    int* d_instmix_control_input;
    
    // Cache test memory  
    float* d_cache_data;
    int* d_cache_indices;
    float* d_cache_conflict_data;

    // Bank conflict test memory
    float* d_bank_conflict_input;
    float* d_bank_conflict_output;
    int bank_conflict_test_size = 1048576;

    // Persistent occupancy test memory
    float* d_persistent_occ_input;
    float* d_persistent_occ_output;
    int* d_persistent_work_flags;
    int persistent_occ_test_size = 1048576;

    // CUDA graph test memory
    float* d_graph_input;
    float* d_graph_output;
    float* d_graph_temp;
    cudaGraph_t cuda_graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t graph_stream;
    int graph_test_size = 1048576;

    // SM Utilization and Thermal Throttling
    float* d_thermal_workload;
    cudaEvent_t* thermal_events;
    int num_thermal_events;
    
    // BVH Traversal
    BVHNode* d_bvh_nodes;
    Ray* d_rays;
    int* d_hit_results;
    float* d_hit_distances;
    int num_bvh_nodes;
    int num_rays_bvh;
    
    // Memory Allocation Overhead
    cudaMemPool_t mempool;
    float** d_malloc_ptrs;
    float** d_malloc_async_ptrs;
    int num_alloc_tests;
    
    // Occupancy limiting test memory
    float* d_occupancy_limit_input;
    float* d_occupancy_limit_output;

    // Multi-Dimensional Convolution test memory
    float* d_conv2d_input;
    float* d_conv2d_output;
    float* d_conv2d_kernel;
    float* d_conv3d_input;
    float* d_conv3d_output;
    float* d_conv3d_kernel;
    int conv2d_size = 512;
    int conv3d_size = 128;

    // BFS/SSSP test memory
    int* d_graph_offsets;
    int* d_graph_edges;
    int* d_graph_weights;
    int* d_bfs_levels;
    int* d_bfs_visited;
    int* d_sssp_distances;
    int num_graph_nodes = 100000;
    int num_graph_edges = 500000;

    // SIMT performance test memory
    float* d_simt_uniform_input;
    float* d_simt_uniform_output;
    float* d_simt_divergent_input;
    float* d_simt_divergent_output;
    int* d_simt_branch_data;
    int simt_test_size = 1048576;

    // Softmax + Flash-Attention test memory
    float* d_softmax_input;     // [M, N] input for softmax microbenchmark
    float* d_softmax_output;    // [M, N] output for softmax microbenchmark
    float* d_attn_Q;            // [M, D]
    float* d_attn_K;            // [N, D]
    float* d_attn_V;            // [N, D]
    float* d_attn_S;            // [M, N] materialized score matrix (naive only)
    float* d_attn_O_naive;      // [M, D] naive attention output
    float* d_attn_O_flash;      // [M, D] flash attention output

public:
    RenderBenchmark(const BenchmarkConfig& cfg) : config(cfg) {
        // Create CUDA events
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        
        // Allocate device memory
        size_t framebuffer_size = config.width * config.height * 3 * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_framebuffer, framebuffer_size));
        
        CUDA_CHECK(cudaMalloc(&d_triangles, config.num_triangles * sizeof(Triangle)));
        CUDA_CHECK(cudaMalloc(&d_particles, config.num_particles * sizeof(Particle)));
        CUDA_CHECK(cudaMalloc(&d_spheres, config.num_triangles * sizeof(Sphere)));
        
        size_t texture_size = config.texture_size * config.texture_size * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_texture_input, texture_size));
        CUDA_CHECK(cudaMalloc(&d_texture_output, texture_size));
        
        CUDA_CHECK(cudaMalloc(&d_compute_data, config.num_particles * sizeof(float)));

        size_t buffer_size = config.width * config.height * 3 * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_position_buffer, buffer_size));
        CUDA_CHECK(cudaMalloc(&d_normal_buffer, buffer_size));
        CUDA_CHECK(cudaMalloc(&d_color_buffer, buffer_size));
        
        CUDA_CHECK(cudaMalloc(&d_matrix_a, matrix_size * matrix_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_matrix_b, matrix_size * matrix_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_matrix_c, matrix_size * matrix_size * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_lights, num_lights * sizeof(Light)));
        CUDA_CHECK(cudaMalloc(&d_tessellated_triangles, config.num_triangles * 16 * sizeof(Triangle)));

        CUDA_CHECK(cudaMalloc(&d_atomic_counters, config.atomic_test_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_atomic_data, config.atomic_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_global_memory, config.memory_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_shared_test_input, config.memory_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_shared_test_output, config.memory_test_size * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_int_data, config.instruction_test_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_fp16_data, config.instruction_test_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_fp32_data, config.instruction_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fp64_data, config.instruction_test_size * sizeof(double)));
        
        CUDA_CHECK(cudaMalloc(&d_occupancy_input, config.occupancy_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_occupancy_output, config.occupancy_test_size * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_tensor_a, config.tensor_size * config.tensor_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_tensor_b, config.tensor_size * config.tensor_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_tensor_c, config.tensor_size * config.tensor_size * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_ilp_input, config.ilp_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ilp_output, config.ilp_test_size * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_divergence_input, config.divergence_test_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_divergence_output, config.divergence_test_size * sizeof(float)));

        // Async execution memory
        async_streams = new cudaStream_t[config.async_streams];
        d_async_inputs = new float*[config.async_streams];
        d_async_outputs = new float*[config.async_streams];
        async_events = new cudaEvent_t[config.async_streams * 2];
        
        for (int i = 0; i < config.async_streams; i++) {
            CUDA_CHECK(cudaStreamCreate(&async_streams[i]));
            CUDA_CHECK(cudaMalloc(&d_async_inputs[i], config.async_test_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_async_outputs[i], config.async_test_size * sizeof(float)));
            CUDA_CHECK(cudaEventCreate(&async_events[i * 2]));
            CUDA_CHECK(cudaEventCreate(&async_events[i * 2 + 1]));
        }
        
        // Instruction mix memory
        CUDA_CHECK(cudaMalloc(&d_instmix_input, config.instruction_mix_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_instmix_output, config.instruction_mix_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_instmix_control_input, config.instruction_mix_size * sizeof(int)));
        
        // Cache test memory
        CUDA_CHECK(cudaMalloc(&d_cache_data, config.cache_thrash_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cache_indices, config.cache_thrash_size * 4 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cache_conflict_data, config.cache_thrash_size * sizeof(float)));

        size_t dp_size = config.dp_test_size * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_dp_data, dp_size));

        h_dp_data = (float*)malloc(dp_size);
    
        // Initialize dynamic parallelism test data
        for (int i = 0; i < config.dp_test_size; i++) {
            h_dp_data[i] = 1.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_dp_data, h_dp_data, dp_size, cudaMemcpyHostToDevice));

        // Memory latency test allocations
        CUDA_CHECK(cudaMalloc(&d_latency_test_data, config.memory_latency_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_latency_indices, config.memory_latency_test_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_latency_output, config.memory_latency_test_size * sizeof(float)));
        
        // PCIe bandwidth test allocations
        CUDA_CHECK(cudaMallocHost(&h_pcie_pinned, config.pcie_transfer_size * sizeof(float)));
        h_pcie_pageable = new float[config.pcie_transfer_size];
        CUDA_CHECK(cudaMalloc(&d_pcie_data, config.pcie_transfer_size * sizeof(float)));
        
        // Check for multi-GPU and allocate P2P memory
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        if (num_gpus > 1) {
            CUDA_CHECK(cudaMalloc(&d_p2p_src, config.p2p_transfer_size * sizeof(float)));
            cudaSetDevice(1);
            CUDA_CHECK(cudaMalloc(&d_p2p_dst, config.p2p_transfer_size * sizeof(float)));
            cudaSetDevice(0);
        }

        // Bank conflict test allocations
        CUDA_CHECK(cudaMalloc(&d_bank_conflict_input, bank_conflict_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bank_conflict_output, bank_conflict_test_size * sizeof(float)));

        // Persistent occupancy test allocations  
        CUDA_CHECK(cudaMalloc(&d_persistent_occ_input, persistent_occ_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_persistent_occ_output, persistent_occ_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_persistent_work_flags, 32 * sizeof(int))); // One flag per SM

        // CUDA graph test allocations
        CUDA_CHECK(cudaMalloc(&d_graph_input, graph_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_graph_output, graph_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_graph_temp, graph_test_size * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&graph_stream));

        h_lights = new Light[num_lights];
        initializeLights();
        
        // Allocate host memory
        h_triangles = new Triangle[config.num_triangles];
        h_particles = new Particle[config.num_particles];
        h_spheres = new Sphere[config.num_triangles];
        
        initializeData();

        // SM Utilization and Thermal Throttling
        CUDA_CHECK(cudaMalloc(&d_thermal_workload, config.instruction_test_size * sizeof(float)));
        num_thermal_events = 100;
        thermal_events = new cudaEvent_t[num_thermal_events];
        for (int i = 0; i < num_thermal_events; i++) {
            CUDA_CHECK(cudaEventCreate(&thermal_events[i]));
        }
        
        // BVH Traversal
        num_bvh_nodes = 10000;
        num_rays_bvh = 100000;
        CUDA_CHECK(cudaMalloc(&d_bvh_nodes, num_bvh_nodes * sizeof(BVHNode)));
        CUDA_CHECK(cudaMalloc(&d_rays, num_rays_bvh * sizeof(Ray)));
        CUDA_CHECK(cudaMalloc(&d_hit_results, num_rays_bvh * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_hit_distances, num_rays_bvh * sizeof(float)));
        
        // Memory Allocation Overhead
        num_alloc_tests = 1000;
        d_malloc_ptrs = new float*[num_alloc_tests];
        d_malloc_async_ptrs = new float*[num_alloc_tests];
        
        // Create memory pool for async allocations
        cudaMemPoolProps poolProps = {};
        poolProps.allocType = cudaMemAllocationTypePinned;
        poolProps.handleTypes = cudaMemHandleTypeNone;
        poolProps.location.type = cudaMemLocationTypeDevice;
        poolProps.location.id = 0;
        CUDA_CHECK(cudaMemPoolCreate(&mempool, &poolProps));
        
        // Occupancy Limiting
        CUDA_CHECK(cudaMalloc(&d_occupancy_limit_input, config.occupancy_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_occupancy_limit_output, config.occupancy_test_size * sizeof(float)));
        
        initializeBVHData();

        // Multi-Dimensional Convolution allocations
        CUDA_CHECK(cudaMalloc(&d_conv2d_input, conv2d_size * conv2d_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2d_output, conv2d_size * conv2d_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2d_kernel, 5 * 5 * sizeof(float))); // 5x5 kernel

        CUDA_CHECK(cudaMalloc(&d_conv3d_input, conv3d_size * conv3d_size * conv3d_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv3d_output, conv3d_size * conv3d_size * conv3d_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv3d_kernel, 3 * 3 * 3 * sizeof(float))); // 3x3x3 kernel

        // BFS/SSSP allocations
        CUDA_CHECK(cudaMalloc(&d_graph_offsets, (num_graph_nodes + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_graph_edges, num_graph_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_graph_weights, num_graph_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bfs_levels, num_graph_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bfs_visited, num_graph_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sssp_distances, num_graph_nodes * sizeof(int)));

        // SIMT performance test allocations
        CUDA_CHECK(cudaMalloc(&d_simt_uniform_input, simt_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_simt_uniform_output, simt_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_simt_divergent_input, simt_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_simt_divergent_output, simt_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_simt_branch_data, simt_test_size * sizeof(int)));

        // Initialize convolution data
        std::vector<float> h_conv2d_input(conv2d_size * conv2d_size);
        std::vector<float> h_conv3d_input(conv3d_size * conv3d_size * conv3d_size);
        for (auto& val : h_conv2d_input) val = (float)rand() / RAND_MAX;
        for (auto& val : h_conv3d_input) val = (float)rand() / RAND_MAX;
        CUDA_CHECK(cudaMemcpy(d_conv2d_input, h_conv2d_input.data(), 
                            conv2d_size * conv2d_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv3d_input, h_conv3d_input.data(), 
                            conv3d_size * conv3d_size * conv3d_size * sizeof(float), cudaMemcpyHostToDevice));

        // Initialize graph data (random graph)
        std::vector<int> h_offsets(num_graph_nodes + 1);
        std::vector<int> h_edges(num_graph_edges);
        std::vector<int> h_weights(num_graph_edges);
        int edge_count = 0;
        for (int i = 0; i < num_graph_nodes; i++) {
            h_offsets[i] = edge_count;
            int degree = rand() % 10 + 1; // 1-10 edges per node
            for (int j = 0; j < degree && edge_count < num_graph_edges; j++) {
                h_edges[edge_count] = rand() % num_graph_nodes;
                h_weights[edge_count] = rand() % 100 + 1;
                edge_count++;
            }
        }
        h_offsets[num_graph_nodes] = edge_count;
        CUDA_CHECK(cudaMemcpy(d_graph_offsets, h_offsets.data(), 
                            (num_graph_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_graph_edges, h_edges.data(), 
                            num_graph_edges * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_graph_weights, h_weights.data(), 
                            num_graph_edges * sizeof(int), cudaMemcpyHostToDevice));

        // Initialize SIMT test data
        std::vector<float> h_simt_input(simt_test_size);
        std::vector<int> h_simt_branch(simt_test_size);
        for (int i = 0; i < simt_test_size; i++) {
            h_simt_input[i] = (float)rand() / RAND_MAX;
            h_simt_branch[i] = rand();
        }
        CUDA_CHECK(cudaMemcpy(d_simt_uniform_input, h_simt_input.data(),
                            simt_test_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_simt_divergent_input, h_simt_input.data(),
                            simt_test_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_simt_branch_data, h_simt_branch.data(),
                            simt_test_size * sizeof(int), cudaMemcpyHostToDevice));

        // -------------------------------------------------------------------
        // Softmax + Flash-Attention allocations & initialisation
        // -------------------------------------------------------------------
        // Sanity: the flash-attention kernel is compiled with FA_D == 64.
        // If the user changed softmax_head_dim, the kernel will silently
        // produce wrong results — guard against that here.
        if (config.softmax_head_dim != 64) {
            fprintf(stderr,
                "WARNING: config.softmax_head_dim=%d but flash_attention_kernel "
                "is compiled with D=64.  Flash attention results will be wrong. "
                "Set softmax_head_dim=64 or recompile with FA_D=%d.\n",
                config.softmax_head_dim, config.softmax_head_dim);
        }

        // Use the same M, N, D across all softmax/attention kernels so the
        // numbers are directly comparable.  N for the softmax microbenchmark
        // is the sequence length; M is the number of rows.
        const int sm_M = config.softmax_batch_heads;
        const int sm_N = config.softmax_seq_len;
        const int sm_D = config.softmax_head_dim;

        CUDA_CHECK(cudaMalloc(&d_softmax_input,  (size_t)sm_M * sm_N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_softmax_output, (size_t)sm_M * sm_N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn_Q,         (size_t)sm_M * sm_D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn_K,         (size_t)sm_N * sm_D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn_V,         (size_t)sm_N * sm_D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn_S,         (size_t)sm_M * sm_N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn_O_naive,   (size_t)sm_M * sm_D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn_O_flash,   (size_t)sm_M * sm_D * sizeof(float)));

        // Fill softmax input and Q/K/V with small random values.  Using a
        // narrow range keeps expf() well-behaved (no overflow / underflow)
        // so the numerical comparison between naive and flash is meaningful.
        std::vector<float> h_sm_input((size_t)sm_M * sm_N);
        std::vector<float> h_qkv((size_t)sm_M * sm_D);
        std::vector<float> h_kv((size_t)sm_N * sm_D);
        for (auto& v : h_sm_input) v = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;  // [-2, 2)
        for (auto& v : h_qkv)     v = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;  // [-0.2, 0.2)
        for (auto& v : h_kv)      v = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;
        CUDA_CHECK(cudaMemcpy(d_softmax_input, h_sm_input.data(),
                              (size_t)sm_M * sm_N * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_attn_Q, h_qkv.data(),
                              (size_t)sm_M * sm_D * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_attn_K, h_kv.data(),
                              (size_t)sm_N * sm_D * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_attn_V, h_kv.data(),
                              (size_t)sm_N * sm_D * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    
    ~RenderBenchmark() {
        // Free device memory
        cudaFree(d_framebuffer);
        cudaFree(d_triangles);
        cudaFree(d_particles);
        cudaFree(d_spheres);
        cudaFree(d_texture_input);
        cudaFree(d_texture_output);
        cudaFree(d_compute_data);
        cudaFree(d_position_buffer);
        cudaFree(d_normal_buffer);
        cudaFree(d_color_buffer);
        cudaFree(d_matrix_a);
        cudaFree(d_matrix_b);
        cudaFree(d_matrix_c);
        cudaFree(d_lights);
        cudaFree(d_tessellated_triangles);
        cudaFree(d_atomic_counters);
        cudaFree(d_atomic_data);
        cudaFree(d_int_data);
        cudaFree(d_fp16_data);
        cudaFree(d_fp32_data);
        cudaFree(d_fp64_data);
        cudaFree(d_occupancy_input);
        cudaFree(d_occupancy_output);
        cudaFree(d_tensor_a);
        cudaFree(d_tensor_b);
        cudaFree(d_tensor_c);
        cudaFree(d_ilp_input);
        cudaFree(d_ilp_output);
        cudaFree(d_divergence_input);
        cudaFree(d_divergence_output);
        cudaFree(d_dp_data);
        cudaFree(d_persistent_input);
        cudaFree(d_persistent_output);
        cudaFree(d_work_queue);
        cudaFree(d_work_count);
        cudaFree(d_texture_cache_output);
        cudaFree(d_texture_indices_coherent);
        cudaFree(d_texture_indices_random);
        cudaFree(d_register_input);
        cudaFree(d_register_output);
        cudaFree(d_latency_test_data);
        cudaFree(d_latency_indices);
        cudaFree(d_latency_output);
        cudaFree(d_pcie_data);
        cudaFree(d_instmix_input);
        cudaFree(d_instmix_output);
        cudaFree(d_instmix_control_input);
        cudaFree(d_cache_data);
        cudaFree(d_cache_indices);
        cudaFree(d_cache_conflict_data);
        cudaFree(d_bank_conflict_input);
        cudaFree(d_bank_conflict_output);
        cudaFree(d_persistent_occ_input);
        cudaFree(d_persistent_occ_output);
        cudaFree(d_persistent_work_flags);
        cudaFree(d_graph_input);
        cudaFree(d_graph_output);
        cudaFree(d_graph_temp);
        cudaFree(d_occupancy_limit_input);
        cudaFree(d_occupancy_limit_output);
        cudaFree(d_thermal_workload);
        cudaFree(d_bvh_nodes);
        cudaFree(d_rays);
        cudaFree(d_hit_results);
        cudaFree(d_hit_distances);

        for (int i = 0; i < num_thermal_events; i++) {
            cudaEventDestroy(thermal_events[i]);
        }
        delete[] thermal_events;

        delete[] d_malloc_ptrs;
        delete[] d_malloc_async_ptrs;
        cudaMemPoolDestroy(mempool);

        cudaStreamDestroy(graph_stream);
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (cuda_graph) cudaGraphDestroy(cuda_graph);
        free(h_dp_data);

        cudaDestroyTextureObject(tex_coherent);
        cudaDestroyTextureObject(tex_random);
        cudaFreeArray(d_texture_array_coherent);
        cudaFreeArray(d_texture_array_random);
        
        // Clean up PCIe test memory
        cudaFreeHost(h_pcie_pinned);
        delete[] h_pcie_pageable;
        
        // Clean up P2P memory
        if (num_gpus > 1) {
            cudaFree(d_p2p_src);
            cudaSetDevice(1);
            cudaFree(d_p2p_dst);
            cudaSetDevice(0);
        }

        // Clean up async execution memory
        for (int i = 0; i < config.async_streams; i++) {
            cudaStreamDestroy(async_streams[i]);
            cudaFree(d_async_inputs[i]);
            cudaFree(d_async_outputs[i]);
            cudaEventDestroy(async_events[i * 2]);
            cudaEventDestroy(async_events[i * 2 + 1]);
        }
        delete[] async_streams;
        delete[] d_async_inputs;
        delete[] d_async_outputs;
        delete[] async_events;
        delete[] h_triangles;
        delete[] h_particles;
        delete[] h_spheres;
        
        // Destroy events
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);

        cudaFree(d_conv2d_input);
        cudaFree(d_conv2d_output);
        cudaFree(d_conv2d_kernel);
        cudaFree(d_conv3d_input);
        cudaFree(d_conv3d_output);
        cudaFree(d_conv3d_kernel);
        cudaFree(d_graph_offsets);
        cudaFree(d_graph_edges);
        cudaFree(d_graph_weights);
        cudaFree(d_bfs_levels);
        cudaFree(d_bfs_visited);
        cudaFree(d_sssp_distances);
        cudaFree(d_simt_uniform_input);
        cudaFree(d_simt_uniform_output);
        cudaFree(d_simt_divergent_input);
        cudaFree(d_simt_divergent_output);
        cudaFree(d_simt_branch_data);

        // Softmax + Flash-Attention frees
        cudaFree(d_softmax_input);
        cudaFree(d_softmax_output);
        cudaFree(d_attn_Q);
        cudaFree(d_attn_K);
        cudaFree(d_attn_V);
        cudaFree(d_attn_S);
        cudaFree(d_attn_O_naive);
        cudaFree(d_attn_O_flash);
    }

    void initializeLights() {
        for (int i = 0; i < num_lights; i++) {
            h_lights[i] = {
                (float)rand() / RAND_MAX * 6.0f - 3.0f,  // x
                (float)rand() / RAND_MAX * 6.0f - 3.0f,  // y
                (float)rand() / RAND_MAX * 6.0f - 3.0f,  // z
                (float)rand() / RAND_MAX,  // r
                (float)rand() / RAND_MAX,  // g
                (float)rand() / RAND_MAX,  // b
                1.0f + (float)rand() / RAND_MAX * 2.0f,  // intensity
                0  // point light
            };
        }
        CUDA_CHECK(cudaMemcpy(d_lights, h_lights, num_lights * sizeof(Light), cudaMemcpyHostToDevice));
    }

    void initializeData() {
        // Initialize triangles
        for (int i = 0; i < config.num_triangles; i++) {
            float x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float y = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float z = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            float size = 0.1f;
            
            h_triangles[i].v0 = {x, y, z, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            h_triangles[i].v1 = {x + size, y, z, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
            h_triangles[i].v2 = {x, y + size, z, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f};
        }
        
        // Initialize particles
        for (int i = 0; i < config.num_particles; i++) {
            h_particles[i] = {
                (float)rand() / RAND_MAX * 2.0f - 1.0f,  // x
                (float)rand() / RAND_MAX * 2.0f - 1.0f,  // y
                (float)rand() / RAND_MAX * 2.0f - 1.0f,  // z
                ((float)rand() / RAND_MAX - 0.5f) * 4.0f,  // vx
                ((float)rand() / RAND_MAX - 0.5f) * 4.0f,  // vy
                ((float)rand() / RAND_MAX - 0.5f) * 4.0f,  // vz
                (float)rand() / RAND_MAX,  // r
                (float)rand() / RAND_MAX,  // g
                (float)rand() / RAND_MAX,  // b
                0.5f,  // a
                2.0f   // size
            };
        }
        
        // Initialize spheres
        for (int i = 0; i < config.num_triangles; i++) {
            h_spheres[i] = {
                (float)rand() / RAND_MAX * 4.0f - 2.0f,  // x
                (float)rand() / RAND_MAX * 4.0f - 2.0f,  // y
                (float)rand() / RAND_MAX * 4.0f - 2.0f,  // z
                0.2f + (float)rand() / RAND_MAX * 0.3f,  // radius
                (float)rand() / RAND_MAX,  // r
                (float)rand() / RAND_MAX,  // g
                (float)rand() / RAND_MAX   // b
            };
        }
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles, 
                             config.num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_particles, h_particles, 
                             config.num_particles * sizeof(Particle), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres, 
                             config.num_triangles * sizeof(Sphere), cudaMemcpyHostToDevice));
    }
    
    void initializeBVHData() {
        // Initialize BVH nodes on host
        BVHNode* h_bvh_nodes = new BVHNode[num_bvh_nodes];
        Ray* h_rays = new Ray[num_rays_bvh];
        
        // Create a simple binary BVH tree
        for (int i = 0; i < num_bvh_nodes; i++) {
            h_bvh_nodes[i].min_bound = make_float3(
                (float)rand() / RAND_MAX * 10.0f - 5.0f,
                (float)rand() / RAND_MAX * 10.0f - 5.0f,
                (float)rand() / RAND_MAX * 10.0f - 5.0f
            );
            h_bvh_nodes[i].max_bound = h_bvh_nodes[i].min_bound + 
                make_float3(1.0f + (float)rand() / RAND_MAX * 2.0f,
                           1.0f + (float)rand() / RAND_MAX * 2.0f,
                           1.0f + (float)rand() / RAND_MAX * 2.0f);
            
            // Simple tree structure
            if (i * 2 + 1 < num_bvh_nodes) {
                h_bvh_nodes[i].left_child = i * 2 + 1;
                h_bvh_nodes[i].right_child = i * 2 + 2;
                h_bvh_nodes[i].triangle_start = -1;
                h_bvh_nodes[i].triangle_count = 0;
            } else {
                h_bvh_nodes[i].left_child = -1;
                h_bvh_nodes[i].right_child = -1;
                h_bvh_nodes[i].triangle_start = i % config.num_triangles;
                h_bvh_nodes[i].triangle_count = 1 + (i % 3);
            }
        }
        
        // Initialize rays
        for (int i = 0; i < num_rays_bvh; i++) {
            h_rays[i].ox = (float)rand() / RAND_MAX * 10.0f - 5.0f;
            h_rays[i].oy = (float)rand() / RAND_MAX * 10.0f - 5.0f;
            h_rays[i].oz = (float)rand() / RAND_MAX * 10.0f - 5.0f;
            
            float3 dir = normalize(make_float3(
                (float)rand() / RAND_MAX * 2.0f - 1.0f,
                (float)rand() / RAND_MAX * 2.0f - 1.0f,
                (float)rand() / RAND_MAX * 2.0f - 1.0f
            ));
            h_rays[i].dx = dir.x;
            h_rays[i].dy = dir.y;
            h_rays[i].dz = dir.z;
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_bvh_nodes, h_bvh_nodes, num_bvh_nodes * sizeof(BVHNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rays, h_rays, num_rays_bvh * sizeof(Ray), cudaMemcpyHostToDevice));
        
        delete[] h_bvh_nodes;
        delete[] h_rays;
    }

    cusparseHandle_t handle;

    // Call this once (e.g. in ctor) to create the handle
    void initCuSparse() {
        CUSPARSE_CHECK(cusparseCreate(&handle));
    }

    // Call this once (e.g. in dtor) to destroy the handle
    void destroyCuSparse() {
        CUSPARSE_CHECK(cusparseDestroy(handle));
    }

    float timeKernelExecution(void (*kernelFunc)(RenderBenchmark*), RenderBenchmark* benchmark) {
        CUDA_CHECK(cudaEventRecord(start_event));
        kernelFunc(benchmark);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds;
    }
    
    static void rasterisationKernel(RenderBenchmark* benchmark) {
        dim3 blockSize(16, 16);
        dim3 gridSize((benchmark->config.width + blockSize.x - 1) / blockSize.x,
                     (benchmark->config.height + blockSize.y - 1) / blockSize.y);
        
        // Rendering kernels are O(pixels × objects) — 1920×1080 × 10000
        // triangles × 100 iterations would take minutes.  Cap at 5 iterations.
        int iters = std::max(5, benchmark->config.iterations / 20);
        for (int i = 0; i < iters; i++) {
            CUDA_CHECK(cudaMemset(benchmark->d_framebuffer, 0, 
                                benchmark->config.width * benchmark->config.height * 3 * sizeof(float)));
            rasteriseTriangles<<<gridSize, blockSize>>>(
                benchmark->d_triangles, benchmark->config.num_triangles, 
                benchmark->d_framebuffer, benchmark->config.width, benchmark->config.height);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    void benchmarkRasterisation() {
        printf("\n=== Rasterisation Benchmark ===\n");
        
        int iters = std::max(5, config.iterations / 20);
        float time = timeKernelExecution(rasterisationKernel, this);
        float fps = (iters * 1000.0f) / time;
        
        printf("Triangles: %d\n", config.num_triangles);
        printf("Resolution: %dx%d\n", config.width, config.height);
        printf("Iterations: %d\n", iters);
        printf("Time: %.2f ms (%.2f FPS)\n", time / iters, fps);
        printf("Triangles/sec: %.2f M\n", (config.num_triangles * fps) / 1e6f);
        printf("Pixels/sec: %.2f M\n", (config.width * config.height * fps) / 1e6f);
    }
    
    static void particleKernel(RenderBenchmark* benchmark) {
        dim3 blockSize(256);
        dim3 gridSize((benchmark->config.num_particles + blockSize.x - 1) / blockSize.x);
        
        for (int i = 0; i < benchmark->config.iterations; i++) {
            CUDA_CHECK(cudaMemset(benchmark->d_framebuffer, 0, 
                                benchmark->config.width * benchmark->config.height * 3 * sizeof(float)));
            renderParticles<<<gridSize, blockSize>>>(
                benchmark->d_particles, benchmark->config.num_particles, benchmark->d_framebuffer, 
                benchmark->config.width, benchmark->config.height, 0.016f);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    void benchmarkParticles() {
        printf("\n=== Particle Simulation Benchmark ===\n");
        
        float time = timeKernelExecution(particleKernel, this);
        float fps = (config.iterations * 1000.0f) / time;
        
        printf("Particles: %d\n", config.num_particles);
        printf("Time: %.2f ms (%.2f FPS)\n", time / config.iterations, fps);
        printf("Particles/sec: %.2f M\n", (config.num_particles * fps) / 1e6f);
    }
    
    static void rayTracingKernel(RenderBenchmark* benchmark) {
        dim3 blockSize(16, 16);
        dim3 gridSize((benchmark->config.width + blockSize.x - 1) / blockSize.x,
                     (benchmark->config.height + blockSize.y - 1) / blockSize.y);
        
        // Ray tracing is O(pixels × spheres) — cap at 5 iterations.
        int iters = std::max(5, benchmark->config.iterations / 20);
        for (int i = 0; i < iters; i++) {
            CUDA_CHECK(cudaMemset(benchmark->d_framebuffer, 0, 
                                benchmark->config.width * benchmark->config.height * 3 * sizeof(float)));
            rayTrace<<<gridSize, blockSize>>>(
                benchmark->d_spheres, benchmark->config.num_triangles, 
                benchmark->d_framebuffer, benchmark->config.width, benchmark->config.height);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    void benchmarkRayTracing() {
        printf("\n=== Ray Tracing Benchmark ===\n");
        
        int iters = std::max(5, config.iterations / 20);
        float time = timeKernelExecution(rayTracingKernel, this);
        float fps = (iters * 1000.0f) / time;
        
        printf("Spheres: %d\n", config.num_triangles);
        printf("Rays: %d\n", config.width * config.height);
        printf("Iterations: %d\n", iters);
        printf("Time: %.2f ms (%.2f FPS)\n", time / iters, fps);
        printf("Rays/sec: %.2f M\n", (config.width * config.height * fps) / 1e6f);
    }
    
    static void textureFilteringKernel(RenderBenchmark* benchmark) {
        dim3 blockSize(16, 16);
        dim3 gridSize((benchmark->config.texture_size + blockSize.x - 1) / blockSize.x,
                     (benchmark->config.texture_size + blockSize.y - 1) / blockSize.y);
        
        for (int i = 0; i < benchmark->config.iterations; i++) {
            textureFilter<<<gridSize, blockSize>>>(
                benchmark->d_texture_input, benchmark->d_texture_output, 
                benchmark->config.texture_size, benchmark->config.texture_size, 5);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    void benchmarkTextureFiltering() {
        printf("\n=== Texture Filtering Benchmark ===\n");
        
        float time = timeKernelExecution(textureFilteringKernel, this);
        float fps = (config.iterations * 1000.0f) / time;
        
        printf("Texture size: %dx%d\n", config.texture_size, config.texture_size);
        printf("Time: %.2f ms (%.2f FPS)\n", time / config.iterations, fps);
        printf("Pixels/sec: %.2f M\n", (config.texture_size * config.texture_size * fps) / 1e6f);
    }
    
    static void computeShaderKernel(RenderBenchmark* benchmark) {
        dim3 blockSize(256);
        dim3 gridSize((benchmark->config.num_particles + blockSize.x - 1) / blockSize.x);
        
        for (int i = 0; i < benchmark->config.iterations; i++) {
            computeShaderSimulation<<<gridSize, blockSize>>>(
                benchmark->d_compute_data, benchmark->config.num_particles, i * 0.016f);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    void benchmarkComputeShader() {
        printf("\n=== Compute Shader Benchmark ===\n");
        
        float time = timeKernelExecution(computeShaderKernel, this);
        float fps = (config.iterations * 1000.0f) / time;
        
        printf("Data points: %d\n", config.num_particles);
        printf("Time: %.2f ms (%.2f FPS)\n", time / config.iterations, fps);
        printf("Operations/sec: %.2f M\n", (config.num_particles * fps) / 1e6f);
    }
    
    void benchmarkDeferredShading() {
        printf("\n=== Deferred Shading Benchmark ===\n");
        
        dim3 blockSize(16, 16);
        dim3 gridSize((config.width + blockSize.x - 1) / blockSize.x,
                    (config.height + blockSize.y - 1) / blockSize.y);
        
        // G-Buffer + lighting is O(pixels × triangles + pixels × lights).
        // Cap at 5 iterations for reasonable runtime.
        int iters = std::max(5, config.iterations / 20);
        
        CUDA_CHECK(cudaEventRecord(start_event));
        
        for (int i = 0; i < iters; i++) {
            // G-Buffer pass
            generateGBuffer<<<gridSize, blockSize>>>(
                d_triangles, config.num_triangles, d_position_buffer, 
                d_normal_buffer, d_color_buffer, config.width, config.height);
            
            // Lighting pass
            deferredLighting<<<gridSize, blockSize>>>(
                d_position_buffer, d_normal_buffer, d_color_buffer,
                d_lights, num_lights, d_framebuffer, config.width, config.height);
            
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        float fps = (iters * 1000.0f) / milliseconds;
        
        printf("Lights: %d\n", num_lights);
        printf("Iterations: %d\n", iters);
        printf("G-Buffer + Lighting Time: %.2f ms (%.2f FPS)\n", milliseconds / iters, fps);
        printf("Light calculations/sec: %.2f M\n", (config.width * config.height * num_lights * fps) / 1e6f);
    }

    void benchmarkMatrixOperations() {
        printf("\n=== Matrix Multiplication Benchmark ===\n");
        
        dim3 blockSize(16, 16);
        dim3 gridSize((matrix_size + blockSize.x - 1) / blockSize.x,
                    (matrix_size + blockSize.y - 1) / blockSize.y);
        
        CUDA_CHECK(cudaEventRecord(start_event));
        
        for (int i = 0; i < config.iterations; i++) {
            matrixMultiply<<<gridSize, blockSize>>>(d_matrix_a, d_matrix_b, d_matrix_c, matrix_size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        float fps = (config.iterations * 1000.0f) / milliseconds;
        
        printf("Matrix size: %dx%d\n", matrix_size, matrix_size);
        printf("Time: %.2f ms (%.2f IPS)\n", milliseconds / config.iterations, fps);
        printf("GFLOPS: %.2f\n", (2.0f * matrix_size * matrix_size * matrix_size * fps) / 1e9f);
    }

    void benchmarkMemoryBandwidth() {
        printf("\n=== Memory Bandwidth Benchmark ===\n");
        
        int data_size = config.width * config.height;
        
        dim3 blockSize(256);
        dim3 gridSize((data_size + blockSize.x - 1) / blockSize.x);
        
        CUDA_CHECK(cudaEventRecord(start_event));
        
        for (int i = 0; i < config.iterations; i++) {
            memoryBandwidthTest<<<gridSize, blockSize>>>(d_framebuffer, d_framebuffer, data_size, 10);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        
        float bytes_transferred = data_size * sizeof(float) * 2.0f * config.iterations; // read + write
        float bandwidth_gb_s = (bytes_transferred / (milliseconds / 1000.0f)) / 1e9f;
        float fps = (config.iterations * 1000.0f) / milliseconds;
        
        printf("Data size: %.2f MB\n", (data_size * sizeof(float)) / 1e6f);
        printf("Time: %.2f ms (%.2f IPS)\n", milliseconds / config.iterations, fps);
        printf("Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    }

    void benchmarkTessellation() {
        printf("\n=== Tessellation Benchmark ===\n");
        
        dim3 blockSize(256);
        dim3 gridSize((config.num_triangles + blockSize.x - 1) / blockSize.x);
        
        CUDA_CHECK(cudaEventRecord(start_event));
        
        for (int i = 0; i < config.iterations / 10; i++) { // Fewer iterations due to complexity
            tessellateTriangles<<<gridSize, blockSize>>>(
                d_triangles, d_tessellated_triangles, config.num_triangles, 4);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        float fps = ((config.iterations / 10) * 1000.0f) / milliseconds;
        
        printf("Input triangles: %d\n", config.num_triangles);
        printf("Tessellation level: 4\n");
        printf("Time: %.2f ms (%.2f FPS)\n", milliseconds / (config.iterations / 10), fps);
        printf("Triangles/sec: %.2f M\n", (config.num_triangles * 16 * fps) / 1e6f);
    }

    void reduction() {
        printf("\n=== Reduction Benchmark ===\n");

        // prepare data
        thrust::device_vector<float> data(config.size, 1.0f);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < config.iterations; ++i) {
            volatile float sum = thrust::reduce(data.begin(), data.end());
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float timePerIter = ms / config.iterations;
        float throughput = (config.size * sizeof(float) * config.iterations / (ms/1000.0f)) / (1 << 30);

        printf("Data size: %d elements\n", config.size);
        printf("Time: %.2f ms per iter\n", timePerIter);
        printf("Throughput: %.2f GB/s\n", throughput);
    }

    void prefixSum() {
        printf("\n=== Prefix-Sum Benchmark ===\n");

        // prepare data
        thrust::device_vector<int> data(config.size);
        thrust::sequence(data.begin(), data.end());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < config.iterations; ++i) {
            thrust::exclusive_scan(data.begin(), data.end(), data.begin());
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float timePerIter = ms / config.iterations;
        float throughput = (config.size * sizeof(int) * config.iterations / (ms/1000.0f)) / (1 << 30);

        printf("Data size: %d elements\n", config.size);
        printf("Time: %.2f ms per iter\n", timePerIter);
        printf("Throughput: %.2f GB/s\n", throughput);
    }

    void sortBench() {
        printf("\n=== Sort Benchmark ===\n");

        // prepare data
        thrust::device_vector<unsigned int> data(config.size);
        thrust::sequence(data.begin(), data.end());
        thrust::reverse(data.begin(), data.end());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < config.iterations; ++i) {
            thrust::sort(data.begin(), data.end());
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float timePerIter = ms / config.iterations;
        float throughput = (config.size * sizeof(unsigned int) * config.iterations / (ms/1000.0f)) / (1 << 30);

        printf("Data size: %d elements\n", config.size);
        printf("Time: %.2f ms per iter\n", timePerIter);
        printf("Throughput: %.2f GB/s\n", throughput);
    }

    void sparseSpMVBenchmark() {
        printf("\n=== Sparse SpMV Benchmark ===\n");
        // 1) Build a simple tridiagonal matrix of size N×N
        const int N   = 1 << 20;              // 1 048 576 rows
        const int nnz = 3 * N - 2;            // 3 diagonals except ends

        std::vector<int>   hRowPtr(N + 1);
        std::vector<int>   hColInd(nnz);
        std::vector<float> hVals(nnz, 1.0f);
        std::vector<float> hX(N, 1.0f), hY(N, 0.0f);

        // Fill CSR for tridiagonal (−1, 0, +1 offsets)
        hRowPtr[0] = 0;
        int idx = 0;
        for (int i = 0; i < N; ++i) {
            if (i > 0) {
                hColInd[idx++] = i - 1;
            }
            hColInd[idx++] = i;
            if (i + 1 < N) {
                hColInd[idx++] = i + 1;
            }
            hRowPtr[i + 1] = idx;
        }

        // 2) Allocate & copy to device
        int   *dRowPtr, *dColInd;
        float *dVals, *dX, *dY;
        CUDA_CHECK(cudaMalloc(&dRowPtr, (N + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dColInd, nnz     * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dVals,   nnz     * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dX,      N       * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dY,      N       * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dRowPtr, hRowPtr.data(), (N + 1) * sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dColInd, hColInd.data(), nnz     * sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVals,   hVals.data(),   nnz     * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dX,      hX.data(),      N       * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dY,      hY.data(),      N       * sizeof(float), cudaMemcpyHostToDevice));

        // 3) Create cuSPARSE descriptors
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        const float alpha = 1.0f, beta = 0.0f;

        CUSPARSE_CHECK(cusparseCreateCsr(
            &matA, N, N, nnz,
            dRowPtr, dColInd, dVals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, N, dX, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, N, dY, CUDA_R_32F));

        // Ensure cuSPARSE handle is initialized
        initCuSparse();

        // 4) Allocate workspace
        size_t bufferSize = 0;
        void*  dBuffer    = nullptr;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
            &bufferSize));
        CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

        // time the SpMV
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));

        CUSPARSE_CHECK(cusparseSpMV(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
            dBuffer));

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float timePerIter = ms; // single iteration
        float throughput = (nnz * sizeof(float) / (ms/1000.0f)) / (1 << 30);

        printf("Matrix size: %dx%d\n", N, N);
        printf("Non-zeros: %d\n", nnz);
        printf("Time: %.2f ms per run\n", timePerIter);
        printf("Throughput: %.2f GB/s\n", throughput);

        // 6) Cleanup
        CUDA_CHECK(cudaFree(dBuffer));
        CUSPARSE_CHECK(cusparseDestroySpMat(matA));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));

        CUDA_CHECK(cudaFree(dRowPtr));
        CUDA_CHECK(cudaFree(dColInd));
        CUDA_CHECK(cudaFree(dVals));
        CUDA_CHECK(cudaFree(dX));
        CUDA_CHECK(cudaFree(dY));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Destroy cuSPARSE handle after use
        destroyCuSparse();
    }

    void fftMultiply() {
        printf("\n=== FFT-Multiply Benchmark ===\n");

        int n = static_cast<int>(config.size);
        cufftHandle plan;
        CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_C2C, 1));

        cufftComplex *dA, *dB;
        CUDA_CHECK(cudaMalloc(&dA, n * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&dB, n * sizeof(cufftComplex)));
        // TODO: initialise dA, dB with your data

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < config.iterations; ++it) {
            CUFFT_CHECK(cufftExecC2C(plan, dA, dA, CUFFT_FORWARD));
            CUFFT_CHECK(cufftExecC2C(plan, dB, dB, CUFFT_FORWARD));
            int threads = 256, blocks = (n + threads - 1) / threads;
            multiplyKernel<<<blocks, threads>>>(dA, dB, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUFFT_CHECK(cufftExecC2C(plan, dA, dA, CUFFT_INVERSE));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float timePerIter = ms / config.iterations;
        float throughput = (n * sizeof(cufftComplex) * config.iterations / (ms/1000.0f)) / (1 << 30);

        printf("Transform size: %d complex samples\n", n);
        printf("Time: %.2f ms per iter\n", timePerIter);
        printf("Throughput: %.2f GB/s\n", throughput);
        printf("\n");

        CUFFT_CHECK(cufftDestroy(plan));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
    }

    void benchmarkAtomicOperations() {
        printf("\n=== Atomic Operations Benchmark ===\n");
        
        dim3 blockSize(256);
        dim3 gridSize((config.atomic_test_size + blockSize.x - 1) / blockSize.x);
        
        // Initialize data
        CUDA_CHECK(cudaMemset(d_atomic_counters, 0, config.atomic_test_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_atomic_data, 0, config.atomic_test_size * sizeof(float)));
        
        // Atomic Add Test
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            atomicAddTest<<<gridSize, blockSize>>>(d_atomic_counters, d_atomic_data, 
                                                config.atomic_test_size, 10);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float timeAtomicAdd;
        CUDA_CHECK(cudaEventElapsedTime(&timeAtomicAdd, start_event, stop_event));
        printf("Atomic Add: %.2f ms/frame\n", timeAtomicAdd / config.iterations);
        
        // Atomic Min/Max Test
        int h_minmax[2] = {INT_MAX, INT_MIN};
        CUDA_CHECK(cudaMemcpy(d_atomic_counters, h_minmax, 2 * sizeof(int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            atomicMinMaxTest<<<gridSize, blockSize>>>(d_atomic_counters, config.atomic_test_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float timeAtomicMinMax;
        CUDA_CHECK(cudaEventElapsedTime(&timeAtomicMinMax, start_event, stop_event));
        printf("Atomic Min/Max: %.2f ms/frame\n", timeAtomicMinMax / config.iterations);
        
        // Atomic CAS Test
        CUDA_CHECK(cudaMemset(d_atomic_counters, 0, config.atomic_test_size * sizeof(int)));
        
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            atomicCASTest<<<gridSize, blockSize>>>(d_atomic_counters, config.atomic_test_size, 5);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float timeAtomicCAS;
        CUDA_CHECK(cudaEventElapsedTime(&timeAtomicCAS, start_event, stop_event));
        printf("Atomic CAS: %.2f ms/frame\n", timeAtomicCAS / config.iterations);
        printf("\n");
    }

    void benchmarkMemoryTypes() {
        printf("\n=== Memory Access Benchmark ===\n");
        
        dim3 blockSize(256);
        dim3 gridSize((config.memory_test_size + blockSize.x - 1) / blockSize.x);
        
        // Initialize test data
        float* h_data = new float[config.memory_test_size];
        for (int i = 0; i < config.memory_test_size; i++) {
            h_data[i] = (float)rand() / RAND_MAX;
        }
        CUDA_CHECK(cudaMemcpy(d_global_memory, h_data, config.memory_test_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_shared_test_input, h_data, config.memory_test_size * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_data;
        
        // Global Memory (Coalesced Access)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            globalMemoryTest<<<gridSize, blockSize>>>(d_global_memory, d_shared_test_output, 
                                                    config.memory_test_size, 1);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float timeGlobalCoalesced;
        CUDA_CHECK(cudaEventElapsedTime(&timeGlobalCoalesced, start_event, stop_event));
        printf("Global Memory (Coalesced): %.2f ms/frame\n", timeGlobalCoalesced / config.iterations);
        
        // Global Memory (Non-coalesced Access)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            globalMemoryTest<<<gridSize, blockSize>>>(d_global_memory, d_shared_test_output, 
                                                    config.memory_test_size, 32);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float timeGlobalNonCoalesced;
        CUDA_CHECK(cudaEventElapsedTime(&timeGlobalNonCoalesced, start_event, stop_event));
        printf("Global Memory (Non-coalesced): %.2f ms/frame\n", timeGlobalNonCoalesced / config.iterations);
        printf("Coalescing Penalty: %.2fx\n", timeGlobalNonCoalesced / timeGlobalCoalesced);
        
        // Shared Memory
        int shared_mem_size = blockSize.x * sizeof(float);
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            sharedMemoryBandwidthTest<<<gridSize, blockSize, shared_mem_size>>>(
                d_shared_test_input, d_shared_test_output, config.memory_test_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float timeShared;
        CUDA_CHECK(cudaEventElapsedTime(&timeShared, start_event, stop_event));
        printf("Shared Memory: %.2f ms/frame\n", timeShared / config.iterations);
        
        // Constant Memory
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            constantMemoryTest<<<gridSize, blockSize>>>(d_shared_test_output, config.memory_test_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float timeConstant;
        CUDA_CHECK(cudaEventElapsedTime(&timeConstant, start_event, stop_event));
        printf("Constant Memory: %.2f ms/frame\n", timeConstant / config.iterations);
        
        // Calculate bandwidth
        size_t bytes_transferred = (size_t)config.memory_test_size * sizeof(float) * 2; // read + write
        float bandwidth_coalesced = (bytes_transferred * config.iterations) / (timeGlobalCoalesced / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);
        printf("Global Memory Bandwidth (Coalesced): %.2f GB/s\n", bandwidth_coalesced);
        printf("\n");
    }
    
    void initializeNewBenchmarkData() {
        // Initialize instruction throughput data
        thrust::device_vector<int> int_init(config.instruction_test_size);
        thrust::sequence(int_init.begin(), int_init.end(), 1);
        CUDA_CHECK(cudaMemcpy(d_int_data, thrust::raw_pointer_cast(int_init.data()), 
                            config.instruction_test_size * sizeof(int), cudaMemcpyDeviceToDevice));
        
        thrust::device_vector<float> float_init(config.instruction_test_size);
        thrust::sequence(float_init.begin(), float_init.end(), 0.1f);
        
        // Convert to half precision
        thrust::device_vector<half> half_init(config.instruction_test_size);
        thrust::transform(float_init.begin(), float_init.end(), half_init.begin(),
                        [] __device__ (float f) { return __float2half(f); });
        CUDA_CHECK(cudaMemcpy(d_fp16_data, thrust::raw_pointer_cast(half_init.data()), 
                            config.instruction_test_size * sizeof(half), cudaMemcpyDeviceToDevice));
        
        CUDA_CHECK(cudaMemcpy(d_fp32_data, thrust::raw_pointer_cast(float_init.data()), 
                            config.instruction_test_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        thrust::device_vector<double> double_init(config.instruction_test_size);
        thrust::transform(float_init.begin(), float_init.end(), double_init.begin(),
                        [] __device__ (float f) { return (double)f; });
        CUDA_CHECK(cudaMemcpy(d_fp64_data, thrust::raw_pointer_cast(double_init.data()), 
                            config.instruction_test_size * sizeof(double), cudaMemcpyDeviceToDevice));
        
        // Initialize other test data
        CUDA_CHECK(cudaMemcpy(d_occupancy_input, thrust::raw_pointer_cast(float_init.data()), 
                            config.occupancy_test_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        CUDA_CHECK(cudaMemcpy(d_ilp_input, thrust::raw_pointer_cast(float_init.data()), 
                            config.ilp_test_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        thrust::device_vector<int> divergence_init(config.divergence_test_size);
        thrust::sequence(divergence_init.begin(), divergence_init.end(), 0);
        CUDA_CHECK(cudaMemcpy(d_divergence_input, thrust::raw_pointer_cast(divergence_init.data()), 
                            config.divergence_test_size * sizeof(int), cudaMemcpyDeviceToDevice));

        // Persistent threads memory
        int persistent_size = config.memory_test_size;
        CUDA_CHECK(cudaMalloc(&d_persistent_input, persistent_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_persistent_output, persistent_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_work_queue, persistent_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_work_count, sizeof(int)));
        
        // Texture cache memory and setup
        int tex_size = 1024;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        
        // Allocate texture arrays
        CUDA_CHECK(cudaMallocArray(&d_texture_array_coherent, &channelDesc, tex_size, tex_size));
        CUDA_CHECK(cudaMallocArray(&d_texture_array_random, &channelDesc, tex_size, tex_size));
        
        // Create and fill texture data
        float* h_texture_data = new float[tex_size * tex_size];
        for (int i = 0; i < tex_size * tex_size; i++) {
            h_texture_data[i] = sinf((float)i * 0.01f);
        }
        
        CUDA_CHECK(cudaMemcpyToArray(d_texture_array_coherent, 0, 0, h_texture_data, 
                                    tex_size * tex_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToArray(d_texture_array_random, 0, 0, h_texture_data, 
                                    tex_size * tex_size * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_texture_data;
        
        // Create texture objects
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = d_texture_array_coherent;
        
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;
        
        CUDA_CHECK(cudaCreateTextureObject(&tex_coherent, &resDesc, &texDesc, NULL));
        
        resDesc.res.array.array = d_texture_array_random;
        CUDA_CHECK(cudaCreateTextureObject(&tex_random, &resDesc, &texDesc, NULL));
        
        CUDA_CHECK(cudaMalloc(&d_texture_cache_output, tex_size * tex_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_texture_indices_coherent, tex_size * tex_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_texture_indices_random, tex_size * tex_size * sizeof(int)));
        
        // Initialize index arrays
        thrust::device_vector<int> indices(tex_size * tex_size);
        thrust::sequence(indices.begin(), indices.end());
        CUDA_CHECK(cudaMemcpy(d_texture_indices_coherent, thrust::raw_pointer_cast(indices.data()), 
                            tex_size * tex_size * sizeof(int), cudaMemcpyDeviceToDevice));
        
        // Shuffle for random access
        std::default_random_engine rng;
        std::vector<int> h_indices(tex_size * tex_size);
        CUDA_CHECK(cudaMemcpy(h_indices.data(), thrust::raw_pointer_cast(indices.data()), 
                             tex_size * tex_size * sizeof(int), cudaMemcpyDeviceToHost));
        std::shuffle(h_indices.begin(), h_indices.end(), rng);
        CUDA_CHECK(cudaMemcpy(d_texture_indices_random, h_indices.data(), 
                             tex_size * tex_size * sizeof(int), cudaMemcpyHostToDevice));
        
        // Register pressure test memory
        CUDA_CHECK(cudaMalloc(&d_register_input, config.memory_test_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_register_output, config.memory_test_size * sizeof(float)));
    }

    // Add these benchmark methods to the class
    void benchmarkInstructionThroughput() {
        printf("=== Instruction Throughput Benchmark ===\n");
        
        dim3 blockSize(256);
        dim3 gridSize((config.instruction_test_size + blockSize.x - 1) / blockSize.x);
        int iterations = 1000;
        
        // INT operations
        CUDA_CHECK(cudaEventRecord(start_event));
        instructionThroughputINT<<<gridSize, blockSize>>>(d_int_data, config.instruction_test_size, iterations);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float int_time;
        CUDA_CHECK(cudaEventElapsedTime(&int_time, start_event, stop_event));
        
        // FP16 operations
        CUDA_CHECK(cudaEventRecord(start_event));
        instructionThroughputFP16<<<gridSize, blockSize>>>(d_fp16_data, config.instruction_test_size, iterations);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float fp16_time;
        CUDA_CHECK(cudaEventElapsedTime(&fp16_time, start_event, stop_event));
        
        // FP32 operations
        CUDA_CHECK(cudaEventRecord(start_event));
        instructionThroughputFP32<<<gridSize, blockSize>>>(d_fp32_data, config.instruction_test_size, iterations);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float fp32_time;
        CUDA_CHECK(cudaEventElapsedTime(&fp32_time, start_event, stop_event));
        
        // FP64 operations
        CUDA_CHECK(cudaEventRecord(start_event));
        instructionThroughputFP64<<<gridSize, blockSize>>>(d_fp64_data, config.instruction_test_size, iterations);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float fp64_time;
        CUDA_CHECK(cudaEventElapsedTime(&fp64_time, start_event, stop_event));
        
        printf("INT Operations: %.2f GOPS\n", (config.instruction_test_size * iterations * 4.0f) / (int_time * 1e6f));
        printf("FP16 Operations: %.2f GOPS\n", (config.instruction_test_size * iterations * 2.0f) / (fp16_time * 1e6f));
        printf("FP32 Operations: %.2f GOPS\n", (config.instruction_test_size * iterations * 4.0f) / (fp32_time * 1e6f));
        printf("FP64 Operations: %.2f GOPS\n", (config.instruction_test_size * iterations * 4.0f) / (fp64_time * 1e6f));
        printf("\n");
    }

    void benchmarkOccupancyAndWarpDivergence() {
        printf("=== Occupancy and Warp Divergence Benchmark ===\n");
        
        dim3 blockSize(256);
        dim3 gridSize((config.occupancy_test_size + blockSize.x - 1) / blockSize.x);
        
        // High occupancy test
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 100; i++) {
            occupancyTestKernel<<<gridSize, blockSize>>>(d_occupancy_input, d_occupancy_output, config.occupancy_test_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float high_occ_time;
        CUDA_CHECK(cudaEventElapsedTime(&high_occ_time, start_event, stop_event));
        
        // Low occupancy test
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 100; i++) {
            occupancyTestKernelLowOccupancy<<<gridSize, blockSize>>>(d_occupancy_input, d_occupancy_output, config.occupancy_test_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float low_occ_time;
        CUDA_CHECK(cudaEventElapsedTime(&low_occ_time, start_event, stop_event));
        
        printf("High Occupancy: %.2f ms\n", high_occ_time / 100);
        printf("Low Occupancy: %.2f ms\n", low_occ_time / 100);
        printf("Occupancy Impact: %.2fx slower\n", low_occ_time / high_occ_time);
        printf("\n");
    }

    void benchmarkGPUIdleContextSwitchOverhead() {
        printf("=== GPU Idle / Context Switch Overhead Benchmark ===\n");
        
        dim3 blockSize(32);
        dim3 gridSize(1);
        
        // Measure context switch overhead with many small kernel launches
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.context_switch_iterations; i++) {
            contextSwitchKernel<<<gridSize, blockSize>>>(d_fp32_data, 32);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float context_switch_time;
        CUDA_CHECK(cudaEventElapsedTime(&context_switch_time, start_event, stop_event));
        
        // Measure single large kernel launch
        gridSize = dim3((config.context_switch_iterations * 32 + 31) / 32);
        CUDA_CHECK(cudaEventRecord(start_event));
        contextSwitchKernel<<<gridSize, blockSize>>>(d_fp32_data, config.context_switch_iterations * 32);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float single_launch_time;
        CUDA_CHECK(cudaEventElapsedTime(&single_launch_time, start_event, stop_event));
        
        printf("Multiple small launches: %.2f ms\n", context_switch_time);
        printf("Single large launch: %.2f ms\n", single_launch_time);
        printf("Context switch overhead per launch: %.4f ms\n", 
            (context_switch_time - single_launch_time) / config.context_switch_iterations);
        printf("\n");
    }

    void benchmarkILPAnalysis() {
        printf("=== Instruction-Level Parallelism (ILP) Analysis ===\n");
        
        dim3 blockSize(256);
        dim3 gridSize((config.ilp_test_size + blockSize.x - 1) / blockSize.x);
        
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 50; i++) {
            ilpTestKernel<<<gridSize, blockSize>>>(d_ilp_input, d_ilp_output, config.ilp_test_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float ilp_time;
        CUDA_CHECK(cudaEventElapsedTime(&ilp_time, start_event, stop_event));
        
        float gflops = (config.ilp_test_size * 8.0f * 50) / (ilp_time * 1e6f); // ~8 operations per thread
        printf("ILP Test Time: %.2f ms\n", ilp_time / 50);
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("Instructions per thread: 8 (4 independent chains)\n");
        printf("\n");
    }

    void benchmarkThreadDivergence() {
        printf("=== Thread Divergence in Control Flow ===\n");
        
        dim3 blockSize(256);
        dim3 gridSize((config.divergence_test_size + blockSize.x - 1) / blockSize.x);
        
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 20; i++) {
            divergenceTestKernel<<<gridSize, blockSize>>>(d_divergence_input, d_divergence_output, config.divergence_test_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float divergence_time;
        CUDA_CHECK(cudaEventElapsedTime(&divergence_time, start_event, stop_event));
        
        printf("Divergent Control Flow Time: %.2f ms\n", divergence_time / 20);
        printf("Divergence Pattern: 50%% warp divergence + nested branching\n");
        printf("Performance Impact: Significant due to serialized execution\n");
        printf("\n");
    }

    void benchmarkDynamicParallelism() {
        printf("=== Dynamic Parallelism Benchmark ===\n");
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
            printf("Dynamic Parallelism not available (requires Compute Capability >= 3.5)\n\n");
            return;
        }
        
        dim3 blockSize(256);
        dim3 gridSize((config.dp_test_size + blockSize.x - 1) / blockSize.x);
        
        // Test different recursion depths
        for (int depth = 0; depth <= 3; depth++) {
            CUDA_CHECK(cudaEventRecord(start_event));
            for (int i = 0; i < 20; i++) {
                dynamicParallelismParent<<<gridSize, blockSize>>>(d_dp_data, config.dp_test_size, depth);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            float dp_time;
            CUDA_CHECK(cudaEventElapsedTime(&dp_time, start_event, stop_event));
            
            float bandwidth = (config.dp_test_size * sizeof(float) * 20) / (dp_time * 1e6f);
            printf("Depth %d Time: %.2f ms\n", depth, dp_time / 20);
            printf("Bandwidth: %.2f MB/s\n", bandwidth * 1024);
            printf("Kernel launches per iteration: %d\n", (int)pow(2, depth + 1) - 1);
            printf("\n");
        }
    }

    void benchmarkFlatKernelEquivalent() {
        printf("=== Flat Kernel Comparison (Depth 2 Equivalent) ===\n");

        dim3 blockSize(256);
        dim3 gridSize((config.dp_test_size + blockSize.x - 1) / blockSize.x);

        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 20; i++) {
            flatKernelEquivalent<<<gridSize, blockSize>>>(d_dp_data, config.dp_test_size, 2);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float flat_time;
        CUDA_CHECK(cudaEventElapsedTime(&flat_time, start_event, stop_event));
        
        float flat_bandwidth = (config.dp_test_size * sizeof(float) * 20) / (flat_time * 1e6f);
        printf("Flat Kernel Time: %.2f ms\n", flat_time / 20);
        printf("Bandwidth: %.2f MB/s\n", flat_bandwidth * 1024);
        printf("Kernel launches per iteration: 1\n");
        printf("\n");
    }

    void benchmarkPersistentThreads() {
        printf("=== Persistent Threads Benchmark ===\n");
        
        int work_items = config.memory_test_size;
        int work_per_thread = 100;
        int block_size = 256;
        int num_blocks = (work_items + block_size - 1) / block_size;
        
        // Initialize input data
        thrust::device_vector<float> input_vec(work_items);
        thrust::sequence(input_vec.begin(), input_vec.end(), 1.0f);
        CUDA_CHECK(cudaMemcpy(d_persistent_input, thrust::raw_pointer_cast(input_vec.data()), 
                            work_items * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Test traditional approach - many kernel launches
        CUDA_CHECK(cudaEventRecord(start_event));
        
        int launches_per_iteration = 10;
        int work_per_launch = work_items / launches_per_iteration;
        
        for (int iter = 0; iter < config.iterations; iter++) {
            for (int launch = 0; launch < launches_per_iteration; launch++) {
                int offset = launch * work_per_launch;
                int remaining = std::min(work_per_launch, work_items - offset);
                
                dim3 grid_size((remaining + block_size - 1) / block_size);
                traditional_work_kernel<<<grid_size, block_size>>>(
                    d_persistent_input + offset, d_persistent_output + offset, 
                    remaining, work_per_thread);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float traditional_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&traditional_time, start_event, stop_event));
        
        // Test persistent threads approach
        CUDA_CHECK(cudaEventRecord(start_event));
        
        for (int iter = 0; iter < config.iterations; iter++) {
            int work_count_init = 0;
            CUDA_CHECK(cudaMemcpy(d_work_count, &work_count_init, sizeof(int), cudaMemcpyHostToDevice));
            
            // Launch fewer blocks than work items - threads will persist and grab more work
            dim3 persistent_grid(num_blocks / 4);  // Use fewer blocks
            persistent_work_kernel<<<persistent_grid, block_size>>>(
                d_persistent_input, d_persistent_output, d_work_queue, 
                d_work_count, work_items, work_per_thread);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float persistent_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&persistent_time, start_event, stop_event));
        
        printf("Work Items: %d\n", work_items);
        printf("Traditional Time: %.2f ms\n", traditional_time / config.iterations);
        printf("Persistent Time: %.2f ms\n", persistent_time / config.iterations);
        printf("Speedup: %.2fx\n", traditional_time / persistent_time);
        printf("Launch Overhead Reduction: %.2f%%\n", 
            100.0f * (1.0f - persistent_time / traditional_time));
    }

    void benchmarkTextureCacheEfficiency() {
        printf("=== Texture Cache Efficiency Benchmark ===\n");
        
        int tex_size = 1024;
        int total_pixels = tex_size * tex_size;
        dim3 block_size(256);
        dim3 grid_size((total_pixels + block_size.x - 1) / block_size.x);
        
        // Test coherent texture access (cache-friendly)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            texture_coherent_access<<<grid_size, block_size>>>(
                tex_coherent, d_texture_cache_output, tex_size, tex_size, d_texture_indices_coherent);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float coherent_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&coherent_time, start_event, stop_event));
        
        // Test random texture access (cache-unfriendly)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            texture_random_access<<<grid_size, block_size>>>(
                tex_random, d_texture_cache_output, tex_size, tex_size, d_texture_indices_random);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float random_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&random_time, start_event, stop_event));
        
        float cache_efficiency = (random_time - coherent_time) / random_time * 100.0f;
        
        printf("Texture Size: %dx%d\n", tex_size, tex_size);
        printf("Coherent Access: %.2f ms\n", coherent_time / config.iterations);
        printf("Random Access: %.2f ms\n", random_time / config.iterations);
        printf("Cache Efficiency: %.2f%%\n", cache_efficiency);
        printf("Random/Coherent Ratio: %.2fx\n", random_time / coherent_time);
        
        float bandwidth_coherent = (total_pixels * 9 * sizeof(float) * config.iterations) / 
                                (coherent_time / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);
        float bandwidth_random = (total_pixels * 9 * sizeof(float) * config.iterations) / 
                                (random_time / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);
        
        printf("Effective Bandwidth (Coherent): %.2f GB/s\n", bandwidth_coherent);
        printf("Effective Bandwidth (Random): %.2f GB/s\n", bandwidth_random);
    }

    void benchmarkRegisterPressureSpilling() {
        printf("=== Register Pressure / Spilling Benchmark ===\n");
        
        int n = config.memory_test_size;
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Initialize input data
        thrust::device_vector<float> input_vec(n);
        thrust::sequence(input_vec.begin(), input_vec.end(), 1.0f);
        CUDA_CHECK(cudaMemcpy(d_register_input, thrust::raw_pointer_cast(input_vec.data()), 
                            n * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Test low register pressure kernel
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            low_register_pressure_kernel<<<grid_size, block_size>>>(
                d_register_input, d_register_output, n);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float low_register_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&low_register_time, start_event, stop_event));
        
        // Test high register pressure kernel (likely to spill)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.iterations; i++) {
            high_register_pressure_kernel<<<grid_size, block_size>>>(
                d_register_input, d_register_output, n);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float high_register_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&high_register_time, start_event, stop_event));
        
        // Calculate performance impact
        float spill_penalty = (high_register_time - low_register_time) / low_register_time * 100.0f;
        
        printf("Elements: %d\n", n);
        printf("Low Register Pressure: %.2f ms\n", low_register_time / config.iterations);
        printf("High Register Pressure: %.2f ms\n", high_register_time / config.iterations);
        printf("Performance Penalty: %.2f%%\n", spill_penalty);
        printf("Slowdown Factor: %.2fx\n", high_register_time / low_register_time);
        
        // Get occupancy information
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        int low_min_grid_size, low_block_size;
        cudaOccupancyMaxPotentialBlockSize(&low_min_grid_size, &low_block_size, 
                                        low_register_pressure_kernel, 0, 0);
        
        int high_min_grid_size, high_block_size;
        cudaOccupancyMaxPotentialBlockSize(&high_min_grid_size, &high_block_size, 
                                        high_register_pressure_kernel, 0, 0);
        
        printf("Low Reg Optimal Block Size: %d\n", low_block_size);
        printf("High Reg Optimal Block Size: %d\n", high_block_size);
        
        // Calculate theoretical occupancy
        int low_max_blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&low_max_blocks_per_sm, 
                                                    low_register_pressure_kernel, 256, 0);
        
        int high_max_blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&high_max_blocks_per_sm, 
                                                    high_register_pressure_kernel, 256, 0);
        
        float low_occupancy = (float)low_max_blocks_per_sm * 256 / prop.maxThreadsPerMultiProcessor;
        float high_occupancy = (float)high_max_blocks_per_sm * 256 / prop.maxThreadsPerMultiProcessor;
        
        printf("Low Reg Occupancy: %.1f%%\n", low_occupancy * 100.0f);
        printf("High Reg Occupancy: %.1f%%\n", high_occupancy * 100.0f);
        printf("Occupancy Drop: %.1f%%\n", (low_occupancy - high_occupancy) * 100.0f);
    }

    void benchmarkMemoryLatency() {
        printf("\n=== Memory Latency Benchmark ===\n");
        
        // Initialize test data
        std::vector<float> h_data(config.memory_latency_test_size, 1.0f);
        std::vector<int> h_indices(config.memory_latency_test_size);
        for (int i = 0; i < config.memory_latency_test_size; i++) {
            h_indices[i] = (i + 1) % config.memory_latency_test_size;
        }
        
        CUDA_CHECK(cudaMemcpy(d_latency_test_data, h_data.data(), 
                            config.memory_latency_test_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_latency_indices, h_indices.data(), 
                            config.memory_latency_test_size * sizeof(int), cudaMemcpyHostToDevice));
        
        dim3 blockSize(256);
        dim3 gridSize((config.memory_latency_test_size + blockSize.x - 1) / blockSize.x);
        
        // L1 Cache Latency Test
        CUDA_CHECK(cudaEventRecord(start_event));
        l1CacheLatencyKernel<<<gridSize, blockSize>>>(d_latency_test_data, d_latency_indices, 
                                                    d_latency_output, config.memory_latency_test_size, 100);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float l1_time;
        CUDA_CHECK(cudaEventElapsedTime(&l1_time, start_event, stop_event));
        
        // L2 Cache Latency Test (large stride)
        CUDA_CHECK(cudaEventRecord(start_event));
        l2CacheLatencyKernel<<<gridSize, blockSize>>>(d_latency_test_data, d_latency_output, 
                                                    config.memory_latency_test_size * 10, 4096);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float l2_time;
        CUDA_CHECK(cudaEventElapsedTime(&l2_time, start_event, stop_event));
        
        // Shared Memory Latency Test
        CUDA_CHECK(cudaEventRecord(start_event));
        sharedMemoryLatencyKernel<<<gridSize, blockSize>>>(d_latency_output, 1000);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float shared_time;
        CUDA_CHECK(cudaEventElapsedTime(&shared_time, start_event, stop_event));
        
        printf("L1 Cache Latency: %.2f cycles (est.)\n", l1_time * 1000.0f);
        printf("L2 Cache Latency: %.2f cycles (est.)\n", l2_time * 1000.0f);
        printf("Shared Memory Latency: %.2f cycles (est.)\n", shared_time * 100.0f);
    }

    void benchmarkPCIeBandwidth() {
        printf("\n=== PCIe Bandwidth Benchmark ===\n");

        size_t bytes = config.pcie_transfer_size * sizeof(float);

        // Initialize data
        for (int i = 0; i < config.pcie_transfer_size; i++) {
            h_pcie_pinned[i] = i * 1.0f;
            h_pcie_pageable[i] = i * 1.0f;
        }

        // Improved methodology: 5 warm-up + 20 timed transfers, then
        // report min / median / max so the user can spot outliers caused
        // by OS interference, PCIe link training, or driver hiccups.
        const int warmup_iters = 5;
        const int timed_iters = 20;

        // Helper that runs warm-up + timed transfers and returns per-iter
        // millisecond timings (vector of timed_iters samples).
        auto run_pcie_test = [&](cudaMemcpyKind kind, void* host_ptr) {
            // Warm up
            for (int i = 0; i < warmup_iters; i++) {
                if (kind == cudaMemcpyHostToDevice)
                    cudaMemcpy(d_pcie_data, host_ptr, bytes, kind);
                else
                    cudaMemcpy(host_ptr, d_pcie_data, bytes, kind);
            }
            cudaDeviceSynchronize();

            // Timed runs - record per-iter ms via CUDA events
            std::vector<float> per_iter_ms;
            per_iter_ms.reserve(timed_iters);
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            for (int i = 0; i < timed_iters; i++) {
                cudaEventRecord(a);
                if (kind == cudaMemcpyHostToDevice)
                    cudaMemcpy(d_pcie_data, host_ptr, bytes, kind);
                else
                    cudaMemcpy(host_ptr, d_pcie_data, bytes, kind);
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                per_iter_ms.push_back(ms);
            }
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return per_iter_ms;
        };

        // Compute min/median/max bandwidth from a vector of per-iter ms.
        auto summarise = [&](const std::vector<float>& per_iter_ms, double bytes_this) {
            std::vector<float> sorted = per_iter_ms;
            std::sort(sorted.begin(), sorted.end());
            float min_ms = sorted.front();
            float max_ms = sorted.back();
            float med_ms = sorted[sorted.size() / 2];
            struct BW { double min, med, max; };
            return BW{
                bytes_this / (min_ms * 1e6),
                bytes_this / (med_ms * 1e6),
                bytes_this / (max_ms * 1e6)
            };
        };

        auto pinned_h2d   = summarise(run_pcie_test(cudaMemcpyHostToDevice, h_pcie_pinned),   bytes);
        auto pageable_h2d = summarise(run_pcie_test(cudaMemcpyHostToDevice, h_pcie_pageable), bytes);
        auto pinned_d2h   = summarise(run_pcie_test(cudaMemcpyDeviceToHost, h_pcie_pinned),   bytes);
        auto pageable_d2h = summarise(run_pcie_test(cudaMemcpyDeviceToHost, h_pcie_pageable), bytes);

        printf("Transfer size: %.2f MB, warmup %d + timed %d transfers\n",
               bytes / (1024.0 * 1024.0), warmup_iters, timed_iters);
        printf("(Reporting min / median / max GB/s across timed transfers)\n\n");
        printf("Host-to-Device:\n");
        printf("  Pinned:  %.2f / %.2f / %.2f GB/s\n",
               pinned_h2d.min, pinned_h2d.med, pinned_h2d.max);
        printf("  Pageable:%.2f / %.2f / %.2f GB/s\n",
               pageable_h2d.min, pageable_h2d.med, pageable_h2d.max);
        printf("Device-to-Host:\n");
        printf("  Pinned:  %.2f / %.2f / %.2f GB/s\n",
               pinned_d2h.min, pinned_d2h.med, pinned_d2h.max);
        printf("  Pageable:%.2f / %.2f / %.2f GB/s\n",
               pageable_d2h.min, pageable_d2h.med, pageable_d2h.max);
        printf("Pinned vs Pageable speedup (median):\n");
        printf("  H2D: %.2fx\n", pinned_h2d.med / pageable_h2d.med);
        printf("  D2H: %.2fx\n", pinned_d2h.med / pageable_d2h.med);
    }

    void benchmarkP2PCommunication() {
        printf("\n=== P2P GPU Communication ===\n");
        
        if (num_gpus < 2) {
            printf("Only 1 GPU detected - P2P test skipped\n");
            return;
        }
        
        printf("GPUs detected: %d\n", num_gpus);
        
        // Check P2P access capability
        int can_access_peer;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, 0, 1));
        
        if (!can_access_peer) {
            printf("P2P access not supported between GPU 0 and 1\n");
            return;
        }

        // Detect NVLink between GPU 0 and 1 via NVML (best-effort).
        int nvlink_ver = detectNvLinkBetween(0, 1);
        if (nvlink_ver > 0) {
            printf("NVLink detected between GPU 0 and 1 (v%d)\n", nvlink_ver);
        } else {
            printf("NVLink not detected by NVML (will infer link type from bandwidth).\n");
        }

        // CUDA P2P performance rank. The canonical API name is
        // cudaDeviceGetP2PAttribute (introduced in CUDA 8.0).  The historical
        // alias cudaGetDeviceP2PAttribute (without "Device") was deprecated
        // for years and finally removed in CUDA 13.0, so we always use the
        // canonical name here and gate on the CUDA 8.0 introduction version.
        {
            int perf_rank = 0;
            cudaError_t rank_err = cudaErrorUnknown;
#if CUDART_VERSION >= 8000
            rank_err = cudaDeviceGetP2PAttribute(
                &perf_rank, cudaDevP2PAttrPerformanceRank, 0, 1);
#endif
            if (rank_err == cudaSuccess) {
                printf("P2P performance rank (0 -> 1): %d\n", perf_rank);
            }
            cudaGetLastError(); // clear
        }

        // Enable P2P access
        cudaSetDevice(0);
        CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
        cudaSetDevice(1);
        CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
        
        size_t bytes = config.p2p_transfer_size * sizeof(float);
        
        // Initialize source data on GPU 0
        cudaSetDevice(0);
        thrust::device_vector<float> init_data(config.p2p_transfer_size);
        thrust::sequence(init_data.begin(), init_data.end(), 1.0f);
        CUDA_CHECK(cudaMemcpy(d_p2p_src, thrust::raw_pointer_cast(init_data.data()), 
                            bytes, cudaMemcpyDeviceToDevice));

        // Warm up (5 transfers, untimed)
        for (int i = 0; i < 5; i++) {
            cudaMemcpyPeer(d_p2p_dst, 1, d_p2p_src, 0, bytes);
        }
        cudaDeviceSynchronize();
        
        // Test P2P bandwidth
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 20; i++) {
            CUDA_CHECK(cudaMemcpyPeer(d_p2p_dst, 1, d_p2p_src, 0, bytes));
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float p2p_time;
        CUDA_CHECK(cudaEventElapsedTime(&p2p_time, start_event, stop_event));
        
        float p2p_bandwidth = (20.0f * bytes) / (p2p_time * 1e6f);
        
        printf("P2P Bandwidth (GPU 0->1): %.2f GB/s\n", p2p_bandwidth);
        printf("Transfer Size: %.2f MB\n", bytes / (1024.0f * 1024.0f));
        // Infer link type from bandwidth
        const char* link_type =
            (nvlink_ver > 0) ? "NVLink (NVML-confirmed)" :
            (p2p_bandwidth > 50.0f) ? "NVLink (inferred from bandwidth)" :
            (p2p_bandwidth > 25.0f) ? "PCIe 4.0/5.0 (inferred)" :
            "PCIe 3.0 (inferred)";
        printf("Link type: %s\n", link_type);
        printf("(See NVLink_Bandwidth benchmark for full multi-pair topology scan.)\n");
        
        // Test bidirectional P2P
        cudaSetDevice(1);
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 10; i++) {
            CUDA_CHECK(cudaMemcpyPeer(d_p2p_src, 0, d_p2p_dst, 1, bytes));
            CUDA_CHECK(cudaMemcpyPeer(d_p2p_dst, 1, d_p2p_src, 0, bytes));
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float bidirectional_time;
        CUDA_CHECK(cudaEventElapsedTime(&bidirectional_time, start_event, stop_event));
        
        float bidirectional_bandwidth = (20.0f * bytes) / (bidirectional_time * 1e6f);
        
        printf("Bidirectional P2P: %.2f GB/s\n", bidirectional_bandwidth);
        
        cudaSetDevice(0);
    }

    void benchmarkAsynchronousExecution() {
        printf("\n=== Asynchronous Execution & Overlap Test ===\n");
        
        // Initialize data
        std::vector<float> h_input(config.async_test_size);
        for (int i = 0; i < config.async_test_size; i++) {
            h_input[i] = (float)rand() / RAND_MAX;
        }
        
        for (int i = 0; i < config.async_streams; i++) {
            CUDA_CHECK(cudaMemcpy(d_async_inputs[i], h_input.data(), 
                                config.async_test_size * sizeof(float), cudaMemcpyHostToDevice));
        }
        
        dim3 blockSize(256);
        dim3 gridSize((config.async_test_size + blockSize.x - 1) / blockSize.x);
        
        // Test synchronous execution
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.async_streams; i++) {
            asyncComputeKernel<<<gridSize, blockSize>>>(
                d_async_inputs[i], d_async_outputs[i], config.async_test_size, 100);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float sync_time;
        CUDA_CHECK(cudaEventElapsedTime(&sync_time, start_event, stop_event));
        
        // Test asynchronous execution with compute overlap
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.async_streams; i++) {
            CUDA_CHECK(cudaEventRecord(async_events[i * 2], async_streams[i]));
            asyncComputeKernel<<<gridSize, blockSize, 0, async_streams[i]>>>(
                d_async_inputs[i], d_async_outputs[i], config.async_test_size, 100);
            CUDA_CHECK(cudaEventRecord(async_events[i * 2 + 1], async_streams[i]));
        }
        for (int i = 0; i < config.async_streams; i++) {
            CUDA_CHECK(cudaStreamSynchronize(async_streams[i]));
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float async_time;
        CUDA_CHECK(cudaEventElapsedTime(&async_time, start_event, stop_event));
        
        // Test memory + compute overlap
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < config.async_streams; i++) {
            // Launch compute and memory kernels on different streams
            asyncComputeKernel<<<gridSize, blockSize, 0, async_streams[i]>>>(
                d_async_inputs[i], d_async_outputs[i], config.async_test_size, 50);
            
            if (i + 1 < config.async_streams) {
                asyncMemoryKernel<<<gridSize, blockSize, 0, async_streams[(i + 1) % config.async_streams]>>>(
                    d_async_inputs[(i + 1) % config.async_streams], 
                    d_async_outputs[(i + 1) % config.async_streams], config.async_test_size);
            }
        }
        for (int i = 0; i < config.async_streams; i++) {
            CUDA_CHECK(cudaStreamSynchronize(async_streams[i]));
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float overlap_time;
        CUDA_CHECK(cudaEventElapsedTime(&overlap_time, start_event, stop_event));
        
        printf("Streams: %d\n", config.async_streams);
        printf("Data size: %d elements\n", config.async_test_size);
        printf("Synchronous time: %.2f ms\n", sync_time);
        printf("Asynchronous time: %.2f ms\n", async_time);
        printf("Overlap time: %.2f ms\n", overlap_time);
        printf("Async speedup: %.2fx\n", sync_time / async_time);
        printf("Overlap efficiency: %.1f%%\n", (sync_time - overlap_time) / sync_time * 100);
    }
    
    void benchmarkInstructionMix() {
        printf("=== Instruction Mix Test ===\n");
        
        // Initialize data
        std::vector<float> h_float_input(config.instruction_mix_size);
        std::vector<int> h_int_input(config.instruction_mix_size);
        
        for (int i = 0; i < config.instruction_mix_size; i++) {
            h_float_input[i] = (float)rand() / RAND_MAX;
            h_int_input[i] = rand();
        }
        
        CUDA_CHECK(cudaMemcpy(d_instmix_input, h_float_input.data(), 
                            config.instruction_mix_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_instmix_control_input, h_int_input.data(), 
                            config.instruction_mix_size * sizeof(int), cudaMemcpyHostToDevice));
        
        dim3 blockSize(256);
        dim3 gridSize((config.instruction_mix_size + blockSize.x - 1) / blockSize.x);
        
        // Test ALU-heavy workload
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 10; i++) {
            instructionMixALUHeavy<<<gridSize, blockSize>>>(
                d_instmix_input, d_instmix_output, config.instruction_mix_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float alu_time;
        CUDA_CHECK(cudaEventElapsedTime(&alu_time, start_event, stop_event));
        
        // Test memory-heavy workload
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 10; i++) {
            instructionMixMemoryHeavy<<<gridSize, blockSize>>>(
                d_instmix_input, d_instmix_output, config.instruction_mix_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float memory_time;
        CUDA_CHECK(cudaEventElapsedTime(&memory_time, start_event, stop_event));
        
        // Test control-heavy workload
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 10; i++) {
            instructionMixControlHeavy<<<gridSize, blockSize>>>(
                d_instmix_control_input, d_instmix_output, config.instruction_mix_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float control_time;
        CUDA_CHECK(cudaEventElapsedTime(&control_time, start_event, stop_event));
        
        printf("Data size: %d elements (10 iterations each)\n", config.instruction_mix_size);
        printf("ALU-heavy workload: %.2f ms (%.2f GFLOPS)\n", 
               alu_time / 10, (config.instruction_mix_size * 16 * 10) / (alu_time * 1e6));
        printf("Memory-heavy workload: %.2f ms (%.2f GB/s)\n", 
               memory_time / 10, (config.instruction_mix_size * 8 * 4 * 10) / (memory_time * 1e6));
        printf("Control-heavy workload: %.2f ms\n", control_time / 10);
        printf("ALU vs Memory ratio: %.2f:1\n", memory_time / alu_time);
        printf("Control flow overhead: %.1f%%\n", (control_time - alu_time) / alu_time * 100);
    }
    
    void benchmarkCacheThrashingAndConflicts() {
        printf("\n=== Cache Thrashing and Conflict Test ===\n");
        
        // Initialize data
        std::vector<float> h_data(config.cache_thrash_size);
        std::vector<int> h_indices(config.cache_thrash_size * 4);
        
        // Sequential pattern
        for (int i = 0; i < config.cache_thrash_size; i++) {
            h_data[i] = (float)i;
        }
        
        // Random access pattern for thrashing
        std::random_device rd;
        std::mt19937 gen(rd());
        for (int i = 0; i < config.cache_thrash_size * 4; i++) {
            h_indices[i] = gen() % config.cache_thrash_size;
        }
        
        CUDA_CHECK(cudaMemcpy(d_cache_data, h_data.data(), 
                            config.cache_thrash_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cache_indices, h_indices.data(), 
                            config.cache_thrash_size * 4 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cache_conflict_data, h_data.data(), 
                            config.cache_thrash_size * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 blockSize(256);
        dim3 gridSize_thrash((config.cache_thrash_size + blockSize.x - 1) / blockSize.x);
        
        // Test sequential access (cache-friendly)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 50; i++) {
            cacheConflictKernel<<<gridSize_thrash, blockSize>>>(
                d_cache_conflict_data, config.cache_thrash_size, 1);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float sequential_time;
        CUDA_CHECK(cudaEventElapsedTime(&sequential_time, start_event, stop_event));
        
        // Test cache thrashing (random access)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 50; i++) {
            cacheThrashingKernel<<<gridSize_thrash, blockSize>>>(
                d_cache_data, d_cache_indices, config.cache_thrash_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float thrashing_time;
        CUDA_CHECK(cudaEventElapsedTime(&thrashing_time, start_event, stop_event));
        
        // Test different stride patterns for cache conflicts
        int strides[] = {1, 32, 64, 128, 256, 512, 1024};
        printf("Data size: %d elements (50 iterations each)\n", config.cache_thrash_size);
        printf("Sequential access: %.2f ms\n", sequential_time / 50);
        printf("Random thrashing: %.2f ms\n", thrashing_time / 50);
        printf("Thrashing penalty: %.2fx slower\n", thrashing_time / sequential_time);
        
        printf("\nCache Conflict Analysis (stride patterns):\n");
        for (int stride : strides) {
            dim3 gridSize_conflict((config.cache_thrash_size / stride + blockSize.x - 1) / blockSize.x);
            
            CUDA_CHECK(cudaEventRecord(start_event));
            for (int i = 0; i < 20; i++) {
                cacheConflictKernel<<<gridSize_conflict, blockSize>>>(
                    d_cache_conflict_data, config.cache_thrash_size, stride);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float conflict_time;
            CUDA_CHECK(cudaEventElapsedTime(&conflict_time, start_event, stop_event));
            
            printf("  Stride %4d: %.2f ms (%.2fx vs sequential)\n", 
                   stride, conflict_time / 20, (conflict_time / 20) / (sequential_time / 50));
        }
        
        float cache_efficiency = (sequential_time / thrashing_time) * 100;
        printf("Cache efficiency: %.1f%%\n", cache_efficiency);
    }

    void benchmarkBankConflictTesting() {
        printf("\n=== Bank Conflict Testing ===\n");
        
        // Initialize test data
        std::vector<float> h_input(bank_conflict_test_size);
        for (int i = 0; i < bank_conflict_test_size; i++) {
            h_input[i] = (float)rand() / RAND_MAX;
        }
        CUDA_CHECK(cudaMemcpy(d_bank_conflict_input, h_input.data(), 
                            bank_conflict_test_size * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 blockSize(256);
        dim3 gridSize((bank_conflict_test_size + blockSize.x - 1) / blockSize.x);
        size_t shared_mem_size = blockSize.x * sizeof(float);
        
        // Test different stride patterns
        int test_strides[] = {1, 2, 4, 8, 16, 32};
        int num_stride_tests = sizeof(test_strides) / sizeof(test_strides[0]);
        
        printf("Testing bank conflict patterns:\n");
        printf("Stride Time(ms) Bandwidth(GB/s) Conflicts\n");
        printf("------ -------- --------------- ---------\n");
        
        for (int i = 0; i < num_stride_tests; i++) {
            int stride = test_strides[i];
            
            // Warm up
            bankConflictKernel<<<gridSize, blockSize, shared_mem_size>>>(
                d_bank_conflict_input, d_bank_conflict_output, bank_conflict_test_size, stride);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Time the kernel
            CUDA_CHECK(cudaEventRecord(start_event));
            for (int iter = 0; iter < 100; iter++) {
                bankConflictKernel<<<gridSize, blockSize, shared_mem_size>>>(
                    d_bank_conflict_input, d_bank_conflict_output, bank_conflict_test_size, stride);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float milliseconds;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
            milliseconds /= 100.0f; // Average per iteration
            
            // Calculate bandwidth
            float bytes_transferred = bank_conflict_test_size * sizeof(float) * 2; // Read + Write
            float bandwidth_gb_s = (bytes_transferred / (milliseconds / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);
            
            // Estimate conflicts (stride > 1 causes conflicts)
            const char* conflict_level;
            if (stride == 1) conflict_level = "None";
            else if (stride <= 2) conflict_level = "Low";
            else if (stride <= 8) conflict_level = "Medium";
            else conflict_level = "High";
            
            printf("%6d %8.3f %15.2f %s\n", stride, milliseconds, bandwidth_gb_s, conflict_level);
        }
    }

    void benchmarkPersistentOccupancyVsContextSwitching() {
        printf("\n=== Persistent Occupancy vs Context Switching ===\n");
        
        // Initialize test data
        std::vector<float> h_input(persistent_occ_test_size);
        for (int i = 0; i < persistent_occ_test_size; i++) {
            h_input[i] = (float)rand() / RAND_MAX;
        }
        CUDA_CHECK(cudaMemcpy(d_persistent_occ_input, h_input.data(), 
                            persistent_occ_test_size * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 blockSize(256);
        dim3 gridSize(32); // One block per SM
        
        printf("Testing persistent threads vs context switching:\n");
        printf("Test Type                     Time(ms) Throughput(GFLOPS)\n");
        printf("----------------------------- -------- ------------------\n");
        
        // Test 1: Traditional context switching (many small kernel launches)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int iter = 0; iter < 1000; iter++) {
            // Small kernel launches that cause context switching
            persistentOccupancyKernel<<<gridSize, blockSize>>>(
                d_persistent_occ_input, d_persistent_occ_output, d_persistent_work_flags, 
                persistent_occ_test_size, 1);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float context_switch_time;
        CUDA_CHECK(cudaEventElapsedTime(&context_switch_time, start_event, stop_event));
        
        // Calculate FLOPS (roughly 100 operations per element per iteration)
        float ops_per_element = 100.0f * 1000.0f; // 100 ops * 1000 iterations
        float total_ops = persistent_occ_test_size * ops_per_element;
        float context_switch_gflops = (total_ops / (context_switch_time / 1000.0f)) / 1e9f;
        
        printf("Context Switching (1000x1)    %8.3f %18.2f\n", 
            context_switch_time, context_switch_gflops);
        
        // Test 2: Persistent occupancy (one long-running kernel)
        // Initialize work flags
        std::vector<int> h_work_flags(32, 1000); // 1000 work units per SM
        CUDA_CHECK(cudaMemcpy(d_persistent_work_flags, h_work_flags.data(), 
                            32 * sizeof(int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start_event));
        persistentOccupancyKernel<<<gridSize, blockSize>>>(
            d_persistent_occ_input, d_persistent_occ_output, d_persistent_work_flags, 
            persistent_occ_test_size, 1000);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float persistent_time;
        CUDA_CHECK(cudaEventElapsedTime(&persistent_time, start_event, stop_event));
        float persistent_gflops = (total_ops / (persistent_time / 1000.0f)) / 1e9f;
        
        printf("Persistent Occupancy (1x1000) %8.3f %18.2f\n", 
            persistent_time, persistent_gflops);
        
        // Calculate improvement
        float speedup = context_switch_time / persistent_time;
        printf("\nPersistent occupancy speedup: %.2fx\n", speedup);
        printf("Context switching overhead: %.2f%%\n", 
            ((context_switch_time - persistent_time) / context_switch_time) * 100.0f);
    }

    void benchmarkCUDAGraphExecutionOverhead() {
        printf("\n=== CUDA Graph Execution Overhead ===\n");
        
        // Initialize test data
        std::vector<float> h_input(graph_test_size);
        for (int i = 0; i < graph_test_size; i++) {
            h_input[i] = (float)rand() / RAND_MAX;
        }
        CUDA_CHECK(cudaMemcpy(d_graph_input, h_input.data(), 
                            graph_test_size * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 blockSize(256);
        dim3 gridSize((graph_test_size + blockSize.x - 1) / blockSize.x);
        
        printf("Comparing traditional kernel launches vs CUDA graphs:\n");
        printf("Method               Time(ms) Launches/sec Overhead\n");
        printf("-------------------- -------- ------------ --------\n");
        
        // Test 1: Traditional kernel launches
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int iter = 0; iter < 1000; iter++) {
            graphKernel1<<<gridSize, blockSize, 0, graph_stream>>>(d_graph_input, d_graph_temp, graph_test_size);
            graphKernel2<<<gridSize, blockSize, 0, graph_stream>>>(d_graph_temp, d_graph_output, graph_test_size);
            graphKernel3<<<gridSize, blockSize, 0, graph_stream>>>(d_graph_output, graph_test_size);
        }
        CUDA_CHECK(cudaStreamSynchronize(graph_stream));
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float traditional_time;
        CUDA_CHECK(cudaEventElapsedTime(&traditional_time, start_event, stop_event));
        float traditional_launches_per_sec = (3000.0f * 1000.0f) / traditional_time; // 3 kernels * 1000 iterations
        
        printf("Traditional Launches %8.3f %12.0f Baseline\n", 
            traditional_time, traditional_launches_per_sec);
        
        // Test 2: CUDA Graph execution
        // First, capture the graph
        CUDA_CHECK(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal));
        
        graphKernel1<<<gridSize, blockSize, 0, graph_stream>>>(d_graph_input, d_graph_temp, graph_test_size);
        graphKernel2<<<gridSize, blockSize, 0, graph_stream>>>(d_graph_temp, d_graph_output, graph_test_size);
        graphKernel3<<<gridSize, blockSize, 0, graph_stream>>>(d_graph_output, graph_test_size);
        
        CUDA_CHECK(cudaStreamEndCapture(graph_stream, &cuda_graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, cuda_graph, NULL, NULL, 0));
        
        // Now benchmark graph execution
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int iter = 0; iter < 1000; iter++) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, graph_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(graph_stream));
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float graph_time;
        CUDA_CHECK(cudaEventElapsedTime(&graph_time, start_event, stop_event));
        float graph_launches_per_sec = (3000.0f * 1000.0f) / graph_time; // 3 kernels * 1000 iterations
        
        printf("CUDA Graph           %8.3f %12.0f ", graph_time, graph_launches_per_sec);
        
        float speedup = traditional_time / graph_time;
        printf("%.2fx faster\n", speedup);
        
        // Test 3: Graph update overhead
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int iter = 0; iter < 100; iter++) {
            // Update graph parameters (simulating dynamic behavior)
            cudaGraphNode_t* nodes = NULL;
            size_t numNodes = 0;
            CUDA_CHECK(cudaGraphGetNodes(cuda_graph, nodes, &numNodes));
            
            if (numNodes > 0) {
                nodes = new cudaGraphNode_t[numNodes];
                CUDA_CHECK(cudaGraphGetNodes(cuda_graph, nodes, &numNodes));
                
                // Update kernel parameters for first node
                cudaKernelNodeParams params;
                CUDA_CHECK(cudaGraphKernelNodeGetParams(nodes[0], &params));
                CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graph_exec, nodes[0], &params));
                
                delete[] nodes;
            }
            
            CUDA_CHECK(cudaGraphLaunch(graph_exec, graph_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(graph_stream));
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float update_time;
        CUDA_CHECK(cudaEventElapsedTime(&update_time, start_event, stop_event));
        float update_launches_per_sec = (300.0f * 1000.0f) / update_time; // 3 kernels * 100 iterations
        
        printf("Graph w/ Updates     %8.3f %12.0f ", update_time / 100.0f * 1000.0f, update_launches_per_sec);
        printf("%.2fx slower than static\n", (update_time / 100.0f) / (graph_time / 1000.0f));
    }

    void benchmarkSMUtilizationThermalThrottling() {
        printf("\n=== SM Utilization vs Thermal Throttling ===\n");
        
        // Initialize test data
        std::vector<float> h_data(config.instruction_test_size, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_thermal_workload, h_data.data(), 
                            config.instruction_test_size * sizeof(float), cudaMemcpyHostToDevice));
        
        int block_size = 256;
        int grid_size = (config.instruction_test_size + block_size - 1) / block_size;
        
        // Test different thermal loads — reduced intensities and iterations
        // to keep total runtime under ~10 seconds instead of minutes.
        std::vector<int> intensities = {50, 200, 500, 1000, 2000};
        const int sm_iters = 3;  // was 10
        
        printf("Intensity  Time(ms)  GFLOPS  Impact\n");
        printf("--------------------------------\n");
        
        float baseline_time = 0;
        
        for (int i = 0; i < intensities.size(); i++) {
            int intensity = intensities[i];
            
            // Warm up GPU
            thermalStressKernel<<<grid_size, block_size>>>(d_thermal_workload, 
                                                        config.instruction_test_size, intensity);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Time the kernel with thermal events
            CUDA_CHECK(cudaEventRecord(start_event));
            
            for (int j = 0; j < sm_iters; j++) {
                CUDA_CHECK(cudaEventRecord(thermal_events[j]));
                thermalStressKernel<<<grid_size, block_size>>>(d_thermal_workload, 
                                                            config.instruction_test_size, intensity);
            }
            
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float total_time;
            CUDA_CHECK(cudaEventElapsedTime(&total_time, start_event, stop_event));
            float avg_time = total_time / (float)sm_iters;
            
            if (i == 0) baseline_time = avg_time;
            
            // Estimate FLOPS (approximate)
            long long ops_per_thread = intensity * 15; // ~15 ops per iteration
            long long total_ops = (long long)config.instruction_test_size * ops_per_thread;
            float gflops = (total_ops / 1e9f) / (avg_time / 1000.0f);
            
            float thermal_impact = (avg_time / baseline_time) * 100.0f - 100.0f;
            
            printf("%9d  %8.2f  %6.1f  %+5.1f%%\n", 
                intensity, avg_time, gflops, thermal_impact);
        }
    }

    void benchmarkBVHTraversalEfficiency() {
        printf("\n=== Ray-BVH Traversal Efficiency ===\n");
        
        int block_size = 256;
        int grid_size = (num_rays_bvh + block_size - 1) / block_size;
        
        // Test different BVH configurations
        std::vector<int> node_counts = {1000, 5000, 10000, 20000};
        
        printf("Nodes   Time(ms)  MRays/s  Trav/Ray\n");
        printf("----------------------------------\n");
        
        for (int node_count : node_counts) {
            if (node_count > num_bvh_nodes) continue;
            
            CUDA_CHECK(cudaEventRecord(start_event));
            
            for (int i = 0; i < 50; i++) {
                bvhTraversalKernel<<<grid_size, block_size>>>(
                    d_bvh_nodes, d_rays, d_hit_results, d_hit_distances, 
                    num_rays_bvh, node_count);
            }
            
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float time;
            CUDA_CHECK(cudaEventElapsedTime(&time, start_event, stop_event));
            float avg_time = time / 50.0f;
            
            float rays_per_sec = (num_rays_bvh / 1e6f) / (avg_time / 1000.0f);
            float avg_traversals = logf((float)node_count) / logf(2.0f); // Approximate for binary tree
            
            printf("%5d   %8.2f  %7.1f     %4.1f\n", 
                node_count, avg_time, rays_per_sec, avg_traversals);
        }
        
        // Test ray coherency impact
        printf("\nCoherency Analysis:\n");
        printf("Type    Time(ms)  Efficiency\n");
        printf("----------------------------\n");
        
        // Test with coherent rays (similar directions)
        CUDA_CHECK(cudaEventRecord(start_event));
        bvhTraversalKernel<<<grid_size, block_size>>>(
            d_bvh_nodes, d_rays, d_hit_results, d_hit_distances, 
            num_rays_bvh, num_bvh_nodes);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float coherent_time;
        CUDA_CHECK(cudaEventElapsedTime(&coherent_time, start_event, stop_event));
        
        printf("Random  %8.2f      100.0%%\n", coherent_time);
    }

    void benchmarkMemoryAllocationOverhead() {
        printf("\n=== Memory Allocation Overhead ===\n");
        
        const int alloc_size = 1024 * 1024; // 1MB allocations
        const int num_tests = 100;
        
        printf("Method       Time(ms)  Allocs/s  Overhead\n");
        printf("-----------------------------------------\n");
        
        // Test traditional cudaMalloc
        CUDA_CHECK(cudaEventRecord(start_event));
        
        for (int i = 0; i < num_tests; i++) {
            CUDA_CHECK(cudaMalloc(&d_malloc_ptrs[i], alloc_size * sizeof(float)));
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float malloc_time;
        CUDA_CHECK(cudaEventElapsedTime(&malloc_time, start_event, stop_event));
        
        // Free traditional allocations
        for (int i = 0; i < num_tests; i++) {
            cudaFree(d_malloc_ptrs[i]);
        }
        
        // Test cudaMallocAsync
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        CUDA_CHECK(cudaEventRecord(start_event));
        
        for (int i = 0; i < num_tests; i++) {
            CUDA_CHECK(cudaMallocAsync(&d_malloc_async_ptrs[i], alloc_size * sizeof(float), stream));
        }
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float malloc_async_time;
        CUDA_CHECK(cudaEventElapsedTime(&malloc_async_time, start_event, stop_event));
        
        // Free async allocations
        for (int i = 0; i < num_tests; i++) {
            cudaFreeAsync(d_malloc_async_ptrs[i], stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        
        // Calculate results
        float malloc_allocs_per_sec = (num_tests * 1000.0f) / malloc_time;
        float async_allocs_per_sec = (num_tests * 1000.0f) / malloc_async_time;
        float overhead_reduction = ((malloc_time - malloc_async_time) / malloc_time) * 100.0f;
        float speedup = malloc_time / malloc_async_time;
        
        printf("cudaMalloc   %8.2f     %5.0f   baseline\n", 
            malloc_time, malloc_allocs_per_sec);
        printf("MallocAsync  %8.2f     %5.0f   %+6.1f%%\n", 
            malloc_async_time, async_allocs_per_sec, -overhead_reduction);
        
        printf("Speedup: %.2fx\n", speedup);
    }

    void benchmarkOccupancyLimitingFactors() {
        printf("\n=== Occupancy Limiting Factors ===\n");

        // Get device properties for occupancy calculations
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        const int test_size = config.occupancy_test_size;
        const int test_iterations = 50;
        
        // Initialize test data
        std::vector<float> h_input(test_size);
        for (int i = 0; i < test_size; i++) {
            h_input[i] = (float)rand() / RAND_MAX;
        }
        CUDA_CHECK(cudaMemcpy(d_occupancy_limit_input, h_input.data(), 
                            test_size * sizeof(float), cudaMemcpyHostToDevice));

        // Test different block sizes and configurations
        int block_sizes[] = {64, 128, 256, 512, 1024};
        int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

        printf("Block   Time(ms)  Occupancy  Best\n");
        printf("---------------------------------\n");

        printf("Low Register Usage:\n");
        for (int i = 0; i < num_block_sizes; i++) {
            int block_size = block_sizes[i];
            if (block_size > prop.maxThreadsPerBlock) continue;
            
            dim3 blockDim(block_size);
            dim3 gridDim((test_size + block_size - 1) / block_size);
            
            // Calculate theoretical occupancy
            int min_grid_size, best_block_size;
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &best_block_size,
                                                        lowRegisterKernel, 0, 0));
            
            float occupancy = 0.0f;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor((int*)&occupancy,
                                                                    lowRegisterKernel, block_size, 0));
            occupancy = occupancy * block_size / (float)prop.maxThreadsPerMultiProcessor;
            
            CUDA_CHECK(cudaEventRecord(start_event));
            for (int iter = 0; iter < test_iterations; iter++) {
                lowRegisterKernel<<<gridDim, blockDim>>>(d_occupancy_limit_input, 
                                                        d_occupancy_limit_output, test_size);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
            float avg_time = time_ms / test_iterations;
            
            printf("%5d   %8.3f     %5.1f%%  %4d\n", 
                block_size, avg_time, occupancy * 100.0f, best_block_size);
        }

        printf("High Register Usage:\n");
        for (int i = 0; i < num_block_sizes; i++) {
            int block_size = block_sizes[i];
            if (block_size > prop.maxThreadsPerBlock) continue;
            
            dim3 blockDim(block_size);
            dim3 gridDim((test_size + block_size - 1) / block_size);
            
            int min_grid_size, best_block_size;
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &best_block_size,
                                                        highRegisterKernel, 0, 0));
            
            float occupancy = 0.0f;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor((int*)&occupancy,
                                                                    highRegisterKernel, block_size, 0));
            occupancy = occupancy * block_size / (float)prop.maxThreadsPerMultiProcessor;
            
            CUDA_CHECK(cudaEventRecord(start_event));
            for (int iter = 0; iter < test_iterations; iter++) {
                highRegisterKernel<<<gridDim, blockDim>>>(d_occupancy_limit_input, 
                                                        d_occupancy_limit_output, test_size);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
            float avg_time = time_ms / test_iterations;
            
            printf("%5d   %8.3f     %5.1f%%  %4d\n", 
                block_size, avg_time, occupancy * 100.0f, best_block_size);
        }

        printf("Low Shared Memory:\n");
        for (int i = 0; i < num_block_sizes; i++) {
            int block_size = block_sizes[i];
            if (block_size > prop.maxThreadsPerBlock) continue;
            
            dim3 blockDim(block_size);
            dim3 gridDim((test_size + block_size - 1) / block_size);
            
            int min_grid_size, best_block_size;
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &best_block_size,
                                                        lowSharedMemoryKernel, 0, 0));
            
            float occupancy = 0.0f;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor((int*)&occupancy,
                                                                    lowSharedMemoryKernel, block_size, 0));
            occupancy = occupancy * block_size / (float)prop.maxThreadsPerMultiProcessor;
            
            CUDA_CHECK(cudaEventRecord(start_event));
            for (int iter = 0; iter < test_iterations; iter++) {
                lowSharedMemoryKernel<<<gridDim, blockDim>>>(d_occupancy_limit_input, 
                                                            d_occupancy_limit_output, test_size);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
            float avg_time = time_ms / test_iterations;
            
            printf("%5d   %8.3f     %5.1f%%  %4d\n", 
                block_size, avg_time, occupancy * 100.0f, best_block_size);
        }

        printf("High Shared Memory:\n");
        for (int i = 0; i < num_block_sizes; i++) {
            int block_size = block_sizes[i];
            if (block_size > prop.maxThreadsPerBlock) continue;
            
            // Calculate required shared memory (4 arrays of block_size floats each)
            size_t shared_mem_size = 4 * block_size * sizeof(float);
            if (shared_mem_size > prop.sharedMemPerBlock) {
                printf("%5d   SKIP (shared mem: %zuB > %zuB)\n",
                    block_size, shared_mem_size, prop.sharedMemPerBlock);
                continue;
            }
            
            dim3 blockDim(block_size);
            dim3 gridDim((test_size + block_size - 1) / block_size);
            
            int min_grid_size, best_block_size;
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &best_block_size,
                                                        highSharedMemoryKernel, shared_mem_size, 0));
            
            float occupancy = 0.0f;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor((int*)&occupancy,
                                                                    highSharedMemoryKernel, block_size, shared_mem_size));
            occupancy = occupancy * block_size / (float)prop.maxThreadsPerMultiProcessor;
            
            CUDA_CHECK(cudaEventRecord(start_event));
            for (int iter = 0; iter < test_iterations; iter++) {
                highSharedMemoryKernel<<<gridDim, blockDim, shared_mem_size>>>(d_occupancy_limit_input, 
                                                                            d_occupancy_limit_output, test_size);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
            float avg_time = time_ms / test_iterations;
            
            printf("%5d   %8.3f     %5.1f%%  %4d\n", 
                block_size, avg_time, occupancy * 100.0f, best_block_size);
        }

        printf("High Reg + Shared Mem:\n");
        for (int i = 0; i < num_block_sizes; i++) {
            int block_size = block_sizes[i];
            if (block_size > prop.maxThreadsPerBlock) continue;
            
            // Calculate required shared memory (6 arrays of block_size floats each)
            size_t shared_mem_size = 6 * block_size * sizeof(float);
            if (shared_mem_size > prop.sharedMemPerBlock) {
                printf("%5d   SKIP (shared mem: %zuB > %zuB)\n",
                    block_size, shared_mem_size, prop.sharedMemPerBlock);
                continue;
            }
            
            dim3 blockDim(block_size);
            dim3 gridDim((test_size + block_size - 1) / block_size);
            
            int min_grid_size, best_block_size;
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &best_block_size,
                                                        highRegisterAndSharedMemoryKernel, shared_mem_size, 0));
            
            float occupancy = 0.0f;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor((int*)&occupancy,
                                                                    highRegisterAndSharedMemoryKernel, block_size, shared_mem_size));
            occupancy = occupancy * block_size / (float)prop.maxThreadsPerMultiProcessor;
            
            CUDA_CHECK(cudaEventRecord(start_event));
            for (int iter = 0; iter < test_iterations; iter++) {
                highRegisterAndSharedMemoryKernel<<<gridDim, blockDim, shared_mem_size>>>(d_occupancy_limit_input, 
                                                                                        d_occupancy_limit_output, test_size);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
            float avg_time = time_ms / test_iterations;
            
            printf("%5d   %8.3f     %5.1f%%  %4d\n", 
                block_size, avg_time, occupancy * 100.0f, best_block_size);
        }
    }

    void benchmark_managed_on_demand(size_t total_bytes = size_t(1) << 30 /*1 GiB*/, size_t stride_bytes = 1<<20 /*1 MiB*/) {
        printf("\n=== Managed on-demand page migration benchmark ===\n");
        // allocate managed memory
        void* data = nullptr;
        CHECK_CUDA(cudaMallocManaged(&data, total_bytes));

        // initialise on host (touch every nth stride)
        printf("Touching host pages every %zu bytes to place pages on host.\n", stride_bytes);
        for (size_t off = 0; off < total_bytes; off += stride_bytes) {
            volatile char* p = (volatile char*)data + off;
            *p = 1;
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        char* base = reinterpret_cast<char*>(data);
        int blocks = 256;
        int threads = 256;

        // warm up
        managed_touch_kernel<<<blocks, threads>>>(base, total_bytes, stride_bytes);
        CHECK_CUDA(cudaDeviceSynchronize());

        GpuTimer t;
        t.startEvent();
        managed_touch_kernel<<<blocks, threads>>>(base, total_bytes, stride_bytes);
        float ms = t.stopMs();
        printf("Accessed %zu MiB (stride %zu KiB) in %.3f ms (kernel). Note: migration cost included.\n",
            total_bytes >> 20, stride_bytes >> 10, ms);

        // cleanup
        CHECK_CUDA(cudaFree(data));
    }

    void benchmark_cooperative_groups(int blocks = 16, int threads = 256, int rounds = 1000) {
        printf("\n=== Cooperative Groups synchronisation & scheduling overhead ===\n");

        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        int cooperativeLaunch = 0;
        CHECK_CUDA(cudaDeviceGetAttribute(&cooperativeLaunch, cudaDevAttrCooperativeLaunch, device));
        if (!cooperativeLaunch) {
            printf("Device does not support cooperative launch. Skipping cooperative benchmark.\n");
            return;
        }

        int* d_counter = nullptr;
        CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

        // warm up cooperative kernel
        void* kernelArgs[] = { &rounds, &d_counter }; // not used by this signature, use direct launch

        GpuTimer t;
        // For cooperative kernel we must use cudaLaunchCooperativeKernel
        dim3 grid(blocks);
        dim3 block(threads);
        t.startEvent();
        void* args[] = { &rounds, &d_counter }; // but kernel expects (int,int*) signature
        // Launch with correct function pointer
        CHECK_CUDA(cudaLaunchCooperativeKernel((void*)cg_sync_kernel, grid, block, args));
        float ms = t.stopMs();

        int host_count = 0;
        CHECK_CUDA(cudaMemcpy(&host_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
        printf("Cooperative kernel: blocks=%d threads=%d rounds=%d -> time=%.3f ms  counter=%d\n",
            blocks, threads, rounds, ms, host_count);

        CHECK_CUDA(cudaFree(d_counter));
    }

    void benchmark_tensor_cores(int M = 1024, int N = 1024, int K = 1024) {
        printf("\n=== Tensor core (WMMA) microbenchmark ===\n");
        // Ensure sizes are multiples of 16 for simplicity
        M = (M + 15) / 16 * 16;
        N = (N + 15) / 16 * 16;
        K = (K + 15) / 16 * 16;

        size_t Asz = size_t(M) * K;
        size_t Bsz = size_t(K) * N;
        size_t Csz = size_t(M) * N;

        half* d_A; half* d_B; float* d_C;
        CHECK_CUDA(cudaMalloc(&d_A, Asz * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_B, Bsz * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_C, Csz * sizeof(float)));

        // initialise A and B with simple values on host and copy
        half* hA = (half*)malloc(Asz * sizeof(half));
        half* hB = (half*)malloc(Bsz * sizeof(half));
        for (size_t i = 0; i < Asz; ++i) hA[i] = __float2half(1.0f);
        for (size_t i = 0; i < Bsz; ++i) hB[i] = __float2half(1.0f);
        CHECK_CUDA(cudaMemcpy(d_A, hA, Asz*sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, hB, Bsz*sizeof(half), cudaMemcpyHostToDevice));

        dim3 grid(M/16, N/16);
        dim3 block(32,1,1); // WMMA kernels often use 32 threads per warp/block

        // warm up
        wmma_gemm_fp16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        GpuTimer t;
        t.startEvent();
        wmma_gemm_fp16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        float ms = t.stopMs();

        double gflops = 2.0 * double(M) * double(N) * double(K) / (ms * 1e6); // GFLOP/s
        printf("WMMA GEMM: M=%d N=%d K=%d time=%.3f ms  approx=%.2f GFLOP/s\n", M, N, K, ms, gflops);

        // cleanup
        free(hA); free(hB);
        CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    }

    void benchmarkMultiDimensionalConvolution() {
        printf("\n=== Multi-Dimensional Convolution Benchmark ===\n");
        
        // 2D Convolution Test
        dim3 blockSize2d(16, 16);
        dim3 gridSize2d((conv2d_size + blockSize2d.x - 1) / blockSize2d.x,
                        (conv2d_size + blockSize2d.y - 1) / blockSize2d.y);
        
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 100; i++) {
            conv2dKernel<<<gridSize2d, blockSize2d>>>(d_conv2d_input, d_conv2d_output, 
                                                    d_conv2d_kernel, conv2d_size, conv2d_size, 5);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float conv2d_time;
        CUDA_CHECK(cudaEventElapsedTime(&conv2d_time, start_event, stop_event));
        
        size_t ops_2d = (size_t)conv2d_size * conv2d_size * 5 * 5 * 2; // multiply + add per output
        float gflops_2d = (ops_2d * 100) / (conv2d_time * 1e6f);
        
        printf("2D Convolution (%dx%d, 5x5 kernel):\n", conv2d_size, conv2d_size);
        printf("  Time: %.2f ms\n", conv2d_time / 100);
        printf("  Performance: %.2f GFLOPS\n", gflops_2d);
        
        // 3D Convolution Test
        dim3 blockSize3d(8, 8, 8);
        dim3 gridSize3d((conv3d_size + blockSize3d.x - 1) / blockSize3d.x,
                        (conv3d_size + blockSize3d.y - 1) / blockSize3d.y,
                        (conv3d_size + blockSize3d.z - 1) / blockSize3d.z);
        
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 50; i++) {
            conv3dKernel<<<gridSize3d, blockSize3d>>>(d_conv3d_input, d_conv3d_output, 
                                                    d_conv3d_kernel, conv3d_size, conv3d_size, 
                                                    conv3d_size, 3);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float conv3d_time;
        CUDA_CHECK(cudaEventElapsedTime(&conv3d_time, start_event, stop_event));
        
        size_t ops_3d = (size_t)conv3d_size * conv3d_size * conv3d_size * 3 * 3 * 3 * 2;
        float gflops_3d = (ops_3d * 50) / (conv3d_time * 1e6f);
        
        printf("3D Convolution (%dx%dx%d, 3x3x3 kernel):\n", conv3d_size, conv3d_size, conv3d_size);
        printf("  Time: %.2f ms\n", conv3d_time / 50);
        printf("  Performance: %.2f GFLOPS\n", gflops_3d);
    }

    void benchmarkBFSSSP() {
        printf("\n=== BFS/SSSP Graph Traversal Benchmark ===\n");
        
        dim3 blockSize(256);
        dim3 gridSize((num_graph_nodes + blockSize.x - 1) / blockSize.x);
        
        // BFS Test
        std::vector<int> h_levels(num_graph_nodes, -1);
        std::vector<int> h_visited(num_graph_nodes, 0);
        h_levels[0] = 0;
        h_visited[0] = 1;
        
        CUDA_CHECK(cudaMemcpy(d_bfs_levels, h_levels.data(), 
                            num_graph_nodes * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bfs_visited, h_visited.data(), 
                            num_graph_nodes * sizeof(int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start_event));
        
        int max_levels = 0;
        for (int level = 0; level < 100; level++) {
            bfsKernel<<<gridSize, blockSize>>>(d_graph_offsets, d_graph_edges, 
                                            d_bfs_levels, d_bfs_visited, 
                                            num_graph_nodes, level);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Check if any node at this level (simplified)
            max_levels = level + 1;
            if (level > 20) break; // Safety limit
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float bfs_time;
        CUDA_CHECK(cudaEventElapsedTime(&bfs_time, start_event, stop_event));
        
        printf("BFS Traversal:\n");
        printf("  Nodes: %d, Edges: %d\n", num_graph_nodes, num_graph_edges);
        printf("  Levels explored: %d\n", max_levels);
        printf("  Time: %.2f ms\n", bfs_time);
        printf("  Throughput: %.2f MNodes/s\n", (num_graph_nodes / 1e6f) / (bfs_time / 1000.0f));
        
        // SSSP Test
        std::vector<int> h_distances(num_graph_nodes, INT_MAX);
        h_distances[0] = 0;
        
        CUDA_CHECK(cudaMemcpy(d_sssp_distances, h_distances.data(), 
                            num_graph_nodes * sizeof(int), cudaMemcpyHostToDevice));
        
        int* d_updated;
        CUDA_CHECK(cudaMalloc(&d_updated, sizeof(int)));
        
        CUDA_CHECK(cudaEventRecord(start_event));
        
        int iterations = 0;
        for (int iter = 0; iter < 100; iter++) {
            int h_updated = 0;
            CUDA_CHECK(cudaMemcpy(d_updated, &h_updated, sizeof(int), cudaMemcpyHostToDevice));
            
            ssspKernel<<<gridSize, blockSize>>>(d_graph_offsets, d_graph_edges, d_graph_weights,
                                            d_sssp_distances, d_updated, num_graph_nodes);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost));
            iterations++;
            
            if (h_updated == 0) break;
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float sssp_time;
        CUDA_CHECK(cudaEventElapsedTime(&sssp_time, start_event, stop_event));
        
        printf("SSSP (Bellman-Ford style):\n");
        printf("  Nodes: %d, Edges: %d\n", num_graph_nodes, num_graph_edges);
        printf("  Iterations: %d\n", iterations);
        printf("  Time: %.2f ms\n", sssp_time);
        printf("  Throughput: %.2f MEdges/s\n", (num_graph_edges / 1e6f) / (sssp_time / 1000.0f));
        
        cudaFree(d_updated);
    }

    void benchmarkSIMTPerformance() {
        printf("\n=== SIMT Performance Improvement Benchmark ===\n");
        
        dim3 blockSize(256);
        dim3 gridSize((simt_test_size + blockSize.x - 1) / blockSize.x);
        
        // Test 1: Uniform execution (good SIMT)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 100; i++) {
            simtUniformKernel<<<gridSize, blockSize>>>(d_simt_uniform_input, 
                                                    d_simt_uniform_output, simt_test_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float uniform_time;
        CUDA_CHECK(cudaEventElapsedTime(&uniform_time, start_event, stop_event));
        
        // Test 2: Divergent execution (poor SIMT)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 100; i++) {
            simtDivergentKernel<<<gridSize, blockSize>>>(d_simt_divergent_input, 
                                                        d_simt_divergent_output,
                                                        d_simt_branch_data, simt_test_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float divergent_time;
        CUDA_CHECK(cudaEventElapsedTime(&divergent_time, start_event, stop_event));
        
        // Test 3: Balanced execution (better SIMT)
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 100; i++) {
            simtBalancedKernel<<<gridSize, blockSize>>>(d_simt_divergent_input, 
                                                        d_simt_divergent_output,
                                                        d_simt_branch_data, simt_test_size);
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float balanced_time;
        CUDA_CHECK(cudaEventElapsedTime(&balanced_time, start_event, stop_event));
        
        printf("Data size: %d elements\n", simt_test_size);
        printf("Pattern               Time(ms)  Relative  Efficiency\n");
        printf("-----------------------------------------------------\n");
        printf("Uniform (no diverge)  %8.2f     1.00x      100.0%%\n", 
            uniform_time / 100);
        printf("Divergent (4-way)     %8.2f     %.2fx      %5.1f%%\n", 
            divergent_time / 100, divergent_time / uniform_time,
            (uniform_time / divergent_time) * 100.0f);
        printf("Balanced (predicated) %8.2f     %.2fx      %5.1f%%\n", 
            balanced_time / 100, balanced_time / uniform_time,
            (uniform_time / balanced_time) * 100.0f);
        
        printf("\nSIMT Efficiency Analysis:\n");
        printf("  Divergence penalty: %.2fx slower\n", divergent_time / uniform_time);
        printf("  Predication improvement: %.2fx faster than divergent\n", 
            divergent_time / balanced_time);
        printf("  Warp utilization (uniform): ~100%%\n");
        printf("  Warp utilization (divergent): ~25%% (serialized execution)\n");
        printf("  Warp utilization (balanced): ~50%% (all execute, selective use)\n");
    }

    // ========================================================================
    // Softmax + Flash-Attention benchmark
    // ------------------------------------------------------------------------
    // Two parts:
    //   1. Softmax microbenchmark: naive 3-pass vs. online 1-pass over the
    //      same [M, N] input.  Reports time and effective HBM bandwidth.
    //
    //   2. Attention benchmark: naive attention (3 launches, materialises
    //      the full M×N score matrix) vs. flash attention (tiled, online
    //      softmax, single kernel, no score matrix in HBM).  Reports time,
    //      GFLOPS, and the max-abs-diff between the two implementations to
    //      confirm they agree numerically.
    // ========================================================================
    void benchmarkSoftmaxFlashAttention() {
        printf("\n=== Softmax + Flash-Attention Benchmark ===\n");

        const int M     = config.softmax_batch_heads;   // rows (batch * heads)
        const int N     = config.softmax_seq_len;        // row length / seq len
        const int D     = config.softmax_head_dim;       // head dim
        const float scale = 1.0f / sqrtf((float)D);
        const int iters = config.softmax_iters;

        printf("Configuration:\n");
        printf("  M (batch*heads): %d\n", M);
        printf("  N (seq_len):     %d\n", N);
        printf("  D (head_dim):    %d\n", D);
        printf("  Scale:           %.4f\n", scale);
        printf("  Iterations:      %d\n\n", iters);

        // ------------------------------------------------------------------
        // Part 1: softmax microbenchmark
        // ------------------------------------------------------------------
        printf("--- Softmax comparison (M=%d rows, N=%d cols) ---\n", M, N);

        // Helper lambda: time `iters` launches of a softmax kernel.
        auto time_softmax = [&](void (*kernel)(const float*, float*, int),
                                const char* tag) {
            // warmup
            kernel<<<M, 32>>>(d_softmax_input, d_softmax_output, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(start_event));
            for (int i = 0; i < iters; i++) {
                kernel<<<M, 32>>>(d_softmax_input, d_softmax_output, N);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));

            // Bytes touched: 1 read + 1 write of M*N floats
            size_t bytes = 2ULL * M * N * sizeof(float);
            double gbs = (bytes * iters) / (ms * 1e6);
            printf("  %-16s %7.3f ms/iter   %7.2f GB/s\n",
                   tag, ms / iters, gbs);
            return ms / iters;
        };

        float ms_naive  = time_softmax(softmax_naive_kernel,  "Naive 3-pass:");
        float ms_online = time_softmax(softmax_online_kernel, "Online 1-pass:");
        printf("  Speedup:         %.2fx  (online vs naive)\n", ms_naive / ms_online);

        // ------------------------------------------------------------------
        // Part 2: attention benchmark
        // ------------------------------------------------------------------
        printf("\n--- Attention comparison (M=N=%d, D=%d) ---\n", N, D);

        // Memory accounting: naive attention materialises S and (in-place) P,
        // each M*N floats.  Flash attention only touches Q, K, V, O plus
        // shared-memory tiles.
        size_t qkv_bytes   = 3ULL * M * D * sizeof(float) + M * D * sizeof(float);
        size_t naive_extra = 2ULL * M * N * sizeof(float);
        size_t flash_extra = 0;
        printf("  QKV+O size:      %.2f MB\n", qkv_bytes   / (1024.0 * 1024.0));
        printf("  Naive extra HBM: %.2f MB (S + P matrices)\n",
               naive_extra / (1024.0 * 1024.0));
        printf("  Flash extra HBM: %.2f MB (only shared-mem tiles)\n",
               flash_extra / (1024.0 * 1024.0));

        // FLOP accounting (per iteration):
        //   QK^T:    2*M*N*D     (mul+add per output element)
        //   softmax: ~3*M*N      (max, exp, div — counted approximately)
        //   PV:      2*M*N*D
        //   Total ≈ 4*M*N*D  (softmax term is small for D=64)
        double flops_per_iter = 4.0 * (double)M * (double)N * (double)D;

        // ----- Naive attention: 3 launches -----
        {
            dim3 block_qk(64);
            dim3 grid_qk((N + block_qk.x - 1) / block_qk.x, M);
            dim3 block_pv(64);
            dim3 grid_pv((D + block_pv.x - 1) / block_pv.x, M);

            // warmup
            attention_qk_kernel<<<grid_qk, block_qk>>>(
                d_attn_Q, d_attn_K, d_attn_S, M, N, D, scale);
            softmax_naive_kernel<<<M, 32>>>(d_attn_S, d_attn_S, N);  // in-place
            attention_pv_kernel<<<grid_pv, block_pv>>>(
                d_attn_S, d_attn_V, d_attn_O_naive, M, N, D);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(start_event));
            for (int i = 0; i < iters; i++) {
                attention_qk_kernel<<<grid_qk, block_qk>>>(
                    d_attn_Q, d_attn_K, d_attn_S, M, N, D, scale);
                softmax_naive_kernel<<<M, 32>>>(d_attn_S, d_attn_S, N);
                attention_pv_kernel<<<grid_pv, block_pv>>>(
                    d_attn_S, d_attn_V, d_attn_O_naive, M, N, D);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));

            double gflops = (flops_per_iter * iters) / (ms * 1e6);
            printf("\n  Naive attention (3 launches, materialises S+P):\n");
            printf("    Time:     %7.3f ms/iter\n", ms / iters);
            printf("    Perf:     %7.2f GFLOPS\n", gflops);
        }

        // ----- Flash attention (tiled, single kernel) -----
        constexpr int Br = 16;
        constexpr int Bc = 64;
        constexpr int Dc = 64;  // must match FA_D / config.softmax_head_dim
        const int blocks  = (M + Br - 1) / Br;
        const int threads = 128;

        // Shared memory layout (bytes):
        //   sK [Bc*D] + sV [Bc*D] + sS [Br*Bc] + sO [Br*D] + sM [Br] + sL [Br]
        size_t smem_bytes = (size_t)(Bc * Dc + Bc * Dc + Br * Bc + Br * Dc + 2 * Br)
                             * sizeof(float);

        // Allow the kernel to use up to 96 KB of dynamic shared memory on
        // devices that support it (Ampere+).  The 48 KB default is enough
        // for our tile sizes, but this guards against future enlargements.
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (smem_bytes > prop.sharedMemPerBlock) {
            printf("\n  Flash attention: shared-memory requirement (%zu B) exceeds "
                   "device default (%zu B).  Skipping.\n",
                   smem_bytes, (size_t)prop.sharedMemPerBlock);
        } else {
            if (smem_bytes > 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    (const void*)flash_attention_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)smem_bytes));
            }

            // warmup
            flash_attention_kernel<<<blocks, threads, smem_bytes>>>(
                d_attn_Q, d_attn_K, d_attn_V, d_attn_O_flash, M, N, scale);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(start_event));
            for (int i = 0; i < iters; i++) {
                flash_attention_kernel<<<blocks, threads, smem_bytes>>>(
                    d_attn_Q, d_attn_K, d_attn_V, d_attn_O_flash, M, N, scale);
            }
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));

            double gflops = (flops_per_iter * iters) / (ms * 1e6);
            printf("\n  Flash attention (tiled, online softmax):\n");
            printf("    Time:     %7.3f ms/iter\n", ms / iters);
            printf("    Perf:     %7.2f GFLOPS\n", gflops);
            printf("    Tile:     Br=%d Bc=%d D=%d  threads=%d  smem=%.1f KB\n",
                   Br, Bc, Dc, threads, smem_bytes / 1024.0);

            // ----- Correctness check: naive vs flash -----
            std::vector<float> h_naive((size_t)M * D);
            std::vector<float> h_flash((size_t)M * D);
            CUDA_CHECK(cudaMemcpy(h_naive.data(), d_attn_O_naive,
                                  (size_t)M * D * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_flash.data(), d_attn_O_flash,
                                  (size_t)M * D * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            float max_diff = 0.0f, sum_diff = 0.0f;
            for (size_t i = 0; i < (size_t)M * D; i++) {
                float diff = fabsf(h_naive[i] - h_flash[i]);
                if (diff > max_diff) max_diff = diff;
                sum_diff += diff;
            }
            printf("\n  Correctness check (max abs diff naive vs flash):\n");
            printf("    Max abs diff:   %e\n", max_diff);
            printf("    Mean abs diff:  %e\n", sum_diff / ((size_t)M * D));
            printf("    (FP32 round-off expected; values < 1e-3 are normal)\n");
        }
    }

    // ========================================================================
    // NUMA bandwidth benchmark
    // ------------------------------------------------------------------------
    // For each NUMA node visible from /sys/devices/system/node, this benchmark
    // pins a worker thread to a CPU on that node, allocates pinned host memory
    // (which the kernel places on the local node via the first-touch policy),
    // and measures Host->Device and Device->Host bandwidth to the current GPU.
    //
    // The GPU's "local" NUMA node is read from
    //   /sys/bus/pci/devices/<domain:bus:dev.func>/numa_node
    // so we can call out the local-vs-remote NUMA penalty.
    // ========================================================================
    struct NumaBenchResult {
        int node_id;
        bool is_local;
        double h2d_gb_s;
        double d2h_gb_s;
    };

    void benchmarkNUMABandwidth() {
        printf("\n=== NUMA Bandwidth Benchmark ===\n");

#if !defined(__linux__)
        printf("NUMA benchmarking requires Linux sysfs (/sys/devices/system/node).\n");
        printf("Skipping on this platform.\n");
        return;
#else
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        int gpu_numa = getGpuNumaNode(prop.pciDomainID, prop.pciBusID,
                                       prop.pciDeviceID, /*pci_func=*/0);

        auto nodes = detectNumaNodes();
        if (nodes.empty()) {
            printf("No multi-NUMA topology detected. Treating system as single NUMA node 0.\n");
            NumaNode synthetic;
            synthetic.node_id = 0;
            synthetic.cpus.push_back(0);
            nodes.push_back(synthetic);
        } else {
            printf("Detected %zu NUMA node(s):", nodes.size());
            for (const auto& n : nodes) printf(" node%d(%zu cpus)", n.node_id, n.cpus.size());
            printf("\n");
        }

        printf("GPU %d ('%s') local NUMA node: ", device, prop.name);
        if (gpu_numa >= 0) printf("%d\n", gpu_numa);
        else printf("unknown (single-NUMA assumed)\n");

        size_t bytes = (size_t)config.numa_transfer_size * sizeof(float);
        int iters = config.numa_transfer_iterations;
        int warmup = config.numa_transfer_warmup;

        printf("Transfer size: %.2f MB, iterations: %d (warmup %d)\n",
               bytes / (1024.0 * 1024.0), iters, warmup);

        // Use the same device buffer for all NUMA nodes
        float* d_buf = d_pcie_data; // reuse existing allocation (sized for pcie_transfer_size)
        size_t d_buf_bytes = (size_t)config.pcie_transfer_size * sizeof(float);
        if (bytes > d_buf_bytes) {
            // Need a bigger device buffer for this test
            CUDA_CHECK(cudaMalloc(&d_buf, bytes));
        }

        printf("\nNode  Local?  H2D GB/s  D2H GB/s  H2D/D2H\n");
        printf("--------------------------------------------\n");

        std::vector<NumaBenchResult> results;
        for (const auto& node : nodes) {
            if (node.cpus.empty()) continue;
            int target_cpu = node.cpus[0];
            NumaBenchResult r;
            r.node_id = node.node_id;
            r.is_local = (gpu_numa >= 0 && node.node_id == gpu_numa);
            r.h2d_gb_s = 0;
            r.d2h_gb_s = 0;

            // Spawn a worker thread pinned to target_cpu; everything from
            // cudaHostAlloc through the timed transfers runs on that thread
            // so the first-touch NUMA policy places the host buffer on the
            // correct node.
            std::thread worker([&]() {
#if defined(__linux__)
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(target_cpu, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
                float* h_pinned = nullptr;
                CUDA_CHECK(cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault));
                // Touch every page so allocation lands on this NUMA node
                for (size_t i = 0; i < config.numa_transfer_size; i += 1024) {
                    h_pinned[i] = (float)i * 1.0e-6f;
                }
                cudaDeviceSynchronize();

                // Warm up
                for (int i = 0; i < warmup; i++) {
                    cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice);
                    cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost);
                }
                cudaDeviceSynchronize();

                cudaEvent_t ev_a, ev_b;
                cudaEventCreate(&ev_a);
                cudaEventCreate(&ev_b);

                // Time H2D
                cudaEventRecord(ev_a);
                for (int i = 0; i < iters; i++) {
                    cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice);
                }
                cudaEventRecord(ev_b);
                cudaEventSynchronize(ev_b);
                float h2d_ms = 0;
                cudaEventElapsedTime(&h2d_ms, ev_a, ev_b);

                // Time D2H
                cudaEventRecord(ev_a);
                for (int i = 0; i < iters; i++) {
                    cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost);
                }
                cudaEventRecord(ev_b);
                cudaEventSynchronize(ev_b);
                float d2h_ms = 0;
                cudaEventElapsedTime(&d2h_ms, ev_a, ev_b);

                r.h2d_gb_s = (iters * bytes) / (h2d_ms * 1e6);
                r.d2h_gb_s = (iters * bytes) / (d2h_ms * 1e6);

                cudaEventDestroy(ev_a);
                cudaEventDestroy(ev_b);
                cudaFreeHost(h_pinned);
            });
            worker.join();

            results.push_back(r);
            printf("%4d  %6s  %9.2f  %9.2f  %7.2f\n",
                   r.node_id,
                   r.is_local ? "YES" : "no",
                   r.h2d_gb_s,
                   r.d2h_gb_s,
                   (r.d2h_gb_s > 0) ? r.h2d_gb_s / r.d2h_gb_s : 0.0);
        }

        // Summary: local vs best-remote NUMA penalty
        const NumaBenchResult* local = nullptr;
        const NumaBenchResult* best_remote = nullptr;
        for (const auto& r : results) {
            if (r.is_local) {
                if (!local || r.h2d_gb_s > local->h2d_gb_s) local = &r;
            } else {
                if (!best_remote || r.h2d_gb_s > best_remote->h2d_gb_s) best_remote = &r;
            }
        }

        printf("\nSummary:\n");
        if (local) {
            printf("  Local NUMA H2D: %.2f GB/s, D2H: %.2f GB/s\n",
                   local->h2d_gb_s, local->d2h_gb_s);
        }
        if (best_remote) {
            printf("  Best remote NUMA H2D: %.2f GB/s, D2H: %.2f GB/s (node %d)\n",
                   best_remote->h2d_gb_s, best_remote->d2h_gb_s, best_remote->node_id);
            if (local && best_remote->h2d_gb_s > 0 && best_remote->d2h_gb_s > 0) {
                printf("  NUMA penalty (H2D): %.2fx slower on remote\n",
                       local->h2d_gb_s / best_remote->h2d_gb_s);
                printf("  NUMA penalty (D2H): %.2fx slower on remote\n",
                       local->d2h_gb_s / best_remote->d2h_gb_s);
            }
        } else if (local) {
            printf("  (Only one NUMA node active; no remote comparison possible.)\n");
        } else if (!results.empty()) {
            printf("  (No local NUMA node identified; GPU may be on an unknown NUMA node.)\n");
        }

        // Cleanup if we allocated our own device buffer
        if (bytes > d_buf_bytes) {
            cudaFree(d_buf);
        }
#endif // __linux__
    }

    // ========================================================================
    // NVLink-aware P2P bandwidth benchmark
    // ------------------------------------------------------------------------
    // Iterates over every unordered pair of GPUs (i, j) and measures:
    //   - unidirectional peer-copy bandwidth i -> j
    //   - bidirectional peer-copy bandwidth (i <-> j overlapped via streams)
    //
    // Before the timed transfers the benchmark queries NVML for active
    // NVLink connections between the two devices (and the NVLink major
    // version), and queries CUDA's cudaDevP2PAttrPerformanceRank attribute.
    // If NVML is unavailable, the link type is inferred from the measured
    // bandwidth (>50 GB/s unidirectional is almost certainly NVLink).
    // ========================================================================
    struct P2PBenchResult {
        int src, dst;
        int nvlink_version;     // 0 = no NVLink detected
        bool p2p_enabled;
        double uni_gb_s;
        double bi_gb_s;
    };

    void benchmarkNVLinkBandwidth() {
        printf("\n=== NVLink / P2P Bandwidth Benchmark ===\n");

        if (num_gpus < 2) {
            printf("Only 1 GPU detected - NVLink/P2P test skipped.\n");
            return;
        }
        printf("Detected %d GPUs. Testing every unordered pair.\n\n", num_gpus);

        std::vector<P2PBenchResult> all_results;

        for (int src = 0; src < num_gpus; src++) {
            for (int dst = src + 1; dst < num_gpus; dst++) {
                P2PBenchResult r;
                r.src = src; r.dst = dst;
                r.nvlink_version = 0;
                r.p2p_enabled = false;
                r.uni_gb_s = 0; r.bi_gb_s = 0;

                cudaDeviceProp psrc, pdst;
                cudaGetDeviceProperties(&psrc, src);
                cudaGetDeviceProperties(&pdst, dst);

                printf("--- GPU %d <-> GPU %d ---\n", src, dst);
                printf("  %s (GPU %d) <-> %s (GPU %d)\n",
                       psrc.name, src, pdst.name, dst);

                int can_01 = 0, can_10 = 0;
                cudaDeviceCanAccessPeer(&can_01, src, dst);
                cudaDeviceCanAccessPeer(&can_10, dst, src);
                printf("  P2P access %d->%d: %s, %d->%d: %s\n",
                       src, dst, can_01 ? "yes" : "no",
                       dst, src, can_10 ? "yes" : "no");

                if (!can_01 && !can_10) {
                    printf("  No peer access supported. Skipping pair.\n\n");
                    all_results.push_back(r);
                    continue;
                }

                // CUDA P2P performance rank (cudaDeviceGetP2PAttribute was
                // introduced in CUDA 8.0; the older cudaGetDeviceP2PAttribute
                // alias was removed in CUDA 13.0).
                int perf_rank = 0;
                cudaError_t rank_err = cudaErrorUnknown;
#if CUDART_VERSION >= 8000
                rank_err = cudaDeviceGetP2PAttribute(
                    &perf_rank, cudaDevP2PAttrPerformanceRank, src, dst);
#endif
                if (rank_err == cudaSuccess) {
                    printf("  P2P performance rank: %d\n", perf_rank);
                }
                cudaGetLastError(); // clear

                // NVLink detection via NVML
                r.nvlink_version = detectNvLinkBetween(src, dst);
                if (r.nvlink_version > 0) {
                    printf("  NVLink detected (v%d)\n", r.nvlink_version);
                } else {
                    printf("  NVLink not detected by NVML; will infer from bandwidth.\n");
                }

                // Enable peer access (ignore failures - they are non-fatal)
                cudaSetDevice(src);
                cudaError_t e1 = cudaDeviceEnablePeerAccess(dst, 0);
                if (e1 != cudaSuccess && e1 != cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError();
                }
                cudaSetDevice(dst);
                cudaError_t e2 = cudaDeviceEnablePeerAccess(src, 0);
                if (e2 != cudaSuccess && e2 != cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError();
                }
                r.p2p_enabled = true;

                size_t bytes = (size_t)config.nvlink_transfer_size * sizeof(float);
                int iters = config.nvlink_transfer_iterations;
                int warmup = config.nvlink_transfer_warmup;

                // Allocate per-pair buffers
                float *d_src_buf = nullptr, *d_dst_buf = nullptr;
                cudaSetDevice(src);
                CUDA_CHECK(cudaMalloc(&d_src_buf, bytes));
                cudaSetDevice(dst);
                CUDA_CHECK(cudaMalloc(&d_dst_buf, bytes));

                // Initialise source with thrust sequence
                cudaSetDevice(src);
                {
                    thrust::device_vector<float> init(config.nvlink_transfer_size);
                    thrust::sequence(init.begin(), init.end(), 1.0f);
                    CUDA_CHECK(cudaMemcpy(d_src_buf,
                                          thrust::raw_pointer_cast(init.data()),
                                          bytes, cudaMemcpyDeviceToDevice));
                }

                // Warm up
                for (int i = 0; i < warmup; i++) {
                    cudaMemcpyPeer(d_dst_buf, dst, d_src_buf, src, bytes);
                }
                cudaDeviceSynchronize();

                // Unidirectional timing (events on src device)
                cudaSetDevice(src);
                cudaEvent_t ev_start, ev_stop;
                cudaEventCreate(&ev_start);
                cudaEventCreate(&ev_stop);
                cudaEventRecord(ev_start);
                for (int i = 0; i < iters; i++) {
                    cudaMemcpyPeer(d_dst_buf, dst, d_src_buf, src, bytes);
                }
                cudaEventRecord(ev_stop);
                cudaEventSynchronize(ev_stop);
                float uni_ms = 0;
                cudaEventElapsedTime(&uni_ms, ev_start, ev_stop);
                r.uni_gb_s = (iters * bytes) / (uni_ms * 1e6);

                // Bidirectional timing using two streams (one on each device)
                cudaStream_t s_src, s_dst;
                cudaSetDevice(src);
                cudaStreamCreate(&s_src);
                cudaSetDevice(dst);
                cudaStreamCreate(&s_dst);

                cudaSetDevice(src);
                cudaEvent_t ev_bi_start, ev_bi_stop;
                cudaEventCreate(&ev_bi_start);
                cudaEventCreate(&ev_bi_stop);

                cudaEventRecord(ev_bi_start, s_src);
                for (int i = 0; i < iters; i++) {
                    cudaMemcpyPeerAsync(d_dst_buf, dst, d_src_buf, src, bytes, s_src);
                    cudaMemcpyPeerAsync(d_src_buf, src, d_dst_buf, dst, bytes, s_dst);
                }
                cudaEventRecord(ev_bi_stop, s_src);
                cudaEventSynchronize(ev_bi_stop);
                cudaStreamSynchronize(s_dst);
                float bi_ms = 0;
                cudaEventElapsedTime(&bi_ms, ev_bi_start, ev_bi_stop);
                // bidirectional: 2 transfers per iteration
                r.bi_gb_s = (2.0 * iters * bytes) / (bi_ms * 1e6);

                // Link-type inference (used as fallback when NVML is silent)
                const char* link_label;
                if (r.nvlink_version > 0) {
                    link_label = (r.nvlink_version >= 3) ? "NVLink (new)"
                                 : (r.nvlink_version == 2) ? "NVLink v2"
                                 : "NVLink v1";
                } else if (r.uni_gb_s > 50.0) {
                    link_label = "NVLink (inferred)";
                } else if (r.uni_gb_s > 25.0) {
                    link_label = "PCIe 4.0/5.0 (inferred)";
                } else {
                    link_label = "PCIe 3.0 (inferred)";
                }

                printf("  Link type:     %s\n", link_label);
                printf("  Transfer size: %.2f MB, iterations: %d\n",
                       bytes / (1024.0 * 1024.0), iters);
                printf("  Unidirectional: %7.2f GB/s\n", r.uni_gb_s);
                printf("  Bidirectional:  %7.2f GB/s\n", r.bi_gb_s);
                printf("  Bi/Uni ratio:   %.2f%s\n",
                       r.uni_gb_s > 0 ? r.bi_gb_s / r.uni_gb_s : 0.0,
                       (r.uni_gb_s > 0 && r.bi_gb_s / r.uni_gb_s > 1.6) ?
                           " (full-duplex)" : "");

                // Cleanup
                cudaEventDestroy(ev_start);
                cudaEventDestroy(ev_stop);
                cudaEventDestroy(ev_bi_start);
                cudaEventDestroy(ev_bi_stop);
                cudaStreamDestroy(s_src);
                cudaStreamDestroy(s_dst);
                cudaSetDevice(src);
                cudaFree(d_src_buf);
                cudaSetDevice(dst);
                cudaFree(d_dst_buf);

                all_results.push_back(r);
                printf("\n");
            }
        }

        // Aggregate summary across all pairs
        printf("--- Aggregate NVLink/P2P Summary ---\n");
        if (all_results.empty()) {
            printf("  No pairs tested.\n");
        } else {
            int nvlink_pairs = 0, pcie_pairs = 0;
            double best_uni = 0, best_bi = 0;
            for (const auto& r : all_results) {
                if (r.nvlink_version > 0 || r.uni_gb_s > 50.0) nvlink_pairs++;
                else pcie_pairs++;
                best_uni = std::max(best_uni, r.uni_gb_s);
                best_bi = std::max(best_bi, r.bi_gb_s);
            }
            printf("  Pairs tested:    %zu\n", all_results.size());
            printf("  NVLink pairs:    %d\n", nvlink_pairs);
            printf("  PCIe-only pairs: %d\n", pcie_pairs);
            printf("  Best unidir BW:  %.2f GB/s\n", best_uni);
            printf("  Best bidir BW:   %.2f GB/s\n", best_bi);
        }

        cudaSetDevice(0);
    }

    // ========================================================================
    // GPU power efficiency benchmark
    // ------------------------------------------------------------------------
    // Runs a "light" workload (partial-grid FMA, low arithmetic intensity)
    // and a "heavy" workload (full-grid FMA + transcendentals, max power
    // draw) for a fixed wall-clock duration each.  While each workload is
    // running, a background NVML sampler polls SM clock, memory clock, power
    // draw, temperature and throttle reasons.
    //
    // Outputs:
    //   - GFLOPS achieved for each workload
    //   - Average / peak power (W) and SM clock (MHz)
    //   - Clock drop-off % (peak vs sustained)
    //   - GFLOPS / Watt efficiency for each workload
    //   - Throttle reason bitmask (decoded)
    //
    // If NVML is unavailable, falls back to GFLOPS-only reporting.
    // ========================================================================
    void benchmarkPowerEfficiency() {
        printf("\n=== GPU Power Efficiency Benchmark ===\n");

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
        printf("SMs: %d, max threads/block: %d\n",
               prop.multiProcessorCount, prop.maxThreadsPerBlock);

        GpuMetricSampler sampler;
        bool has_nvml = sampler.init(device);
        if (!has_nvml) {
            printf("NVML unavailable - power/clock sampling disabled.\n");
            printf("Will report GFLOPS only; GFLOPS/Watt requires NVML.\n");
        } else {
            unsigned int power_limit_mw = sampler.getPowerLimitMw();
            printf("NVML initialised. Power limit: ");
            if (power_limit_mw > 0) printf("%.1f W\n", power_limit_mw / 1000.0);
            else printf("unknown\n");
            printf("Sampling every %d ms for %d ms per workload.\n",
                   config.power_efficiency_sample_ms,
                   config.power_efficiency_duration_ms);
        }
        printf("\n");

        const int block_size = 256;
        const int full_grid =
            (config.power_efficiency_workload_size + block_size - 1) / block_size;
        // Light grid: ~1/4 of the SMs worth of blocks, so the GPU is only
        // partially occupied (typical "light load" scenario).
        const int light_grid = std::max(1, prop.multiProcessorCount / 4);
        const int heavy_grid = std::max(full_grid, prop.multiProcessorCount * 2);

        const size_t workload_bytes =
            (size_t)config.power_efficiency_workload_size * sizeof(float);
        // Reuse d_thermal_workload if it is large enough, else allocate fresh
        float* d_workload = d_thermal_workload;
        bool owns_d_workload = false;
        size_t thermal_bytes = (size_t)config.instruction_test_size * sizeof(float);
        if (workload_bytes > thermal_bytes) {
            CUDA_CHECK(cudaMalloc(&d_workload, workload_bytes));
            owns_d_workload = true;
        }
        // Initialise workload
        std::vector<float> h_init(config.power_efficiency_workload_size, 1.0f);
        for (int i = 0; i < config.power_efficiency_workload_size; i++) {
            h_init[i] = (float)(i % 1024) * 1.0e-3f + 0.5f;
        }
        CUDA_CHECK(cudaMemcpy(d_workload, h_init.data(), workload_bytes,
                              cudaMemcpyHostToDevice));

        // Helper: run a kernel of choice for the configured duration while
        // the NVML sampler runs in the background.  Returns total GFLOPS
        // achieved, the elapsed GPU-side time in ms, and the NVML stats.
        struct WorkloadOutcome {
            double gflops = 0;
            float gpu_ms = 0;
            GpuMetricSampler::Stats stats;
        };

        auto run_workload = [&](const char* tag, int grid, int ops_per_thread,
                                int flops_per_iter, int threads_per_block,
                                bool use_mixed) -> WorkloadOutcome {
            long long flops_per_launch =
                (long long)grid * threads_per_block * (long long)ops_per_thread * flops_per_iter;

            // Warm up
            for (int i = 0; i < 3; i++) {
                if (use_mixed)
                    heavyMixedWorkloadKernel<<<grid, threads_per_block>>>(
                        d_workload, config.power_efficiency_workload_size, ops_per_thread);
                else if (flops_per_iter == 2)
                    lightWorkloadKernel<<<grid, threads_per_block>>>(
                        d_workload, config.power_efficiency_workload_size, ops_per_thread);
                else
                    heavyWorkloadKernel<<<grid, threads_per_block>>>(
                        d_workload, config.power_efficiency_workload_size, ops_per_thread);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // Optional warm-up window for NVML to settle
            if (has_nvml && config.power_efficiency_warmup_ms > 0) {
                auto t0 = std::chrono::steady_clock::now();
                while (true) {
                    if (use_mixed)
                        heavyMixedWorkloadKernel<<<grid, threads_per_block>>>(
                            d_workload, config.power_efficiency_workload_size, ops_per_thread);
                    else if (flops_per_iter == 2)
                        lightWorkloadKernel<<<grid, threads_per_block>>>(
                            d_workload, config.power_efficiency_workload_size, ops_per_thread);
                    else
                        heavyWorkloadKernel<<<grid, threads_per_block>>>(
                            d_workload, config.power_efficiency_workload_size, ops_per_thread);
                    auto t1 = std::chrono::steady_clock::now();
                    int ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                    if (ms >= config.power_efficiency_warmup_ms) break;
                }
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // Begin timed run + sampler
            if (has_nvml) sampler.start(config.power_efficiency_sample_ms);
            CUDA_CHECK(cudaEventRecord(start_event));
            auto wall_t0 = std::chrono::steady_clock::now();

            long long total_flops = 0;
            int launches = 0;
            while (true) {
                if (use_mixed)
                    heavyMixedWorkloadKernel<<<grid, threads_per_block>>>(
                        d_workload, config.power_efficiency_workload_size, ops_per_thread);
                else if (flops_per_iter == 2)
                    lightWorkloadKernel<<<grid, threads_per_block>>>(
                        d_workload, config.power_efficiency_workload_size, ops_per_thread);
                else
                    heavyWorkloadKernel<<<grid, threads_per_block>>>(
                        d_workload, config.power_efficiency_workload_size, ops_per_thread);
                launches++;
                total_flops += flops_per_launch;

                auto wall_t1 = std::chrono::steady_clock::now();
                int elapsed_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                    wall_t1 - wall_t0).count();
                if (elapsed_ms >= config.power_efficiency_duration_ms) break;
            }

            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            if (has_nvml) sampler.stop();

            float gpu_ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start_event, stop_event));
            double gflops = (total_flops / 1e9) / (gpu_ms / 1000.0);

            WorkloadOutcome out;
            out.gflops = gflops;
            out.gpu_ms = gpu_ms;
            if (has_nvml) out.stats = sampler.computeStats();

            printf("--- %s ---\n", tag);
            printf("  Grid: %d blocks x %d threads, %d ops/thread (%d FLOPs/iter)\n",
                   grid, threads_per_block, ops_per_thread, flops_per_iter);
            printf("  Duration: %.2f s (GPU time), %d launches\n",
                   gpu_ms / 1000.0, launches);
            printf("  Total work: %.3f GFLOP\n", total_flops / 1e9);
            printf("  Throughput: %.2f GFLOPS\n", gflops);

            if (out.stats.valid) {
                printf("  Power:        avg %.1f W, peak %.1f W\n",
                       out.stats.avg_power_w, out.stats.peak_power_w);
                printf("  SM clock:     avg %.0f MHz, peak %.0f MHz, min %.0f MHz\n",
                       out.stats.avg_sm_clock_mhz, out.stats.peak_sm_clock_mhz,
                       out.stats.min_sm_clock_mhz);
                printf("  Mem clock:    avg %.0f MHz\n", out.stats.avg_mem_clock_mhz);
                printf("  Temperature:  avg %.1f C, peak %.1f C\n",
                       out.stats.avg_temp_c, out.stats.peak_temp_c);
                printf("  Clock drop:   %.1f%% (peak -> sustained avg)\n",
                       out.stats.clock_drop_pct);
                printf("  Throttle:     %zu/%zu samples",
                       out.stats.throttle_samples, out.stats.total_samples);
                if (out.stats.last_throttle_reasons) {
                    printf(" [%s]", decodeThrottleReasons(out.stats.last_throttle_reasons).c_str());
                }
                printf("\n");
                if (out.stats.avg_power_w > 0) {
                    printf("  Efficiency:   %.3f GFLOPS/W (%.2f GFLOPS / %.1f W)\n",
                           gflops / out.stats.avg_power_w, gflops, out.stats.avg_power_w);
                }
            } else if (has_nvml) {
                printf("  (NVML sampler did not collect enough samples.)\n");
            }
            printf("\n");
            return out;
        };

        // Light workload: partial grid, few ops/thread, 2 FLOPs/iter
        WorkloadOutcome light = run_workload(
            "Light Workload",
            light_grid,
            config.power_efficiency_light_ops,
            /*flops_per_iter=*/2,
            block_size,
            /*use_mixed=*/false);

        // Reset workload data between runs so accumulators do not overflow
        CUDA_CHECK(cudaMemcpy(d_workload, h_init.data(), workload_bytes,
                              cudaMemcpyHostToDevice));

        // Heavy workload: full grid, many ops/thread, 8 FLOPs/iter, pure FMA
        WorkloadOutcome heavy = run_workload(
            "Heavy Workload (pure FMA)",
            heavy_grid,
            config.power_efficiency_heavy_ops,
            /*flops_per_iter=*/8,
            block_size,
            /*use_mixed=*/false);

        CUDA_CHECK(cudaMemcpy(d_workload, h_init.data(), workload_bytes,
                              cudaMemcpyHostToDevice));

        // Heavy mixed workload: full grid + transcendentals (worst-case power)
        // FLOPs/iter is approximate due to transcendentals - we use 8 FMA + ~4 SFU
        WorkloadOutcome heavy_mixed = run_workload(
            "Heavy Workload (FMA + SFU)",
            heavy_grid,
            config.power_efficiency_heavy_ops / 2,
            /*flops_per_iter=*/12,
            block_size,
            /*use_mixed=*/true);

        // Summary table - uses the per-workload captured stats, not a proxy.
        printf("--- Power Efficiency Summary ---\n");
        printf("Workload           GFLOPS   Power(W)  GFLOPS/W  ClockDrop  Throttle\n");
        printf("-------------------------------------------------------------------\n");
        auto print_row = [&](const char* name, const WorkloadOutcome& w) {
            double pwr = w.stats.valid ? w.stats.avg_power_w : 0;
            double drop = w.stats.valid ? w.stats.clock_drop_pct : 0;
            const char* thr = w.stats.valid
                ? (w.stats.throttle_samples > 0 ? "yes" : "no")
                : "n/a";
            printf("%-18s %8.2f %9.1f %9.3f %9.1f%%  %8s\n",
                   name, w.gflops, pwr, (pwr > 0 ? w.gflops / pwr : 0.0), drop, thr);
        };
        print_row("Light", light);
        print_row("Heavy (FMA)", heavy);
        print_row("Heavy (FMA+SFU)", heavy_mixed);

        // Clock drop-off comparison
        if (has_nvml && light.stats.valid && heavy.stats.valid) {
            printf("\nClock drop-off analysis (peak -> sustained):\n");
            printf("  Light SM clock: %.0f -> %.0f MHz (%.1f%% drop)\n",
                   light.stats.peak_sm_clock_mhz, light.stats.avg_sm_clock_mhz,
                   light.stats.clock_drop_pct);
            printf("  Heavy SM clock: %.0f -> %.0f MHz (%.1f%% drop)\n",
                   heavy.stats.peak_sm_clock_mhz, heavy.stats.avg_sm_clock_mhz,
                   heavy.stats.clock_drop_pct);
            if (light.stats.avg_power_w > 0 && heavy.stats.avg_power_w > 0) {
                printf("  Light:Heavy power ratio: %.2fx\n",
                       heavy.stats.avg_power_w / light.stats.avg_power_w);
                printf("  Light:Heavy GFLOPS ratio: %.2fx\n",
                       heavy.gflops / light.gflops);
                double eff_light = light.gflops / light.stats.avg_power_w;
                double eff_heavy = heavy.gflops / heavy.stats.avg_power_w;
                printf("  Light vs Heavy efficiency: %.3f vs %.3f GFLOPS/W (%.2fx)\n",
                       eff_light, eff_heavy, eff_light / eff_heavy);
            }
        }

        if (owns_d_workload) cudaFree(d_workload);
    }

    void runAllBenchmarks() {
        printf("                  ---  CUBench  ---\n");
        printf(" The Definitive Open-Source GPU Benchmarking Utility\n");
        printf(" ===================================================\n\n");
        
        // Detect number of GPUs
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        
        if (deviceCount == 0) {
            printf("No CUDA-capable devices found!\n");
            return;
        }
        
        printf("Detected %d CUDA-capable device(s)\n\n", deviceCount);
        
        // Loop through each GPU
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            printf("\n");
            printf("========================================================\n");
            printf("         BENCHMARKING GPU %d OF %d\n", deviceId + 1, deviceCount);
            printf("========================================================\n\n");
            
            // Set current device
            CUDA_CHECK(cudaSetDevice(deviceId));
            
            // Print a random funny message before benchmarks
            const char* funny_messages[] = {
                "Warning: GPU may become sentient during benchmarking.",
                "If your GPU starts making toast, please contact support.",
                "Benchmarking: Because your GPU needs a workout too.",
                "Caution: Results may cause feelings of inadequacy in your CPU.",
                "If you can read this, your GPU is faster than a potato.",
                "Running benchmarks... Please do not feed the GPU.",
                "Your GPU called. It wants more triangles.",
                "This benchmark was brought to you by electrons.",
                "GPU motto: 'I compute, therefore I am.'",
                "If this crashes, blame cosmic rays.",
                "Pro tip: You can get faster results if you cheer for your GPU.",
                "Warning: May cause spontaneous pixelation. If this happens, consult your GPU's manufacturer. If you are a GPU manufacturer, hello!",
                "Proof that AI frame gen is useless",
                "GPU is currently busy flexing its FLOPS. give us a moment as we fix that...",
                "If you see smoke, that's just the GPU working hard.",
                "The one benchmarking software where a fire extinguisher is optional.",
                "Remember: All benchmarks are fun until someone divides by zero.",
                "It's all fun and games until this starts testing your CPU.",
                "Completely Unorganised Benchmark software",
                "If your fans get louder, it means it's working.",
                "Sacrificing development sanity to the CUDA gods...",
                "CUDA: Completely Unnecessary Driver Absurdity.",
                "Programming is all fun and games until you somehow divide by a string.",
                "Benchmarking in progress: Please hold onto your hats (and your data).",
                "Caution: Excessive benchmarking may lead to GPU enlightenment.",
                "If your GPU starts humming, it's just happy to help.",
                "If your GPU starts speaking in binary, don't be alarmed.",
                "This benchmark is powered by caffeine and sheer determination.",
                "If your GPU starts glowing, get better fans.",
            };
            int num_msgs = sizeof(funny_messages) / sizeof(funny_messages[0]);
            srand((unsigned int)time(NULL) + deviceId); // Add deviceId for variation
            int msg_idx = rand() % num_msgs;
            printf(">>> %s\n\n", funny_messages[msg_idx]);

            // Print GPU info
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
            printf("Device ID: %d\n", deviceId);
            printf("GPU: %s\n", prop.name);
            printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
            printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
            printf("Multiprocessors: %d\n", prop.multiProcessorCount);
            printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
            printf("PCI Bus ID: %04x:%02x:%02x.%d\n",
                   prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, 0);
            // NUMA node (Linux only - shows which CPU NUMA node the GPU is on)
            int gpu_numa = getGpuNumaNode(prop.pciDomainID, prop.pciBusID,
                                          prop.pciDeviceID, 0);
            if (gpu_numa >= 0) {
                printf("NUMA Node: %d\n", gpu_numa);
            } else {
                printf("NUMA Node: unknown (single-NUMA system or unsupported platform)\n");
            }
            // Memory / L2 / clock info
            printf("L2 Cache: %d KB\n", prop.l2CacheSize / 1024);
            // memoryClockRate and clockRate in cudaDeviceProp are platform-dependent
            // (not always present on Windows). Use cudaDeviceGetAttribute() instead,
            // which is portable across all supported platforms.
            int mem_clock_khz = 0, sm_clock_khz = 0;
            cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, deviceId);
            cudaDeviceGetAttribute(&sm_clock_khz,  cudaDevAttrClockRate,       deviceId);
            printf("Memory Clock: %.0f MHz\n", mem_clock_khz * 1e-3);
            printf("SM Clock (max): %.0f MHz\n", sm_clock_khz * 1e-3);
            printf("\n");
            printf("Please wait as benchmarks are being performed. this may take a while...\n\n");

            // Rest of the benchmark code remains the same (list of benchmark lambdas, capture function, etc.)
            struct BenchCall { const char* name; std::function<void(RenderBenchmark*)> fn; };
            std::vector<BenchCall> bench_calls = {
                // ... (keep all the existing benchmark calls)
                {"Rasterisation", [](auto self){ self->benchmarkRasterisation(); }},
                {"Particles", [](auto self){ self->benchmarkParticles(); }},
                {"RayTracing", [](auto self){ self->benchmarkRayTracing(); }},
                {"TextureFiltering", [](auto self){ self->benchmarkTextureFiltering(); }},
                {"ComputeShader", [](auto self){ self->benchmarkComputeShader(); }},
                {"DeferredShading", [](auto self){ self->benchmarkDeferredShading(); }},
                {"MatrixOperations", [](auto self){ self->benchmarkMatrixOperations(); }},
                {"MemoryBandwidth", [](auto self){ self->benchmarkMemoryBandwidth(); }},
                {"Tessellation", [](auto self){ self->benchmarkTessellation(); }},
                {"Reduction", [](auto self){ self->reduction(); }},
                {"PrefixSum", [](auto self){ self->prefixSum(); }},
                {"Sort", [](auto self){ self->sortBench(); }},
                {"SparseSpMV", [](auto self){ self->sparseSpMVBenchmark(); }},
                {"FFTMul", [](auto self){ self->fftMultiply(); }},
                {"AtomicOps", [](auto self){ self->benchmarkAtomicOperations(); }},
                {"MemoryTypes", [](auto self){ self->benchmarkMemoryTypes(); }},
                {"InitNewBenchData", [](auto self){ self->initializeNewBenchmarkData(); }},
                {"InstrThroughput", [](auto self){ self->benchmarkInstructionThroughput(); }},
                {"OccupancyWarpDiv", [](auto self){ self->benchmarkOccupancyAndWarpDivergence(); }},
                {"GPUIdleContext", [](auto self){ self->benchmarkGPUIdleContextSwitchOverhead(); }},
                {"ILPAnalysis", [](auto self){ self->benchmarkILPAnalysis(); }},
                {"ThreadDivergence", [](auto self){ self->benchmarkThreadDivergence(); }},
                {"DynamicParallelism", [](auto self){ self->benchmarkDynamicParallelism(); }},
                {"FlatKernelEq", [](auto self){ self->benchmarkFlatKernelEquivalent(); }},
                {"PersistentThreads", [](auto self){ self->benchmarkPersistentThreads(); }},
                {"TextureCacheEff", [](auto self){ self->benchmarkTextureCacheEfficiency(); }},
                {"RegisterPressure", [](auto self){ self->benchmarkRegisterPressureSpilling(); }},
                {"MemoryLatency", [](auto self){ self->benchmarkMemoryLatency(); }},
                {"PCIeBandwidth", [](auto self){ self->benchmarkPCIeBandwidth(); }},
                {"P2PComm", [](auto self){ self->benchmarkP2PCommunication(); }},
                {"AsyncExec", [](auto self){ self->benchmarkAsynchronousExecution(); }},
                {"InstructionMix", [](auto self){ self->benchmarkInstructionMix(); }},
                {"CacheThrashConf", [](auto self){ self->benchmarkCacheThrashingAndConflicts(); }},
                {"BankConflictTest", [](auto self){ self->benchmarkBankConflictTesting(); }},
                {"PersistentOccupancy", [](auto self){ self->benchmarkPersistentOccupancyVsContextSwitching(); }},
                {"CUDAGraphOverhead", [](auto self){ self->benchmarkCUDAGraphExecutionOverhead(); }},
                {"SMUtilisation", [](auto self){ self ->benchmarkSMUtilizationThermalThrottling(); }},
                {"BVHEfficiency", [](auto self){ self ->benchmarkBVHTraversalEfficiency(); }},
                {"MemoryAllocOverhead", [](auto self){ self ->benchmarkMemoryAllocationOverhead(); }},
                {"OccupancyLimiting", [](auto self){ self ->benchmarkOccupancyLimitingFactors(); }},
                {"ManagedOnDemand", [](auto self){ self ->benchmark_managed_on_demand(); }},
                {"CooperativeGroups", [](auto self){ self ->benchmark_cooperative_groups(); }},
                {"TensorCores", [](auto self){ self ->benchmark_tensor_cores(); }},
                {"MultiDimConv", [](auto self){ self->benchmarkMultiDimensionalConvolution(); }},
                {"BFS_SSSP", [](auto self){ self->benchmarkBFSSSP(); }},
                {"SIMT_Performance", [](auto self){ self->benchmarkSIMTPerformance(); }},
                {"SoftmaxFlashAttn", [](auto self){ self->benchmarkSoftmaxFlashAttention(); }},
                {"NUMA_Bandwidth", [](auto self){ self->benchmarkNUMABandwidth(); }},
                {"NVLink_Bandwidth", [](auto self){ self->benchmarkNVLinkBandwidth(); }},
                {"PowerEfficiency", [](auto self){ self->benchmarkPowerEfficiency(); }},
            };

            struct BenchResult {
                std::string name;
                std::string output;
            };
            std::vector<BenchResult> results;

            auto capture = [](auto fn, const char* name, RenderBenchmark* self) {
                std::string out;
                FILE* temp = tmpfile();
                int temp_fd = fileno(temp);
                int old_fd = dup(fileno(stdout));
                fflush(stdout);
                dup2(temp_fd, fileno(stdout));
                fn(self);
                fflush(stdout);
                dup2(old_fd, fileno(stdout));
                close(old_fd);
                fseek(temp, 0, SEEK_SET);
                char buf[4096];
                while (fgets(buf, sizeof(buf), temp)) out += buf;
                fclose(temp);
                return BenchResult{name, out};
            };

            // Run and capture each benchmark, with a live progress bar on stderr.
            // The progress bar writes to stderr so it bypasses the stdout-capture
            // mechanism — the user sees live progress while benchmark output is
            // buffered for the column layout below.
            {
                char gpu_label[300];
                snprintf(gpu_label, sizeof(gpu_label), "GPU %d/%d (%s)",
                         deviceId + 1, deviceCount, prop.name);
                ProgressBar progress((int)bench_calls.size(), gpu_label);

                for (size_t i = 0; i < bench_calls.size(); i++) {
                    progress.start_task((int)(i + 1), bench_calls[i].name);
                    results.push_back(capture(bench_calls[i].fn, bench_calls[i].name, this));
                    progress.finish_task();
                }
                progress.finish_all();
            }

            // Print results in columns, balancing by output length (lines) per column
            printf("\n--- Benchmark Results ---\n\n");
            const int num_cols = 3;
            size_t num_bench = results.size();

            // Count lines in each result
            std::vector<size_t> bench_lines(num_bench, 0);
            for (size_t i = 0; i < num_bench; ++i) {
                size_t lines = 0;
                std::istringstream iss(results[i].output);
                std::string line;
                while (std::getline(iss, line)) ++lines;
                bench_lines[i] = lines;
            }

            // Greedy assignment: fill columns to balance total lines
            std::vector<std::vector<size_t>> col_indices(num_cols);
            std::vector<size_t> col_line_totals(num_cols, 0);
            for (size_t i = 0; i < num_bench; ++i) {
                // Find column with least total lines
                size_t min_col = 0;
                for (size_t c = 1; c < num_cols; ++c)
                if (col_line_totals[c] < col_line_totals[min_col]) min_col = c;
                col_indices[min_col].push_back(i);
                col_line_totals[min_col] += bench_lines[i];
            }

            // Build columns: split output into lines, track max width
            std::vector<std::vector<std::string>> columns(num_cols);
            std::vector<size_t> col_widths(num_cols, 0);
            for (int col = 0; col < num_cols; ++col) {
                for (size_t idx : col_indices[col]) {
                std::istringstream iss(results[idx].output);
                std::string line;
                while (std::getline(iss, line)) {
                    columns[col].push_back(line);
                    col_widths[col] = std::max(col_widths[col], line.size());
                }
                // Add a blank line between benchmarks
                columns[col].push_back("");
                }
            }

            // Find max lines per column
            size_t max_lines = 0;
            for (auto& col : columns)
                max_lines = std::max(max_lines, col.size());

            // Print each line of each column side by side
            for (size_t line = 0; line < max_lines; ++line) {
                for (size_t col = 0; col < num_cols; ++col) {
                if (line < columns[col].size())
                    printf("%-*s  ", (int)col_widths[col], columns[col][line].c_str());
                else
                    printf("%-*s  ", (int)col_widths[col], "");
                }
                printf("\n");
            }
        } // End of device loop
        
        printf("\n\n");
        printf("========================================================\n");
        printf("   ALL GPU BENCHMARKS COMPLETED!\n");
        printf("========================================================\n");
        
        std::cout << "\nBenchmark completed! Press any key to quit." << std::endl;
        _getch();
    }
};

int main() {
    // Configure benchmark
    BenchmarkConfig config;
    
    // Detect all GPUs and create benchmark instances for each
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found!\n");
        return 1;
    }
    
    // Note: We only create one benchmark instance and switch devices within runAllBenchmarks
    // This avoids memory allocation issues with multiple instances
    CUDA_CHECK(cudaSetDevice(0));
    RenderBenchmark benchmark(config);
    benchmark.runAllBenchmarks();
    
    return 0;
}
