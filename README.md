# CUBench

**The Definitive Open-Source GPU Benchmarking Utility**

A comprehensive CUDA-based tool for evaluating GPU performance across a variety of kernel configurations, memory access patterns and occupancy scenarios.

## Features

* Measures execution time over multiple trials for statistical significance
* Tests a range of thread block sizes and grid configurations
* Reports per-kernel occupancy and register utilisation
* Customisable benchmark parameters via `BenchmarkConfig`
* Outputs human-readable summaries to the console

## Requirements

* CUDA Toolkit (version 11.0 or later)
* NVCC compiler
* C++17-compatible standard library
* A CUDA-capable GPU

## Building

1. Ensure the CUDA environment variables are set (e.g. `CUDA_HOME`).

2. Compile with NVCC:

   ```bash
   nvcc -std=c++17 main.cu -o cubench -lcusparse -lcufft
   ```

3. (Optional) Add optimisation flags:

   ```bash
   nvcc -O3 -std=c++17 main.cu -o cubench -lcusparse -lcufft
   ```

## Usage

Run the benchmark executable. By default, it uses device 0 and the settings in `BenchmarkConfig`:

```bash
./cubench
```

Sample output:

```
=== Rasterisation Benchmark ===
Triangles: 10000
Resolution: 1920x1080
Time: 158.24 ms (6.32 FPS)
Triangles/sec: 0.06 M
Pixels/sec: 13.10 M
```

### Custom Configuration

Modify the `BenchmarkConfig` struct in `main.cu` to tweak:

* Number of trials per test
* Input sizes for occupancy and memory benchmarks
* Minimum and maximum block sizes

Recompile after changes.

## Notes

I have a lot of other projects i need to create and maintain. expect delayed bugfixes / features / responses.

### Planned features

```
Large page migration between host and device (on-demand memory)
Synchronisation and Scheduling overhead (Cooperative Groups Performance)
Specialised Unit testing (RT cores, Tensor cores, etc.)
NUMA & NVLink Bandwidth
Rewrite all benchmarks to test all GPUs and get average or individual results
GPU power efficiency (GFLOPs/W under light and heavy sustained loads to measure clock drop-off)
Multi Dimension Convolution testing
BFS/SSSP Benchmarking
SIMT performance improvement
```

## Licence

This project is licensed under the Apache License 2.0. See [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) for details.

## Author

Stevenson Parker

Created: 24 July 2025
