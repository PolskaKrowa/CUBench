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
#else
    #include <unistd.h>
#endif
#include <algorithm>
#include <random>
#include <cstdarg>
#include <iomanip>
#include <string>

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

// Benchmark configuration
struct BenchmarkConfig {
    int width = 1920;
    int height = 1080;
    int iterations = 100;
    int num_triangles = 10000;
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
        
        for (int i = 0; i < benchmark->config.iterations; i++) {
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
        
        float time = timeKernelExecution(rasterisationKernel, this);
        float fps = (config.iterations * 1000.0f) / time;
        
        printf("Triangles: %d\n", config.num_triangles);
        printf("Resolution: %dx%d\n", config.width, config.height);
        printf("Time: %.2f ms (%.2f FPS)\n", time / config.iterations, fps);
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
        
        for (int i = 0; i < benchmark->config.iterations; i++) {
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
        
        float time = timeKernelExecution(rayTracingKernel, this);
        float fps = (config.iterations * 1000.0f) / time;
        
        printf("Spheres: %d\n", config.num_triangles);
        printf("Rays: %d\n", config.width * config.height);
        printf("Time: %.2f ms (%.2f FPS)\n", time / config.iterations, fps);
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
        
        CUDA_CHECK(cudaEventRecord(start_event));
        
        for (int i = 0; i < config.iterations; i++) {
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
        float fps = (config.iterations * 1000.0f) / milliseconds;
        
        printf("Lights: %d\n", num_lights);
        printf("G-Buffer + Lighting Time: %.2f ms (%.2f FPS)\n", milliseconds / config.iterations, fps);
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
        
        // Test Host-to-Device transfers
        // Pinned memory
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 10; i++) {
            CUDA_CHECK(cudaMemcpy(d_pcie_data, h_pcie_pinned, bytes, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float pinned_h2d_time;
        CUDA_CHECK(cudaEventElapsedTime(&pinned_h2d_time, start_event, stop_event));
        
        // Pageable memory
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 10; i++) {
            CUDA_CHECK(cudaMemcpy(d_pcie_data, h_pcie_pageable, bytes, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float pageable_h2d_time;
        CUDA_CHECK(cudaEventElapsedTime(&pageable_h2d_time, start_event, stop_event));
        
        // Test Device-to-Host transfers
        // Pinned memory
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 10; i++) {
            CUDA_CHECK(cudaMemcpy(h_pcie_pinned, d_pcie_data, bytes, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float pinned_d2h_time;
        CUDA_CHECK(cudaEventElapsedTime(&pinned_d2h_time, start_event, stop_event));
        
        // Pageable memory
        CUDA_CHECK(cudaEventRecord(start_event));
        for (int i = 0; i < 10; i++) {
            CUDA_CHECK(cudaMemcpy(h_pcie_pageable, d_pcie_data, bytes, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float pageable_d2h_time;
        CUDA_CHECK(cudaEventElapsedTime(&pageable_d2h_time, start_event, stop_event));
        
        // Calculate bandwidths (GB/s)
        float pinned_h2d_bw = (10.0f * bytes) / (pinned_h2d_time * 1e6f);
        float pageable_h2d_bw = (10.0f * bytes) / (pageable_h2d_time * 1e6f);
        float pinned_d2h_bw = (10.0f * bytes) / (pinned_d2h_time * 1e6f);
        float pageable_d2h_bw = (10.0f * bytes) / (pageable_d2h_time * 1e6f);
        
        printf("Host-to-Device:\n");
        printf("  Pinned Memory: %.2f GB/s\n", pinned_h2d_bw);
        printf("  Pageable Memory: %.2f GB/s\n", pageable_h2d_bw);
        printf("Device-to-Host:\n");
        printf("  Pinned Memory: %.2f GB/s\n", pinned_d2h_bw);
        printf("  Pageable Memory: %.2f GB/s\n", pageable_d2h_bw);
        printf("Pinned vs Pageable Speedup:\n");
        printf("  H2D: %.2fx\n", pinned_h2d_bw / pageable_h2d_bw);
        printf("  D2H: %.2fx\n", pinned_d2h_bw / pageable_d2h_bw);
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
        
        // Test different thermal loads
        std::vector<int> intensities = {100, 500, 1000, 2000, 5000};
        
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
            
            for (int j = 0; j < 10; j++) {
                CUDA_CHECK(cudaEventRecord(thermal_events[j]));
                thermalStressKernel<<<grid_size, block_size>>>(d_thermal_workload, 
                                                            config.instruction_test_size, intensity);
            }
            
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float total_time;
            CUDA_CHECK(cudaEventElapsedTime(&total_time, start_event, stop_event));
            float avg_time = total_time / 10.0f;
            
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

    void runAllBenchmarks() {
        printf("                  ---  CUBench  ---\n");
        printf(" The Definitive Open-Source GPU Benchmarking Utility\n");
        printf(" ===================================================\n\n");
        
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
            "Programming is all fun and games until you somehow divide by a string."
        };
        int num_msgs = sizeof(funny_messages) / sizeof(funny_messages[0]);
        srand((unsigned int)time(NULL));
        int msg_idx = rand() % num_msgs;
        printf(">>> %s\n\n", funny_messages[msg_idx]);

        // Print GPU info
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("GPU: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
        printf("\n");
        printf("Please wait as benchmarks are being performed. this may take a while...\n\n");

        // List of benchmark lambdas and names
        struct BenchCall { const char* name; std::function<void(RenderBenchmark*)> fn; };
        std::vector<BenchCall> bench_calls = {
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
            {"OccupancyLimiting", [](auto self){ self ->benchmarkOccupancyLimitingFactors(); }}
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

        // Run and capture each benchmark
        for (const auto& b : bench_calls) {
            results.push_back(capture(b.fn, b.name, this));
        }

        // Print results in columns, balancing by output length (lines) per column
        printf("\n--- Benchmark Results ---\n\n");
        const int num_cols = 4;
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

        std::cout << "\nBenchmark completed! Press any key to quit." << std::endl;
        _getch();
    }
};

int main() {
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    // Configure benchmark
    BenchmarkConfig config;
    
    // Run benchmark
    RenderBenchmark benchmark(config);
    benchmark.runAllBenchmarks();
    
    return 0;
}
