# GPU Benchmark Results

## Overview

This document contains benchmark results comparing CPU vs GPU performance for various computer vision operations in the RETINA library.

## Test Environment

- **Platform**: Linux
- **GPU**: Intel Meteor Lake (integrated GPU)
- **CPU**: Modern x86_64 with AVX2 SIMD
- **Backend**: WGPU/WebGPU

## Key Finding: Integrated GPU vs Discrete GPU

**Integrated GPUs** (like Intel Meteor Lake) share memory with the CPU and are not optimized for compute-heavy workloads. **Discrete GPUs** have dedicated VRAM and compute units.

For integrated GPUs:
- Memory transfer overhead is lower (unified memory)
- But compute parallelism is limited
- CPU SIMD (AVX2) is extremely efficient for element-wise ops

For discrete GPUs:
- Massive parallelism at scale
- Dedicated memory bandwidth
- TensorCore acceleration for matmul/conv

## Benchmark Results

### Matrix Multiplication (Matmul) ✅ GPU WINS

| Size | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| 256x256 | 28 ms | 2.8 ms | **10x** |
| 512x512 | ~220 ms | 12 ms | **18x** |

**Why GPU wins**: O(n³) compute intensity parallelizes perfectly across GPU cores.

### Voxel Grid Downsampling ✅ GPU WINS (at scale)

| Points | CPU Time | GPU Time | Speedup |
|--------|----------|----------|---------|
| 16K | 1.6 ms | 6.5 ms | CPU wins |
| 65K | 5.5 ms | 8.3 ms | CPU wins |
| 262K | 19 ms | 15 ms | **1.3x** |
| 1M | ~60 ms | 49 ms | **1.2x** |

**Why GPU wins at scale**: Hash-based voxel assignment with atomic accumulation parallelizes well for large point clouds.

### Element-wise Operations ❌ CPU WINS

| Operation | Size | CPU Time | GPU Time | Winner |
|-----------|------|----------|----------|--------|
| Add | 65K | 90 µs | 2.5 ms | CPU 27x |
| Fused (add+relu+sigmoid) | 262K | 950 µs | 6 ms | CPU 6x |

**Why CPU wins**: 
1. GPU command submission overhead (~2ms) dominates
2. CPU SIMD (AVX2) processes 8 floats/cycle
3. For memory-bound ops, CPU is extremely efficient

## Optimizations Implemented

### 1. GPU Voxel Grid (`crates/hal/src/gpu_kernels/mod.rs`)

- **Proper centroid computation**: Uses atomic<u32> with bitcast for float accumulation
- **Buffer pooling**: Reuses GPU buffers across calls to reduce allocation overhead
- **Direct voxel indexing**: Uses direct indexing when voxel grid is small (<65K voxels)
- **Hash-based for large**: Falls back to spatial hash for large voxel grids

### 2. GPU Matmul

- Simple tiled matrix multiplication using WGSL compute shaders
- 8x8 workgroup size for good occupancy

### 3. Fused Operations

- Demonstrates doing multiple ops (add + relu + sigmoid) in one kernel
- Reduces transfer overhead by doing more computation per dispatch

## Recommendations

### Use GPU When:
- Matrix multiplication (any size > 32x32)
- Convolution operations
- Large point clouds (> 100K points)
- Neural network inference
- Data is already on GPU (no transfer needed)

### Use CPU When:
- Element-wise operations (add, relu, sigmoid, etc.)
- Small data (< 10K elements)
- Integrated GPU without dedicated VRAM
- Latency-critical operations

## Running Benchmarks

```bash
# CPU benchmarks only
cargo bench --bench cubecl_perf

# CPU + GPU benchmarks
cargo bench --bench cubecl_perf --features gpu
```

## Future Work

1. **Test on discrete GPU** - Results would show GPU winning at smaller sizes
2. **Add Conv2D benchmarks** - Would likely show GPU winning at moderate sizes
3. **Add pipeline benchmarks** - Measure GPU advantage when data stays on GPU
4. **Optimize element-wise** - Use compute shader arrays for better throughput
