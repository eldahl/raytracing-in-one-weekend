# Add GPU Rendering with CUDA

I will implement GPU rendering using CUDA. This involves porting the core ray tracing logic to run on the GPU. I will adapt the existing C++ classes to be compatible with CUDA by adding `__host__ __device__` qualifiers and managing memory on the device.

## User Review Required

> [!IMPORTANT]
> This implementation requires an NVIDIA GPU and the CUDA Toolkit installed. I verified `nvcc` is available.

## Proposed Changes

### Build System

#### [MODIFY] [CMakeLists.txt](file:///home/eldahl/Projects/raytracing-in-one-weekend/CMakeLists.txt)
- Enable CUDA language.
- Add a new executable `rtiow_cuda` (or similar) compiled from `src/main.cu`.
- Link against CUDA libraries.

### Core Headers

#### [MODIFY] [src/vec3.h](file:///home/eldahl/Projects/raytracing-in-one-weekend/src/vec3.h)
- Add `__host__ __device__` qualifiers to all methods.
- Add preprocessor definitions to handle `__host__` and `__device__` when compiling with a standard C++ compiler.

#### [MODIFY] [src/ray.h](file:///home/eldahl/Projects/raytracing-in-one-weekend/src/ray.h)
- Add `__host__ __device__` qualifiers.

### New CUDA Implementation

#### [NEW] [src/main.cu](file:///home/eldahl/Projects/raytracing-in-one-weekend/src/main.cu)
- Implement the main entry point for the CUDA renderer.
- Implement CUDA kernels for rendering (`render_kernel`) and initialization (`render_init`, `create_world`).
- Use `curand` for random number generation on the GPU.
- Manage device memory for the image and scene objects.

#### [NEW] [src/cuda_utils.h](file:///home/eldahl/Projects/raytracing-in-one-weekend/src/cuda_utils.h)
- Helper macros for CUDA error checking (`checkCudaErrors`).

## Verification Plan

### Automated Tests
- I will run the new executable `rtiow_cuda` and check if it produces an image.
- `build/rtiow_cuda > image_cuda.ppm`

### Manual Verification
- Compare the output image `image_cuda.ppm` with the CPU-generated `image.ppm` (visually).
- Verify that the GPU rendering is significantly faster.
