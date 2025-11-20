#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <iostream>
#include <memory>
#include <cstdlib>

#ifdef __CUDACC__
#include <curand_kernel.h>
#define HOST_DEVICE __host__ __device__
#define RAND_STATE curandState *local_rand_state = nullptr

__host__ __device__ inline double cuda_random_double(curandState *local_rand_state) {
#ifdef __CUDA_ARCH__
    return curand_uniform(local_rand_state);
#else
    return std::rand() / (RAND_MAX + 1.0);
#endif
}

#define RANDOM_DOUBLE cuda_random_double(local_rand_state)
template<typename T> using SharedPtr = T*;
#define MAKE_SHARED(T, ...) (new T(__VA_ARGS__))

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#else
#define HOST_DEVICE
#define RAND_STATE void* local_rand_state = nullptr
#define RANDOM_DOUBLE random_double()
template<typename T> using SharedPtr = std::shared_ptr<T>;
#define MAKE_SHARED(T, ...) std::make_shared<T>(__VA_ARGS__)
#endif

#endif
