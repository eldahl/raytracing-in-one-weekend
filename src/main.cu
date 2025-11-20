#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"
#include "cuda_utils.h"

// Kernel to initialize random state
__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

// Kernel to create the world and camera on device
__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState *local_rand_state = rand_state; // For random number generation in world creation if needed
        
        // Create materials
        auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
        d_list[0] = new sphere(point3(0,-1000,0), 1000, ground_material);
        
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = curand_uniform(local_rand_state);
                point3 center(a + 0.9*curand_uniform(local_rand_state), 0.2, b + 0.9*curand_uniform(local_rand_state));
                
                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    material *sphere_material;
                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = color::random(local_rand_state) * color::random(local_rand_state);
                        sphere_material = new lambertian(albedo);
                        d_list[i++] = new sphere(center, 0.2, sphere_material);
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = color::random(0.5, 1, local_rand_state);
                        auto fuzz = curand_uniform(local_rand_state) * 0.5;
                        sphere_material = new metal(albedo, fuzz);
                        d_list[i++] = new sphere(center, 0.2, sphere_material);
                    } else {
                        // glass
                        sphere_material = new dielectric(1.5);
                        d_list[i++] = new sphere(center, 0.2, sphere_material);
                    }
                }
            }
        }
        
        auto material1 = new dielectric(1.5);
        d_list[i++] = new sphere(point3(0, 1, 0), 1.0, material1);
        
        auto material2 = new lambertian(color(0.4, 0.2, 0.1));
        d_list[i++] = new sphere(point3(-4, 1, 0), 1.0, material2);
        
        auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
        d_list[i++] = new sphere(point3(4, 1, 0), 1.0, material3);
        
        *d_world = new hittable_list(d_list, i);
        
        // Create camera
        *d_camera = new camera();
        (*d_camera)->aspect_ratio = 16.0 / 9.0;
        (*d_camera)->image_width = nx;
        (*d_camera)->samples_per_pixel = 10; // Lower samples for GPU demo speed, or keep high?
        (*d_camera)->max_depth = 50;
        (*d_camera)->vfov = 20;
        (*d_camera)->lookfrom = point3(13,2,3);
        (*d_camera)->lookat = point3(0,0,0);
        (*d_camera)->vup = vec3(0,1,0);
        (*d_camera)->defocus_angle = 0.6;
        (*d_camera)->focus_dist = 10.0;
        (*d_camera)->initialize();
    }
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera) {
    // TODO: Free individual objects if needed, but for now just relying on OS cleanup or simple reset
    // In a real app, we should track count and delete loop.
    // Since we allocated d_list[i] we should delete them.
    // But we don't have the count here easily unless stored in d_world.
    // For this simple demo, we might leak device memory if we run multiple times, but for one run it's fine.
    delete *d_world;
    delete *d_camera;
}

__global__ void render_kernel(vec3 *fb, int max_x, int max_y, int ns, int max_depth, camera **d_camera, hittable **d_world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    
    color col(0,0,0);
    for(int s=0; s < ns; s++) {
        ray r = (*d_camera)->get_ray(i, j, &local_rand_state);
        col += (*d_camera)->ray_color(r, max_depth, **d_world, &local_rand_state);
    }
    
    // Normalize and gamma correct
    auto r = col.x();
    auto g = col.y();
    auto b = col.z();
    
    double scale = 1.0 / ns;
    r *= scale;
    g *= scale;
    b *= scale;
    
    r = std::sqrt(r);
    g = std::sqrt(g);
    b = std::sqrt(b);
    
    fb[pixel_index] = vec3(r, g, b);
    
    // Update state
    rand_state[pixel_index] = local_rand_state;
}

int main() {
    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int ns = 10;
    const int max_depth = 50;
    const int tx = 8;
    const int ty = 8;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Increase stack size for recursion
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 65536));

    // Allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    
    // Initialize random state
    dim3 blocks(image_width/tx + 1, image_height/ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Create world
    hittable **d_list;
    int num_hittables = 22*22 + 1 + 3; // Roughly
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hittables * sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    
    create_world<<<1, 1>>>(d_list, d_world, d_camera, image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render
    clock_t start, stop;
    start = clock();
    render_kernel<<<blocks, threads>>>(fb, image_width, image_height, ns, max_depth, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j*image_width + i;
            auto r = fb[pixel_index].x();
            auto g = fb[pixel_index].y();
            auto b = fb[pixel_index].z();
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // Clean up
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    
    return 0;
}
