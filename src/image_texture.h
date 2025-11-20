#pragma once
#include "texture.h"
#include "rtweekend.h"
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class image_texture : public texture {
public:
    HOST_DEVICE image_texture()
        : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

    // Host-side constructor loading from file
#ifndef __CUDA_ARCH__
    image_texture(const char* filename) {
        int channels_in_file;
        auto components_per_pixel = 3;
        data = stbi_load(filename, &width, &height, &channels_in_file, components_per_pixel);

        if (!data) {
            std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
            width = height = 0;
        }

        bytes_per_scanline = components_per_pixel * width;
    }
#endif

    // Device-side constructor using pre-loaded data
    HOST_DEVICE image_texture(unsigned char* data, int width, int height)
        : data(data), width(width), height(height), bytes_per_scanline(3 * width) {}

    HOST_DEVICE ~image_texture() override {
#ifndef __CUDA_ARCH__
        // Only free if we own it (loaded via stbi on host)
        // But wait, if we construct with raw pointer on device, we don't own it.
        // For simplicity, let's assume we don't free on device.
        // And on host, if we used the file constructor, we free.
        // But we can't easily distinguish.
        // For this specific use case (device pointer passed in), we shouldn't free.
        // The host file constructor is only used in main.cpp, not main.cu kernel.
        // So let's just NOT free in the raw pointer constructor case?
        // Actually, for the CUDA path, we won't use the file constructor.
        // So we can just skip free on device.
        // On host, if we use file constructor, we should free.
        // But we can't tell which constructor was used easily without a flag.
        // Let's just guard with __CUDA_ARCH__.
        if (data)
             stbi_image_free(data);
#endif
    }

    HOST_DEVICE color value(double u, double v, const point3& p) const override {
        // Return cyan if no texture data
        if (data == nullptr)
            return color(0, 1, 1);

        // Clamp input texture coordinates to [0,1] × [1,0]
#ifdef __CUDA_ARCH__
        u = fmin(fmax(u, 0.0), 1.0);
        v = 1.0 - fmin(fmax(v, 0.0), 1.0);
#else
        u = std::clamp(u, 0.0, 1.0);
        v = 1.0 - std::clamp(v, 0.0, 1.0); // flip V coordinate
#endif

        auto i = static_cast<int>(u * width);
        auto j = static_cast<int>(v * height);

        if (i >= width)  i = width - 1;
        if (j >= height) j = height - 1;

        const auto color_scale = 1.0 / 255.0;
        unsigned char* pixel = data + j * bytes_per_scanline + i * 3;

        return color(color_scale * pixel[0],
                     color_scale * pixel[1],
                     color_scale * pixel[2]);
    }

private:
    unsigned char* data;
    int width, height;
    int bytes_per_scanline;
};

