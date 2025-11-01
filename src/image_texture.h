#pragma once
#include "texture.h"
#include "rtweekend.h"
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class image_texture : public texture {
public:
    image_texture()
        : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

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

    ~image_texture() override {
        if (data)
            stbi_image_free(data);
    }

    color value(double u, double v, const point3& p) const override {
        // Return cyan if no texture data
        if (data == nullptr)
            return color(0, 1, 1);

        // Clamp input texture coordinates to [0,1] × [1,0]
        u = std::clamp(u, 0.0, 1.0);
        v = 1.0 - std::clamp(v, 0.0, 1.0); // flip V coordinate

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

