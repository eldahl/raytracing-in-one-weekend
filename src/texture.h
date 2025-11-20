#pragma once

#include "rtweekend.h"

class texture {
public:
  HOST_DEVICE virtual ~texture() = default;

  HOST_DEVICE virtual color value(double u, double v, const point3 &p) const = 0;
};

class solid_color : public texture {
public:
  HOST_DEVICE solid_color() {}
  HOST_DEVICE solid_color(color c) : color_value(c) {}
  HOST_DEVICE solid_color(double r, double g, double b) : solid_color(color(r, g, b)) {}

  HOST_DEVICE color value(double u, double v, const point3 &p) const override {
    return color_value;
  }

private:
  color color_value;
};
