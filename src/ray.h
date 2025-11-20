#pragma once
#ifndef RAY_H
#define RAY_H

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#include "rtweekend.h"

class ray {
public:
  HOST_DEVICE ray() {}

  HOST_DEVICE ray(const point3 &origin, const vec3 &direction)
      : orig(origin), dir(direction) {}

  HOST_DEVICE const point3 &origin() const { return orig; }
  HOST_DEVICE const vec3 &direction() const { return dir; }

  HOST_DEVICE point3 at(double t) const { return orig + t * dir; }

private:
  point3 orig;
  vec3 dir;
};

#endif
