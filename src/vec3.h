#pragma once
#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>
#include "cuda_utils.h"

// Forward declarations
double random_double();
double random_double(double min, double max);
#ifdef __CUDACC__
__device__ double random_double_cuda(curandState *local_rand_state);
__device__ double random_double_cuda(double min, double max, curandState *local_rand_state);
#endif

class vec3 {
public:
  double e[3];

  HOST_DEVICE vec3() : e{0, 0, 0} {}
  HOST_DEVICE vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

  HOST_DEVICE double x() const { return e[0]; }
  HOST_DEVICE double y() const { return e[1]; }
  HOST_DEVICE double z() const { return e[2]; }

  HOST_DEVICE vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
  HOST_DEVICE double operator[](int i) const { return e[i]; }
  HOST_DEVICE double &operator[](int i) { return e[i]; }

  HOST_DEVICE vec3 &operator+=(const vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  HOST_DEVICE vec3 &operator*=(double t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }

  HOST_DEVICE vec3 &operator/=(double t) { return *this *= 1 / t; }

  HOST_DEVICE double length() const { return std::sqrt(length_squared()); }

  HOST_DEVICE double length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }

  HOST_DEVICE bool near_zero() const {
    // Return true if the vector is close to zero in all dimensions.
    auto s = 1e-8;
    return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) &&
           (std::fabs(e[2]) < s);
  }

  static vec3 random() {
    return vec3(random_double(), random_double(), random_double());
  }

  static vec3 random(double min, double max) {
    return vec3(random_double(min, max), random_double(min, max),
                random_double(min, max));
  }

#ifdef __CUDACC__
  __device__ static vec3 random(curandState *local_rand_state) {
    return vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
  }

  __device__ static vec3 random(double min, double max, curandState *local_rand_state) {
    return vec3(random_double_cuda(min, max, local_rand_state), random_double_cuda(min, max, local_rand_state),
                random_double_cuda(min, max, local_rand_state));
  }
#endif
};

// point3 is just an alias for vec3, but useful for geometric clarity in the
// code.
using point3 = vec3;

// Vector Utility Functions

inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
  return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

HOST_DEVICE inline vec3 operator+(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

HOST_DEVICE inline vec3 operator-(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

HOST_DEVICE inline vec3 operator*(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

HOST_DEVICE inline vec3 operator*(double t, const vec3 &v) {
  return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

HOST_DEVICE inline vec3 operator*(const vec3 &v, double t) { return t * v; }

HOST_DEVICE inline vec3 operator/(const vec3 &v, double t) { return (1 / t) * v; }

HOST_DEVICE inline double dot(const vec3 &u, const vec3 &v) {
  return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

HOST_DEVICE inline vec3 cross(const vec3 &u, const vec3 &v) {
  return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
              u.e[2] * v.e[0] - u.e[0] * v.e[2],
              u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

HOST_DEVICE inline vec3 unit_vector(const vec3 &v) { return v / v.length(); }

HOST_DEVICE inline vec3 random_in_unit_disk(RAND_STATE) {
    while (true) {
#ifdef __CUDA_ARCH__
        auto p = vec3::random(-1, 1, local_rand_state);
#else
        auto p = vec3::random(-1, 1);
#endif
        if (p.length_squared() < 1)
            return p;
    }
}

HOST_DEVICE inline vec3 random_unit_vector(RAND_STATE) {
  while (true) {
#ifdef __CUDA_ARCH__
    auto p = vec3::random(-1, 1, local_rand_state);
#else
    auto p = vec3::random(-1, 1);
#endif
    auto lensq = p.length_squared();
    if (1e-160 < lensq && lensq <= 1)
      return p / sqrt(lensq);
  }
}

HOST_DEVICE inline vec3 random_on_hemisphere(const vec3 &normal, RAND_STATE) {
  vec3 on_unit_sphere = random_unit_vector(local_rand_state);
  if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
    return on_unit_sphere;
  else
    return -on_unit_sphere;
}

HOST_DEVICE inline vec3 reflect(const vec3 &v, const vec3 &n) {
  return v - 2 * dot(v, n) * n;
}

HOST_DEVICE inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = std::fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif