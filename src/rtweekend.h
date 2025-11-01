#pragma once
#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

// C++ Std Usings
using std::make_shared;
using std::shared_ptr;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
  return degrees * pi / 180.0;
}
inline double radians_to_degrees(double radians) {
  return radians * 180.0 / pi;
}
inline double random_double() {
  // Returns a random real in [0,1).
  return std::rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
  // Returns a random real in [min,max).
  return min + (max - min) * random_double();
}


// Common Headers
#include "vec3.h"
#include "color.h"
#include "interval.h"
#include "ray.h"

// Convert point on unit sphere to spherical coordinates (u,v):
// u in [0,1] across longitude (phi), v in [0,1] across latitude (theta)
inline void get_sphere_uv(const point3& p, double& u, double& v) {
    // p is assumed to be a point on the unit sphere centered at origin.
    // theta: angle from -y to +y (like latitude)
    // phi: angle around y axis from +x
    double theta = acos(-p.y());
    double phi   = atan2(-p.z(), p.x()) + pi;

    u = phi / (2*pi);
    v = theta / pi;
}

#endif
