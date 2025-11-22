#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "rtweekend.h"
#include <memory>

class sphere : public hittable {
public:
  HOST_DEVICE sphere(const point3 &center, double radius, SharedPtr<material> mat)
      : center(center), radius(fmax(0.0, radius)), mat(mat) {}

  HOST_DEVICE bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = h * h - a * c;
    if (discriminant < 0)
      return false;

    auto sqrtd = std::sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
      root = (h + sqrtd) / a;
      if (!ray_t.surrounds(root))
        return false;
    }


    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat = mat;
		
    // compute u,v from the point on the unit sphere
    get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);

    return true;
  }

private:
  point3 center;
  double radius;
  SharedPtr<material> mat;
};

#endif