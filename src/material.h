#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "rtweekend.h"
#include "vec3.h"
#include "texture.h"

class material {
public:
  HOST_DEVICE virtual ~material() = default;

  HOST_DEVICE virtual bool scatter(const ray &r_in, const hit_record &rec,
                       color &attenuation, ray &scattered, RAND_STATE) const {
    return false;
  }
};

class lambertian : public material {
public:
  HOST_DEVICE lambertian(const color &albedo) : albedo(albedo) {}

  HOST_DEVICE bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
               ray &scattered, RAND_STATE) const override {
    auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);

    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
      scatter_direction = rec.normal;

    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
  }

private:
  color albedo;
};

class lambertian_texture : public material {
public:
  HOST_DEVICE lambertian_texture(const color &albedo) : albedo(MAKE_SHARED(solid_color, albedo)) {}
  HOST_DEVICE lambertian_texture(SharedPtr<texture> tex) : albedo(tex) {}

  HOST_DEVICE bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
               ray &scattered, RAND_STATE) const override {
    auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);

    if (scatter_direction.near_zero())
      scatter_direction = rec.normal;

    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
  }

private:
  SharedPtr<texture> albedo;
};

class metal : public material {
public:
  HOST_DEVICE metal(const color &albedo, double fuzz)
      : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

  HOST_DEVICE bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
               ray &scattered, RAND_STATE) const override {
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (fuzz * random_unit_vector(local_rand_state));
    scattered = ray(rec.p, reflected);
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }

private:
  color albedo;
  double fuzz;
};

class dielectric : public material {
public:
  HOST_DEVICE dielectric(double refraction_index) : refraction_index(refraction_index) {}

  HOST_DEVICE bool scatter(const ray &r_in, const hit_record &rec, color &attenuation,
               ray &scattered, RAND_STATE) const override {
    attenuation = color(1.0, 1.0, 1.0);
    double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

    vec3 unit_direction = unit_vector(r_in.direction());
    double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;
    vec3 direction;

    if (cannot_refract || reflectance(cos_theta, ri) > RANDOM_DOUBLE)
      direction = reflect(unit_direction, rec.normal);
    else
      direction = refract(unit_direction, rec.normal, ri);

    scattered = ray(rec.p, direction);
    return true;
  }

private:
  // Refractive index in vacuum or air, or the ratio of the material's
  // refractive index over the refractive index of the enclosing media
  double refraction_index;

  HOST_DEVICE static double reflectance(double cosine, double refraction_index) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::pow((1 - cosine), 5);
  }
};

#endif
