#pragma once
#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H


#include "hittable.h"
#include "rtweekend.h"

#include <vector>

class hittable_list : public hittable {
public:
#ifdef __CUDACC__
  hittable** objects;
  int list_size;
#else
  std::vector<shared_ptr<hittable>> objects;
#endif

  HOST_DEVICE hittable_list() {}
  
#ifdef __CUDACC__
  HOST_DEVICE hittable_list(hittable** l, int n) { objects = l; list_size = n; }
#else
  hittable_list(shared_ptr<hittable> object) { add(object); }
  void clear() { objects.clear(); }
  void add(shared_ptr<hittable> object) { objects.push_back(object); }
#endif

  HOST_DEVICE bool hit(const ray &r, interval ray_t,
           hit_record &rec) const override {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;

#ifdef __CUDACC__
    for (int i = 0; i < list_size; i++) {
      if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }
#else
    for (const auto &object : objects) {
      if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }
#endif

    return hit_anything;
  }
};

#endif
