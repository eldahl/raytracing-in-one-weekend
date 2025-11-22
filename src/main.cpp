#include "image_texture.h"
#include "rtweekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include <memory>

int main() {
  hittable_list world;

  std::shared_ptr<material> ground_material = make_shared<lambertian>(color(0.8, 0.5, 0.6));
  world.add(shared_ptr<hittable>(new sphere(point3(0, -1000, 0), 1000, ground_material)));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto choose_mat = random_double();
      point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

      if ((center - point3(4, 0.2, 0)).length() > 0.9) {
        shared_ptr<material> sphere_material;

        if (choose_mat < 0.8) {
          // diffuse
          auto albedo = color::random() * color::random();
          sphere_material = make_shared<lambertian>(albedo);
          world.add(shared_ptr<hittable>(new sphere(center, 0.2, sphere_material)));
        } else if (choose_mat < 0.95) {
          // metal
          auto albedo = color::random(0.5, 1);
          auto fuzz = random_double(0, 0.5);
          sphere_material = make_shared<metal>(albedo, fuzz);
          world.add(shared_ptr<hittable>(new sphere(center, 0.2, sphere_material)));
        } else {
          // glass
          sphere_material = make_shared<dielectric>(1.5);
          world.add(shared_ptr<hittable>(new sphere(center, 0.2, sphere_material)));
        }
      }
    }
  }
  
	std::shared_ptr<material> left1 = make_shared<metal>(color(0.137, 0.922, 0.439), 0.2);
  world.add(shared_ptr<hittable>(new sphere(point3(-(pi/2), 4, -1), 1.0, left1)));
	
	std::shared_ptr<material> left2 = make_shared<metal>(color(0.8, 0.5, 0.8), 0.2);
  world.add(shared_ptr<hittable>(new sphere(point3(-6, 8, -5), 5.0, left2)));
	
	std::shared_ptr<material> right1 = make_shared<metal>(color(0.85, 0.188, 0.188), 0.2);
  world.add(shared_ptr<hittable>(new sphere(point3((pi/2), 4, -1), 1.0, right1)));
	
	std::shared_ptr<material> right2 = make_shared<metal>(color(0.827, 0.686, 0.215), 0.2);
  world.add(shared_ptr<hittable>(new sphere(point3(6, 8, -5), 5.0, right2)));
  
	auto texture = make_shared<image_texture>("../textures/test.png");
	shared_ptr<material> material1 = make_shared<lambertian_texture>(texture);
    shared_ptr<hittable> sphere1(new sphere(point3(0, 5, 0), 1.0, material1));
    world.add(sphere1);
  
	shared_ptr<material> material2 = make_shared<metal>(color(0.137, 0.922, 0.439), 0.2);
    shared_ptr<hittable> sphere2(new sphere(point3(0, 3, 0), 1.0, material2));
    world.add(sphere2);

  shared_ptr<material> material3 = make_shared<dielectric>(1.5);
    shared_ptr<hittable> sphere3(new sphere(point3(0, 1, 0), 1.0, material3));
    world.add(sphere3);
  
	// auto material2 = make_shared<lambertian>(color(0.8, 0.4, 0.8));
	//  world.add(make_shared<sphere>(point3(0, 5, 0), 1.0, material2));


  camera cam;

  cam.aspect_ratio = 16.0 / 9.0;
  cam.image_width = 1200;
  cam.samples_per_pixel = 10;
  cam.max_depth = 50;

  cam.vfov = 50;
  cam.lookfrom = point3(0, 2, 10);
  cam.lookat = point3(0, 2.5, 0);
  cam.vup = vec3(0, 1, 0);

  cam.defocus_angle = 1.2;
  cam.focus_dist = 10.0;
  cam.render(world);
}
