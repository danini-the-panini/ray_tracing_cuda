#ifndef CAMERA_H
#define CAMERA_H

#include <math.h>

#include "managed.cuh"
#include "ray.cuh"

class camera : public Managed {
public:
  camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect) {
    vec3 u, v, w;
    float theta = vfov*PI/180;
    float half_height = tan(theta/2);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);
    lower_left_corner = origin - half_width*u - half_height*v - w;
    horizontal = 2*half_width*u;
    vertical = 2*half_height*v;
  }
  camera(const camera &c) {
    origin = c.origin;
    lower_left_corner = c.lower_left_corner;
    horizontal = c.horizontal;
    vertical = c.vertical;
  }

  __device__ ray get_ray(float u, float v) const {
    return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
  }

  vec3 origin;
  vec3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
};

#endif