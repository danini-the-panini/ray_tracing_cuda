#ifndef CAMERA_H
#define CAMERA_H

#include <math.h>

#include "managed.cuh"
#include "ray.cuh"

class camera : public Managed {
public:
  camera(float vfov, float aspect) {
    float theta = vfov*PI/180;
    float half_height = tan(theta/2);
    float half_width = aspect * half_height;
    lower_left_corner = vec3(-half_width, -half_height, -1.0);
    horizontal = vec3(2*half_width, 0.0, 0.0);
    vertical = vec3(0.0, 2*half_height, 0.0);
    origin = vec3(0.0, 0.0, 0.0);
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