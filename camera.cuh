#ifndef CAMERA_H
#define CAMERA_H

#include <math.h>

#include "managed.cuh"
#include "ray.cuh"

__device__ vec3 random_in_unit_disk(curandState &local_state) {
  return vec3(
    curand_normal(&local_state),
    curand_normal(&local_state),
    0
  );
}

class camera : public Managed {
public:
  camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) {
    lens_radius = aperture/2;
    float theta = vfov*PI/180;
    float half_height = tan(theta/2);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);
    lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
    horizontal = 2*half_width*focus_dist*u;
    vertical = 2*half_height*focus_dist*v;
  }
  camera(const camera &c) {
    origin = c.origin;
    lower_left_corner = c.lower_left_corner;
    horizontal = c.horizontal;
    vertical = c.vertical;
    u = c.u; v = c.v; w = c.w;
    lens_radius = c.lens_radius;
  }

  __device__ ray get_ray(curandState &local_state, float s, float t) const {
    vec3 rd = lens_radius*random_in_unit_disk(local_state);
    vec3 offset = u * rd.x() + v * rd.y();
    vec3 offset_origin = origin + offset;
    return ray(offset_origin, lower_left_corner + s*horizontal + t*vertical - offset_origin);
  }

  vec3 origin;
  vec3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  vec3 u, v, w;
  float lens_radius;
};

#endif