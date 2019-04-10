#ifndef SPHERE_H
#define SPHERE_H

#include "hit_record.cuh"
#include "material.cuh"

typedef struct sphere {
  vec3 center;
  float radius;
  material *mat_ptr;
} sphere;

__device__ bool hit(const sphere *s, const ray &r, float t_min, float t_max, hit_record &rec) {
  vec3 oc = r.origin() - s->center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - s->radius*s->radius;
  float discriminant = b*b - a*c;
  if (discriminant > 0) {
    float temp = (-b - sqrt(b*b-a*c))/a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - s->center) / s->radius;
      rec.mat_ptr = s->mat_ptr;
      return true;
    }
    temp = (-b + sqrt(b*b-a*c))/a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - s->center) / s->radius;
      rec.mat_ptr = s->mat_ptr;
      return true;
    }
  }
  return false;
}

__host__ sphere *make_shared_sphere(const vec3 &center, float radius, material *mat_ptr) {
  sphere *s;
  cudaMallocManaged(&s, sizeof(sphere));
  s->center = center;
  s->radius = radius;
  s->mat_ptr = mat_ptr;

  return s;
}

__host__ void clean_up_sphere(sphere *s) {
  cudaFree(s->mat_ptr);
  cudaFree(s);
}

#endif