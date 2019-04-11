#ifndef SPHERE_H
#define SPHERE_H

#include "hit_record.cuh"
#include "material.cuh"
#include "managed.cuh"

class sphere : public Managed {
public:
  __host__ sphere(vec3 center, float radius, material *mat_ptr) : center(center), radius(radius), mat_ptr(mat_ptr) {}
  __host__ sphere(const sphere &s) {
    cudaMallocManaged(&mat_ptr, sizeof(material));
    memcpy(mat_ptr, s.mat_ptr, sizeof(material));
  }
  __host__ ~sphere() {
    if (mat_ptr) delete mat_ptr;
  }
  __device__ bool hit(const ray &r, float t_min, float t_max, hit_record &rec);

  vec3 center;
  float radius;
  material *mat_ptr;
};

__device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) {
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - a*c;
  if (discriminant > 0) {
    float temp = (-b - sqrt(b*b-a*c))/a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
    temp = (-b + sqrt(b*b-a*c))/a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
  }
  return false;
}

#endif