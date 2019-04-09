#ifndef SPHERE_H
#define SPHERE_H

#include "hitable.h"

typedef struct sphere {
  vec3 center;
  float radius;
} sphere;

__device__ bool hit(const sphere *s, const ray &r, float t_min, float t_max, hit_record &rec) {
  vec3 oc = minus(r.origin, s->center);
  float a = dot(r.direction, r.direction);
  float b = dot(oc, r.direction);
  float c = dot(oc, oc) - s->radius*s->radius;
  float discriminant = b*b - a*c;
  if (discriminant > 0) {
    float temp = (-b - sqrt(b*b-a*c))/a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = point_at_parameter(r, rec.t);
      rec.normal = divided_by(minus(rec.p, s->center), s->radius);
      return true;
    }
    temp = (-b + sqrt(b*b-a*c))/a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = point_at_parameter(r, rec.t);
      rec.normal = divided_by(minus(rec.p, s->center), s->radius);
      return true;
    }
  }
  return false;
}

#endif