#ifndef HITABLE_H
#define HITABLE_H

#include "vec3.h"

typedef struct hit_record {
  float t;
  vec3 p;
  vec3 normal;
} hit_record;

typedef enum hitable_type {
  SPHERE,
  HITABLE_LIST
} hitable_type;

typedef struct hitable {
  void *v;
  hitable_type type;
} hitable;

__device__ bool hit(hitable *h, const ray &r, float t_min, float t_max, hit_record &rec);
__host__ void clean_up_hitable(hitable *h);

#endif