#ifndef HIT_RECORD_H
#define HIT_RECORD_H

struct material;

#include "vec3.cuh"

typedef struct hit_record {
  float t;
  vec3 p;
  vec3 normal;
  material *mat_ptr;
} hit_record;

#endif