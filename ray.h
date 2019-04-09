#ifndef RAY_H
#define RAY_H

#include "vec3.h"

typedef struct ray {
    vec3 origin;
    vec3 direction;
} ray;

__device__ ray make_ray(const vec3 &a, const vec3 &b) {
    ray r;
    r.origin = a;
    r.direction = b;
    return r;
}

__device__ vec3 point_at_parameter(ray r, float t) {
    return plus(r.origin, times(t, r.direction));
}

#endif