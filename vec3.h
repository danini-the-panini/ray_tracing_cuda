#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

typedef union {
    struct {
        float x, y, z;
    };
    struct {
        float r, g, b;
    };
    float e[3];
} vec3;

__device__ vec3 make_vec3(float x, float y, float z) {
    vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

__device__ float dot(const vec3 &v1, const vec3 &v2) {
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

__device__ float length_sq(const vec3 &v) {
    return dot(v, v);
}

__device__ float length(const vec3 &v) {
    return sqrt(length_sq(v));
}





__device__ void negate(vec3 &v) {
    v.x = -v.x;
    v.y = -v.y;
    v.z = -v.z;
}

__device__ void add(vec3 &v1, const vec3 &v2) {
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
}

__device__ void sub(vec3 &v1, const vec3 &v2) {
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z;
}

__device__ void mul(vec3 &v1, const vec3 &v2) {
    v1.x *= v2.x;
    v1.y *= v2.y;
    v1.z *= v2.z;
}

__device__ void div(vec3 &v1, const vec3 &v2) {
    v1.x /= v2.x;
    v1.y /= v2.y;
    v1.z /= v2.z;
}

__device__ void mul(vec3 &v1, float t) {
    v1.x *= t;
    v1.y *= t;
    v1.z *= t;
}

__device__ void div(vec3 &v1, float t) {
    v1.x /= t;
    v1.y /= t;
    v1.z /= t;
}



__device__ vec3 negative(const vec3 &v) {
    return make_vec3(-v.x, -v.y, -v.z);
}

__device__ vec3 plus(const vec3 &v1, const vec3 &v2) {
    return make_vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ vec3 minus(const vec3 &v1, const vec3 &v2) {
    return make_vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__device__ vec3 times(const vec3 &v1, const vec3 &v2) {
    return make_vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

__device__ vec3 divided_by(const vec3 &v1, const vec3 &v2) {
    return make_vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

__device__ vec3 times(const vec3 &v1, float t) {
    return make_vec3(v1.x * t, v1.y * t, v1.z * t);
}

__device__ vec3 divided_by(const vec3 &v1, float t) {
    return make_vec3(v1.x / t, v1.y / t, v1.z / t);
}

__device__ vec3 times(float t, const vec3 &v1) {
    return make_vec3(v1.x * t, v1.y * t, v1.z * t);
}

__device__ vec3 divided_by(float t, const vec3 &v1) {
    return make_vec3(v1.x / t, v1.y / t, v1.z / t);
}

__device__ vec3 cross(const vec3 &v1, const vec3 &v2) {
    return make_vec3( (v1.y*v2.z - v1.z*v2.y),
                     -(v1.x*v2.z - v1.z*v2.x),
                      (v1.x*v2.y - v1.y*v2.x));
}

__device__ vec3 normal(const vec3 &v) {
    float l = length(v);
    return make_vec3(v.x/l, v.x/l, v.x/l);
}



__device__ void make_unit_vector(vec3 &v) {
    div(v, length(v));
}

__device__ vec3 unit_vector(const vec3 &v) {
    return divided_by(v, length(v));
}

#endif