#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "vec3.h"
#include "ray.h"

#define NX 200
#define NY 100

__device__ bool hit_sphere(const vec3 &center, float radius, const ray &r) {
    vec3 oc = minus(r.origin, center);
    float a = dot(r.direction, r.direction);
    float b = 2.0 * dot(oc, r.direction);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;
    return (discriminant > 0);
}

__device__ vec3 color(const ray &r) {
    if (hit_sphere(make_vec3(0,0,-1), 0.5, r)) {
        return make_vec3(1,0,0);
    }
    vec3 unit_direction = unit_vector(r.direction);
    float t = 0.5 * unit_direction.y + 1.0;
    return plus(
        times(1.0-t, make_vec3(1.0, 1.0, 1.0)),
        times(    t, make_vec3(0.5, 0.7, 1.0))
    );
}

__global__ void kernel(int nx, int ny, unsigned char *out) {
    vec3 lower_left_corner = make_vec3(-2.0, -1.0, -1.0);
    vec3 horizontal = make_vec3(4.0, 0.0, 0.0);
    vec3 vertical = make_vec3(0.0, 2.0, 0.0);
    vec3 origin = make_vec3(0.0, 0.0, 0.0);

    int n = blockIdx.x;
    int j = ny-n/nx-1;
    int i = n%nx;

    float u = float(i) / float(nx);
    float v = float(j) / float(ny);

    ray r = make_ray(origin, plus(lower_left_corner, plus(times(u, horizontal), times(v, vertical))));
    vec3 col = color(r);

    out[n*3+0] = int(255.99*col.r);
    out[n*3+1] = int(255.99*col.g);
    out[n*3+2] = int(255.99*col.b);
}

int main(void) {
    size_t BUFFER_SIZE = sizeof(unsigned char)*NX*NY*3;

    printf("P3\n%d %d\n255\n", NX, NY);
    
    unsigned char *out = (unsigned char*)malloc(BUFFER_SIZE); // host ouput
    unsigned char *d_out; // device output
    cudaMalloc(&d_out, BUFFER_SIZE);

    kernel<<<NX*NY,1>>>(NX, NY, d_out);

    cudaMemcpy(out, d_out, BUFFER_SIZE, cudaMemcpyDeviceToHost);

    for (int n = 0; n < NX*NY; n++) {
        printf("%d %d %d\n", out[n*3+0], out[n*3+1], out[n*3+2]);
    }

    cudaFree(d_out);
    free(out);

    return 0;
}