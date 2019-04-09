#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "vec3.h"
#include "ray.h"
#include "hitable_utils.h"

#define NX 200
#define NY 100

__device__ vec3 color(const ray &r, hitable *world) {
    hit_record rec;
    if (hit(world, r, 0.001, FLT_MAX, rec)) {
        return times(.5, make_vec3(rec.normal.x+1, rec.normal.y+1, rec.normal.z+1));
    }
    else {
        vec3 unit_direction = unit_vector(r.direction);
        float t = 0.5 * unit_direction.y + 1.0;
        return plus(
            times(1.0-t, make_vec3(1.0, 1.0, 1.0)),
            times(    t, make_vec3(0.5, 0.7, 1.0))
        );
    }
}

__global__ void kernel(int nx, int ny, hitable *world, unsigned char *out) {
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
    vec3 col = color(r, world);

    out[n*3+0] = int(255.99*col.r);
    out[n*3+1] = int(255.99*col.g);
    out[n*3+2] = int(255.99*col.b);
}

int main(void) {
    size_t BUFFER_SIZE = sizeof(unsigned char)*NX*NY*3;

    printf("P3\n%d %d\n255\n", NX, NY);

    hitable *world = make_shared_hitable_list(2);
    hitable **list = ((hitable_list*)world->v)->list;
    list[0] = make_shared_sphere(make_vec3(0,0,-1), 0.5);
    list[1] = make_shared_sphere(make_vec3(0,-100.5,-1), 100);
    
    unsigned char *out = (unsigned char*)malloc(BUFFER_SIZE); // host ouput
    unsigned char *d_out; // device output
    cudaMalloc(&d_out, BUFFER_SIZE);

    kernel<<<NX*NY,1>>>(NX, NY, world, d_out);

    cudaMemcpy(out, d_out, BUFFER_SIZE, cudaMemcpyDeviceToHost);

    for (int n = 0; n < NX*NY; n++) {
        printf("%d %d %d\n", out[n*3+0], out[n*3+1], out[n*3+2]);
    }

    cudaFree(d_out);
    free(out);

    clean_up_hitable(world);

    return 0;
}