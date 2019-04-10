#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cub/cub.cuh>

#include "vec3.h"
#include "ray.h"
#include "hitable_utils.h"

#define NX 200
#define NY 100
#define NS 100
#define MAX_DEPTH 2

__device__ vec3 random_in_unit_sphere(curandState &local_state) {
    return make_vec3(
        curand_normal(&local_state),
        curand_normal(&local_state),
        curand_normal(&local_state)
    );
}

__device__ vec3 color(curandState &local_state, const ray &r, hitable *world, int depth) {
    hit_record rec;
    if (hit(world, r, 0.001, FLT_MAX, rec)) {
        vec3 target = plus(plus(rec.p, rec.normal), random_in_unit_sphere(local_state));
        if (depth < MAX_DEPTH) {
            return times(0.5, color(local_state, make_ray(rec.p, minus(target, rec.p)), world, depth+1));
        } else {
            return make_vec3(0,0,0);
        }
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

__global__ void setup_kernel(curandState * state, unsigned long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init((seed<<20)+id, 0, 0, &state[id]);
}

__global__ void kernel(curandState* global_state, int nx, int ny, hitable *world, unsigned char *out) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState local_state = global_state[id];

    typedef cub::BlockReduce<vec3, NS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    vec3 lower_left_corner = make_vec3(-2.0, -1.0, -1.0);
    vec3 horizontal = make_vec3(4.0, 0.0, 0.0);
    vec3 vertical = make_vec3(0.0, 2.0, 0.0);
    vec3 origin = make_vec3(0.0, 0.0, 0.0);

    int n = blockIdx.x;
    int j = ny-n/nx-1;
    int i = n%nx;

    float u = (float(i) + curand_uniform(&local_state)) / float(nx);
    float v = (float(j) + curand_uniform(&local_state)) / float(ny);

    ray r = make_ray(origin, plus(lower_left_corner, plus(times(u, horizontal), times(v, vertical))));
    vec3 col = color(local_state, r, world, 0);
    div(col, float(NS));

    vec3 final_col = BlockReduce(temp_storage).Reduce(col, plus);

    if (threadIdx.x == 0) {
        out[n*3+0] = int(255.99*final_col.r);
        out[n*3+1] = int(255.99*final_col.g);
        out[n*3+2] = int(255.99*final_col.b);
    }
}

int main(void) {
    curandState* device_states;
    cudaMalloc(&device_states, NX*NY*NS*sizeof(curandState));

    setup_kernel<<<NX*NY,NS>>>(device_states, time(NULL));

    size_t BUFFER_SIZE = sizeof(unsigned char)*NX*NY*3;

    printf("P3\n%d %d\n255\n", NX, NY);

    hitable *world = make_shared_hitable_list(2);
    hitable **list = ((hitable_list*)world->v)->list;
    list[0] = make_shared_sphere(make_vec3(0,0,-1), 0.5);
    list[1] = make_shared_sphere(make_vec3(0,-100.5,-1), 100);
    
    unsigned char *out = (unsigned char*)malloc(BUFFER_SIZE); // host ouput
    unsigned char *d_out; // device output
    cudaMalloc(&d_out, BUFFER_SIZE);

    kernel<<<NX*NY,NS>>>(device_states, NX, NY, world, d_out);

    cudaMemcpy(out, d_out, BUFFER_SIZE, cudaMemcpyDeviceToHost);

    for (int n = 0; n < NX*NY; n++) {
        printf("%d %d %d\n", out[n*3+0], out[n*3+1], out[n*3+2]);
    }

    cudaFree(d_out);
    free(out);

    clean_up_hitable(world);

    return 0;
}