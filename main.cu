#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cub/cub.cuh>

#define PI 3.14159265358979323846

#include "sphere_list.cuh"
#include "camera.cuh"
#include "drand48.h"

#define NX 1200
#define NY 800
#define NS 100
#define MAX_DEPTH 50

// #define NO_OUTPUT

__device__ vec3 sky(const ray &r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5 * unit_direction.y() + 1.0;
    return (1.0-t) * vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__device__ vec3 color(curandState &local_state, const ray &r, sphere_list *world) {
    hit_record rec;
    ray next_ray = r;
    ray scattered;
    vec3 attenuation[MAX_DEPTH];
    int num_hits = 0;
    vec3 col = vec3(0.0, 0.0, 0.0);

    for (int i = 0; i < MAX_DEPTH; i++) {
        if (world->hit(next_ray, 0.001, FLT_MAX, rec)) {
            if (rec.mat_ptr->scatter(local_state, next_ray, rec, attenuation[i], scattered)) {
                next_ray = scattered;
                num_hits++;
            } else {
                break;
            }
        } else {
            col = sky(scattered);
            break;
        }
    }

    for (int i = num_hits-1; i >= 0; i--) {
        col *= attenuation[i];
    }

    return col;
}

__global__ void setup_kernel(curandState * state, unsigned long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init((seed<<20)+id, 0, 0, &state[id]);
}

__device__ vec3 add_vec3(const vec3 &v1, const vec3 &v2) {
    return v1 + v2;
}

__global__ void kernel(curandState* global_state, int nx, int ny, sphere_list *world, const camera &cam, unsigned char *out) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState local_state = global_state[id];

#if NS > 1
    typedef cub::BlockReduce<vec3, NS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
#endif

    vec3 lower_left_corner = vec3(-2.0, -1.0, -1.0);
    vec3 horizontal = vec3(4.0, 0.0, 0.0);
    vec3 vertical = vec3(0.0, 2.0, 0.0);
    vec3 origin = vec3(0.0, 0.0, 0.0);

    int n = blockIdx.x;
    int j = ny-n/nx-1;
    int i = n%nx;

    float u = (float(i) + curand_uniform(&local_state)) / float(nx);
    float v = (float(j) + curand_uniform(&local_state)) / float(ny);

    ray r = cam.get_ray(local_state, u, v);
    vec3 col = color(local_state, r, world);
    col/=float(NS);

#if NS > 1
    col = BlockReduce(temp_storage).Reduce(col, add_vec3);
#endif

    col = vec3(sqrt(col.r()), sqrt(col.g()), sqrt(col.b()));
    if (threadIdx.x == 0) {
        out[n*3+0] = int(255.99*col.r());
        out[n*3+1] = int(255.99*col.g());
        out[n*3+2] = int(255.99*col.b());
    }
}

__host__ sphere_list *random_scene() {
    int n = 485;
    sphere_list *world = new sphere_list(n);
    world->list[0] =  new sphere(vec3(0,-1000,0), 1000, new material(LAMBERTIAN, vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = drand48();
            vec3 center(a+0.9*drand48(),0.2,b+0.9*drand48()); 
            if ((center-vec3(4,0.2,0)).length() > 0.9) { 
                if (choose_mat < 0.8) {  // diffuse
                    world->list[i++] = new sphere(center, 0.2, new material(LAMBERTIAN, vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));
                }
                else if (choose_mat < 0.95) { // metal
                    world->list[i++] = new sphere(center, 0.2,
                            new material(METAL, vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())),  0.5*drand48()));
                }
                else {  // glass
                    world->list[i++] = new sphere(center, 0.2, new material(DIELECTRIC, 1.5));
                }
            }
        }
    }

    world->list[i++] = new sphere(vec3(0, 1, 0), 1.0, new material(DIELECTRIC, 1.5));
    world->list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new material(LAMBERTIAN, vec3(0.4, 0.2, 0.1)));
    world->list[i++] = new sphere(vec3(4, 1, 0), 1.0, new material(METAL, vec3(0.7, 0.6, 0.5), 0.0));

    return world;
}

int main(void) {
    curandState* device_states;
    cudaMalloc(&device_states, NX*NY*NS*sizeof(curandState));

    setup_kernel<<<NX*NY,NS>>>(device_states, time(NULL));

    size_t BUFFER_SIZE = sizeof(unsigned char)*NX*NY*3;

#ifndef NO_OUTPUT
    printf("P3\n%d %d\n255\n", NX, NY);
#endif

    sphere_list *world = random_scene();
    
    unsigned char *out = (unsigned char*)malloc(BUFFER_SIZE); // host ouput
    unsigned char *d_out; // device output
    cudaMalloc(&d_out, BUFFER_SIZE);

    vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;

    camera *cam = new camera(lookfrom, lookat, vec3(0,1,0), 20, float(NX)/float(NY), aperture, dist_to_focus);

    kernel<<<NX*NY,NS>>>(device_states, NX, NY, world, *cam, d_out);

    cudaMemcpy(out, d_out, BUFFER_SIZE, cudaMemcpyDeviceToHost);

#ifndef NO_OUTPUT
    for (int n = 0; n < NX*NY; n++) {
        printf("%d %d %d\n", out[n*3+0], out[n*3+1], out[n*3+2]);
    }
#endif

    cudaFree(d_out);
    free(out);

    for (int i = 0; i < world->size; i++) {
        delete world->list[i]->mat_ptr;
        delete world->list[i];
    }

    delete world;
    delete cam;

    return 0;
}