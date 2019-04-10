//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#ifndef MATERIALH
#define MATERIALH 

#include "ray.cuh"
#include "hit_record.cuh"

typedef enum material_type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC
} material_type;

typedef struct material {
    material_type type;
    vec3 albedo;
    union {
        float fuzz;
        float ref_idx;
    };
} material;

__host__ material *make_shared_material() {
    material *m;
    cudaMallocManaged(&m, sizeof(material));
    return m;
}

__host__ material *make_shared_lambertian(const vec3 &albedo) {
    material *m = make_shared_material();

    m->type = LAMBERTIAN;
    m->albedo = albedo;

    return m;
}

__host__ material *make_shared_metal(const vec3 &albedo, float fuzz) {
    material *m = make_shared_material();

    m->type = METAL;
    m->albedo = albedo;
    m->fuzz = fuzz > 1 ? 1 : fuzz;

    return m;
}

__host__ material *make_shared_dielectric(float ref_idx) {
    material *m = make_shared_material();

    m->type = DIELECTRIC;
    m->ref_idx = ref_idx;

    return m;
}

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else 
        return false;
}


__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2*dot(v,n)*n;
}

__device__ vec3 random_in_unit_sphere(curandState &local_state) {
    return vec3(
        curand_normal(&local_state),
        curand_normal(&local_state),
        curand_normal(&local_state)
    );
}


__device__ bool scatter_lambertian(curandState &local_state, const material *m, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) {
    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_state);
    scattered = ray(rec.p, target-rec.p);
    attenuation = m->albedo;
    return true;
}

__device__ bool scatter_metal(curandState &local_state, const material *m, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + m->fuzz*random_in_unit_sphere(local_state));
    attenuation = m->albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}

__device__ bool scatter_dielectric(curandState &local_state, const material *m, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) {
    vec3 outward_normal;
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0); 
    vec3 refracted;
    float reflect_prob;
    float cosine;
    if (dot(r_in.direction(), rec.normal) > 0) {
         outward_normal = -rec.normal;
         ni_over_nt = m->ref_idx;
         cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
         cosine = sqrt(1 - m->ref_idx*m->ref_idx*(1-cosine*cosine));
    }
    else {
         outward_normal = rec.normal;
         ni_over_nt = 1.0 / m->ref_idx;
         cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) 
       reflect_prob = schlick(cosine, m->ref_idx);
    else 
       reflect_prob = 1.0;
    if (curand_uniform(&local_state) < reflect_prob) 
       scattered = ray(rec.p, reflected);
    else 
       scattered = ray(rec.p, refracted);
    return true;
}

__device__ bool scatter(curandState &local_state, const material *m, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) {
    switch (m->type) {
    default:
    case LAMBERTIAN:
        return scatter_lambertian(local_state, m, r_in, rec, attenuation, scattered);
    case METAL:
        return scatter_metal(local_state, m, r_in, rec, attenuation, scattered);
    case DIELECTRIC:
        return scatter_dielectric(local_state, m, r_in, rec, attenuation, scattered);
    }
}

#endif




