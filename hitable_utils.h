#ifndef HITABLE_UTILS_H
#define HITABLE_UTILS_H

#include "sphere.h"
#include "hitable_list.h"

__host__ hitable *make_shared_sphere(const vec3 &center, float radius) {
  sphere *s;
  cudaMallocManaged(&s, sizeof(sphere));
  s->center = center;
  s->radius = radius;

  hitable *h;
  cudaMallocManaged(&h, sizeof(hitable));
  h->type = SPHERE;
  h->v = (void*)s;

  return h;
}

__host__ hitable *make_shared_hitable_list(int size) {
  hitable **list;
  cudaMallocManaged(&list, sizeof(hitable *) * size);

  hitable_list *l;
  cudaMallocManaged(&l, sizeof(hitable_list));
  l->list = list;
  l->size = size;

  hitable *h;
  cudaMallocManaged(&h, sizeof(hitable));
  h->type = HITABLE_LIST;
  h->v = (void*)l;

  return h;
}

__host__ void clean_up_hitable_list(hitable_list *l) {
  for (int i = 0; i < l->size; i++) {
    clean_up_hitable(l->list[i]);
  }
  cudaFree(l->list);
  cudaFree(l);
}

__host__ void clean_up_hitable(hitable *h) {
  switch (h->type) {
    case HITABLE_LIST:
    clean_up_hitable_list((hitable_list*)h->v);
    default:
    cudaFree(h->v);
    break;
  }
  cudaFree(h);
}

__device__ bool hit(hitable *h, const ray &r, float t_min, float t_max, hit_record &rec) {
  switch (h->type) {
    case SPHERE:
    return hit(((sphere*)h->v), r, t_min, t_max, rec);
    case HITABLE_LIST:
    return hit(((hitable_list*)h->v), r, t_min, t_max, rec);
    default:
    return false;
  }
}

#endif