#ifndef SPHERE_LIST_H
#define SPHERE_LIST_H

#include "sphere.cuh"

typedef struct sphere_list {
  sphere **list;
  int size;
} sphere_list;

__device__ bool hit(const sphere_list *l, const ray &r, float t_min, float t_max, hit_record &rec) {
  hit_record temp_rec;
  bool hit_anything = false;
  double closest_so_far = t_max;
  for (int i = 0; i < l->size; i++) {
    if (hit(l->list[i], r, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }
  return hit_anything;
}

__host__ sphere_list *make_shared_sphere_list(int size) {
  sphere **list;
  cudaMallocManaged(&list, sizeof(sphere *) * size);

  sphere_list *l;
  cudaMallocManaged(&l, sizeof(sphere_list));
  l->list = list;
  l->size = size;

  return l;
}

__host__ void clean_up_sphere_list(sphere_list *l) {
  for (int i = 0; i < l->size; i++) {
    clean_up_sphere(l->list[i]);
  }
  cudaFree(l->list);
  cudaFree(l);
}

#endif