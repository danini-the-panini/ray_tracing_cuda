#ifndef SPHERE_LIST_H
#define SPHERE_LIST_H

#include "sphere.cuh"
#include "managed.cuh"

class sphere_list : public Managed {
public:
  __host__ sphere_list(int size) : size(size) {
    cudaMallocManaged(&list, sizeof(sphere*) * size);
  }
  __host__ ~sphere_list() {
    cudaFree(list);
  }

  __device__ bool hit(const ray &r, float t_min, float t_max, hit_record &rec);
  
  sphere **list;
  int size;
};

__device__ bool sphere_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) {
  hit_record temp_rec;
  bool hit_anything = false;
  double closest_so_far = t_max;
  for (int i = 0; i < size; i++) {
    if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }
  return hit_anything;
}

#endif