#ifndef HITABLE_LIST_H
#define HITABLE_LIST_H

#include "hitable.h"

typedef struct hitable_list {
  hitable **list;
  int size;
} hitable_list;

__device__ bool hit(const hitable_list *l, const ray &r, float t_min, float t_max, hit_record &rec) {
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

#endif