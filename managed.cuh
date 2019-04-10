#ifndef MANAGED_H
#define MANAGED_H

// from https://devblogs.nvidia.com/unified-memory-in-cuda-6/
class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

#endif