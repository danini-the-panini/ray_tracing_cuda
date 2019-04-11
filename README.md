# ray_tracing_cuda
CUDA implementation of "Ray Tracing in One Weekend"

# requirements

1. [An NVIDIA GPU](https://www.nvidia.com/en-us/shop/geforce) (tested with Pascal and Turing cards)
2. [CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-downloads)
3. [cub 1.8.0](https://github.com/NVlabs/cub)
4. A C++ compiler (tested with gcc and MSVC)

# how to compile and run it

1. Unzip cub somewhere and remember where you put it.
2. Run the following to compile:

```
nvcc main.cu -o main -I/path/to/cub-1.8.0
```

3. and then run it:

```
./main > output.ppm
```

(If you're on Windows I recommend [Irfanview](https://www.irfanview.com/) for viewing PPM images.)

If you get a garbage image then something is funky with the memory. Either there isn't enough available or something else has gone wrong, like a null pointer.
