#include "cudalibrary.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "kernel.cuh"
#include "windows.h"

namespace CudaLibrary {

void CUDALIBRARY_EXPORT InitMandelbrot(const int halfWidth, const int halfHeight,
                                      const uint *colormap, const uint ColormapSize)
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    gpuErrchk(cudaSetDevice(0));

    colormap_size = ColormapSize;
    if (size == 0) {
        size = (halfWidth * 2) * (halfHeight * 2);
        goto alloc;
    } else if (size != (halfWidth * 2) * (halfHeight * 2)) {
        size = (halfWidth * 2) * (halfHeight * 2);
        goto dealloc;
    } else {
        return;
    }

dealloc:
    gpuErrchk(cudaFree(dev_bits));
    gpuErrchk(cudaFree(dev_colormap));
    gpuErrchk(cudaFree(dev_allBlack));
    gpuErrchk(cudaFreeHost(host_progress));
    gpuErrchk(cudaFreeHost(host_stop));

alloc:
    gpuErrchk(cudaSetDeviceFlags(cudaDeviceMapHost));
    // Allocate GPU buffers.
    gpuErrchk(cudaMalloc((void**)&dev_bits, size * sizeof(uint)));
    gpuErrchk(cudaMalloc((void**)&dev_colormap, colormap_size * sizeof(uint)));
    gpuErrchk(cudaMalloc((void**)&dev_allBlack, 1 * sizeof(bool)));
    gpuErrchk(cudaHostAlloc((void **)&host_progress, sizeof(int), cudaHostAllocMapped));
    gpuErrchk(cudaHostAlloc((void **)&host_stop, sizeof(bool), cudaHostAllocMapped));

    // Copy input vectors from host memory to GPU buffers.
    gpuErrchk(cudaMemcpy((void*)dev_colormap, (void*)colormap, colormap_size * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrchk(cudaHostGetDevicePointer((int **)&dev_progress, (int *)host_progress, 0));
    gpuErrchk(cudaHostGetDevicePointer((bool **)&dev_stop, (bool *)host_stop, 0));
}

void CUDALIBRARY_EXPORT FillMandelbrot(uint *const bits, void *params)
{
    // Fill Mandelbrot set with CUDA.
    FillMandelbrotWithCuda(bits, params);
}

} // CudaLibrary

