
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "kernel.cuh"

__global__ void FillMandelbrotKernel(uint *const bits,
                                     const double scaleFactor, const double centerX, const double centerY,
                                     const int Limit, const int MaxIterations,
                                     const uint *colormap, const uint ColormapSize,
                                     bool *const allBlack)
{
    int halfWidth = blockDim.x / 2;
    int halfHeight = gridDim.x / 2;
    int y = -halfHeight + blockIdx.x;
    int x = -halfWidth + threadIdx.x;
    uint *scanLine = bits + (y + halfHeight) * (halfWidth * 2) + (x + halfWidth);

    double ay = centerY + (y * scaleFactor);

    double ax = centerX + (x * scaleFactor);
    double a1 = ax;
    double b1 = ay;
    int numIterations = 0;

    do {
        ++numIterations;
        double a2 = (a1 * a1) - (b1 * b1) + ax;
        double b2 = (2 * a1 * b1) + ay;
        if ((a2 * a2) + (b2 * b2) > Limit)
            break;

        ++numIterations;
        a1 = (a2 * a2) - (b2 * b2) + ax;
        b1 = (2 * a2 * b2) + ay;
        if ((a1 * a1) + (b1 * b1) > Limit)
            break;
    } while (numIterations < MaxIterations);

    if (numIterations < MaxIterations) {
        *scanLine = colormap[numIterations % ColormapSize];
        if (*allBlack == true) {
            *allBlack = false;
        }
    } else {
        *scanLine = 0xff000000u;
    }
}

// Helper function for using CUDA.
cudaError_t FillMandelbrotWithCuda(uint *const bits, const int halfWidth, const int halfHeight,
                                   const double scaleFactor, const double centerX, const double centerY,
                                   const int Limit, const int MaxIterations,
                                   const uint *colormap, const uint ColormapSize,
                                   bool *const allBlack)
{
    uint *dev_bits = 0;
    uint *dev_colormap = 0;
    bool *dev_allBlack = 0;
    const int size = (halfWidth * 2) * (halfHeight * 2);
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_bits, size * sizeof(uint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_colormap, ColormapSize * sizeof(uint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_allBlack, 1 * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_colormap, colormap, ColormapSize * sizeof(uint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_allBlack, allBlack, 1 * sizeof(bool), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    FillMandelbrotKernel<<<halfHeight * 2, halfWidth * 2>>>(dev_bits,
                                                            scaleFactor, centerX, centerY,
                                                            Limit, MaxIterations,
                                                            dev_colormap, ColormapSize,
                                                            dev_allBlack);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FillMandelbrotKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching FillMandelbrotKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(bits, dev_bits, size * sizeof(uint), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(bits, dev_allBlack, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_bits);
    cudaFree(dev_colormap);
    cudaFree(dev_allBlack);
    
    return cudaStatus;
}
