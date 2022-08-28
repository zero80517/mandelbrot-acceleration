#include "cudalibrary.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "kernel.cuh"

namespace CudaLibrary {

int CUDALIBRARY_EXPORT FillMandelbrot(uint *const bits, const int halfWidth, const int halfHeight,
                                      const double scaleFactor, const double centerX, const double centerY,
                                      const int Limit, const int MaxIterations,
                                      const uint *colormap, const uint ColormapSize,
                                      bool *const allBlack)
{
  // Fill Mandelbrot set with CUDA.
  cudaError_t cudaStatus = FillMandelbrotWithCuda(bits, halfWidth, halfHeight,
                                                  scaleFactor, centerX, centerY,
                                                  Limit, MaxIterations,
                                                  colormap, ColormapSize,
                                                  allBlack);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "FillMandelbrotWithCuda failed!");
    return 1;
  }

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
  }

  return 0;
}

} // CudaLibrary

