#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned int uint;

cudaError_t FillMandelbrotWithCuda(uint *const bits, const int halfWidth, const int halfHeight,
                                   const double scaleFactor, const double centerX, const double centerY,
                                   const int Limit, const int MaxIterations,
                                   const uint *colormap, const uint ColormapSize,
                                   bool *const allBlack);
