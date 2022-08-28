#ifndef CUDALIBRARY_H
#define CUDALIBRARY_H

#include "cudalibrary_global.h"

typedef unsigned int uint;

namespace CudaLibrary
{

#ifdef __cplusplus
extern "C++" {  // only need to export C++ interface
#endif

int CUDALIBRARY_EXPORT FillMandelbrot(uint *const bits, const int halfWidth, const int halfHeight,
                                      const double scaleFactor, const double centerX, const double centerY,
                                      const int Limit, const int MaxIterations,
                                      const uint *colormap, const uint ColormapSize,
                                      bool *const allBlack);

#ifdef __cplusplus
}
#endif

} // CUDALIBRARY_EXPORT

#endif // CUDALIBRARY_H
