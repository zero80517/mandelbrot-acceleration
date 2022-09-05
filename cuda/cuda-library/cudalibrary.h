#ifndef CUDALIBRARY_H
#define CUDALIBRARY_H

#include "cudalibrary_global.h"

namespace CudaLibrary
{

#ifdef __cplusplus
extern "C++" {  // only need to export C++ interface
#endif

void CUDALIBRARY_EXPORT InitMandelbrot(const int halfWidth, const int halfHeight,
                                      const uint *colormap, const uint ColormapSize);
void CUDALIBRARY_EXPORT FillMandelbrot(uint *const bits, void *params);

#ifdef __cplusplus
}
#endif

} // CUDALIBRARY_EXPORT

#endif // CUDALIBRARY_H
