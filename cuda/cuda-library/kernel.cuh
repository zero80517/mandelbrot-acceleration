#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "types.h"

extern uint *dev_bits;
extern int size;
extern uint *dev_colormap;
extern int colormap_size;
extern bool *dev_allBlack;
extern int *host_progress;
extern int *dev_progress;
extern bool *host_stop;
extern bool *dev_stop;

/**
 * @brief Wrapper to check CUDA runtime errors
 * @see <a href="https://stackoverflow.com/a/14038590">stackoverflow</a>
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void FillMandelbrotWithCuda(uint *const bits, void *params);
