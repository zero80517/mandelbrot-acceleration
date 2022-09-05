
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "kernel.cuh"
#include "types.h"

uint *dev_bits;
int size = 0;
uint *dev_colormap;
int colormap_size;
bool *dev_allBlack;
int *host_progress;
int *dev_progress;
bool *host_stop;
bool *dev_stop;

__global__ void FillMandelbrotKernel(
    uint *const bits,
    const int halfWidth, const int halfHeight, const double scaleFactor, const double centerX, const double centerY,
    const int Limit, const int MaxIterations,
    const uint *colormap, const uint ColormapSize,
    bool *const allBlack,
    int *const progress,
    bool *const stop
)
{
    int x = -halfWidth + blockDim.x * blockIdx.x + threadIdx.x;
    int y = -halfHeight + blockDim.y * blockIdx.y + threadIdx.y;

    if (x < halfWidth && y < halfHeight) {
        uint *scanLine = bits + (y + halfHeight) * (halfWidth * 2) + (x + halfWidth);

        double ay = centerY + (y * scaleFactor);

        double ax = centerX + (x * scaleFactor);
        double a1 = ax;
        double b1 = ay;
        int numIterations = 0;

        if (!(threadIdx.x || threadIdx.y)) {
            atomicAdd(progress, 1);
            __threadfence_system();
        }

        do {
            if (*stop)
                return;

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
            if (*allBlack == true)
                *allBlack = false;
        } else {
            *scanLine = 0xff000000u;
        }
    }
}

int ceil(int a, int b) noexcept
{
    return a / b + ((a % b) != 0);
}

// Helper function for using CUDA.
void FillMandelbrotWithCuda(uint *const bits, void *params)
{
    Params *p = (Params *)params;

    const int &halfWidth = p->halfWidth;
    const int &halfHeight = p->halfHeight;
    const double &scaleFactor = p->scaleFactor;
    const double &centerX = p->centerX;
    const double &centerY = p->centerY;
    const int &Limit = p->Limit;
    const int &MaxIterations = p->MaxIterations;
    bool *const &allBlack = p->allBlack;
    bool *const &restart = p->restart;
    bool *const &abort = p->abort;

    // copy from host memory to to GPU buffer
    gpuErrchk(cudaMemcpy(dev_allBlack, allBlack, 1 * sizeof(bool), cudaMemcpyHostToDevice));
    *host_progress = 0;
    *host_stop = *restart || *abort;

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start)); gpuErrchk(cudaEventCreate(&stop));
    printf("FillMandelbrotWithCuda starting\n");
    gpuErrchk(cudaEventRecord(start));
    const int num_thread_y = 32;
    const int num_thread_x = 32;
    const int num_block_y = ceil(2 * halfHeight, num_thread_y);
    const int num_block_x = ceil(2 * halfWidth, num_thread_x);
    dim3 block(num_thread_x, num_thread_y);
    dim3 grid(num_block_x, num_block_y);
    FillMandelbrotKernel<<<grid, block>>>(
        dev_bits,
        halfWidth, halfHeight, scaleFactor, centerX, centerY,
        Limit, MaxIterations,
        dev_colormap, colormap_size,
        dev_allBlack,
        dev_progress,
        dev_stop
    );
    gpuErrchk(cudaEventRecord(stop));
    unsigned int num_blocks = num_block_x * num_block_y;
    float my_progress = 0.0f;
    printf("Progress:\n"); fflush(stdout);
    do {
        if (*restart || *abort) {
            *host_stop = true;
            printf("STOP!\n"); fflush(stdout);
            gpuErrchk(cudaDeviceReset()); // free all memory, destroy all events
            size = 0;
            return;
        }
        cudaEventQuery(stop); // may help WDDM scenario
        int progress = *host_progress;
        float kern_progress = (float)progress/(float)num_blocks;
        if ((kern_progress - my_progress)> 0.1f) {
            printf("percent complete = %2.1f\n", (kern_progress*100)); fflush(stdout);
            my_progress = kern_progress;
        }
    } while (my_progress < 0.9f);
    printf("\n"); fflush(stdout);
    gpuErrchk(cudaEventSynchronize(stop));

    // Check for any errors launching the kernel
    gpuErrchk(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    gpuErrchk(cudaDeviceSynchronize());

    // Copy output from GPU buffer to host memory.
    gpuErrchk(cudaMemcpy(bits, dev_bits, size * sizeof(uint), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(allBlack, dev_allBlack, 1 * sizeof(bool), cudaMemcpyDeviceToHost));

    // destroy all events
    gpuErrchk(cudaEventDestroy(start)); gpuErrchk(cudaEventDestroy(stop));
}
