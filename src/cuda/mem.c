#include <stdio.h>
#include <cuda.h>

__global__ void IncOne(double *x)
{
    int tid = threadIdx.x;
    x[tid] += 1
}

int main()
{
    int N = 32;

    int nbytes = N * sizeof(double)

    double *dx = NULL;
    double *hx = NULL;

    cudaMalloc((void **) &dx, nbytes);

    if (dx == NULL) {
        printf("could not allocate GPU memeory")
    } 
}