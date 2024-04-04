#include <iostream>

__global__ void copyOnGPU(double *src, double *dest, size_t nx) {
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = start; i < nx; i += stride)
        dest[i] = src[i] + 1;
}

void initOnCPU(double *src, size_t nx) {
    for (size_t i = 0; i < nx; ++i)
        src[i] = 1337.;
}

void checkOnCPU(double *dest, size_t nx) {
    for (size_t i = 0; i < nx; ++i)
        if (1338. != dest[i]) {
            std::cout << "Error in element " << i << " (expected 1338., got " << dest[i] << ")" << std::endl;
            return;
        }
}

int main(int argc, char *argv[]) {
    size_t nx = 32 * 1024 * 1024;
    if (argc > 1)
        nx = atoi(argv[1]);

    double *src, *dest;
    cudaMallocManaged(&src, sizeof(double) * nx);
    cudaMallocManaged(&dest, sizeof(double) * nx);

    initOnCPU(src, nx);

    cudaMemPrefetchAsync(src, sizeof(double) * nx, 0);
    cudaMemPrefetchAsync(dest, sizeof(double) * nx, 0);

    copyOnGPU<<<(nx + 255) / 256, 256>>>(src, dest, nx);
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(dest, sizeof(double) * nx, cudaCpuDeviceId);

    checkOnCPU(dest, nx);

    cudaFree(src);
    cudaFree(dest);
}
