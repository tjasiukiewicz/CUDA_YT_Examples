#include <cuda_runtime.h>
#include <cstdio>

__global__ void hello() {
	printf("Hello World form CUDA!\n");
}

int main() {
	hello<<<1, 4>>>();
	cudaDeviceSynchronize();
}
