#include <cuda_runtime.h>
#include <cstdio>

__global__ void myindex() {
	auto const tidx = threadIdx.x;	// Indeks wątku w osi x w ramach 1 bloku.
	auto const tidy = threadIdx.y;	// Indeks wątku w osi y w ramach 1 bloku.
	auto const bidxx = blockIdx.x;	// Indeks bloku w osi x w gridzie 
	auto const bidxy = blockIdx.y;	// Indeks bloku w osi y w gridzie

	auto const row = blockDim.y * blockIdx.y + threadIdx.y;	// Indeks w osi y wątku czyli jego wiersz
	auto const col = blockDim.x * blockIdx.x + threadIdx.x;	// Indeks w osi x wątku czyli jego kolumna

	auto const fidx = row * blockDim.x * gridDim.x + col;	// "Płaski indeks", wiersz po wierszu... 

	printf("bidxx: %d, bidxy: %d, tidx: %d, tidy: %d, row: %d, col: %d, fidx: %d\n",
			bidxx, bidxy, tidx, tidy, row, col, fidx);
}

int main() {
	dim3 threads_in_block = {2, 2};
	dim3 blocks = {2, 2};

	/*
		blocks{2, 2} 
		+---------+ +---------+
		| +--+--+ | | +--+--+ |
		| | 0| 1| | | | 2| 3| | 
		| +--+--+ | | +--+--+ |
		| | 4| 5| | | | 6| 7| |
		| +--+--+ | | +--+--+ |
		+---------+ +---------+             threads_in_block = {2, 2}
		+---------+ +---------+.           +---------+
		| +--+--+ | | +--+--+ | .......... | +--+--+ |
		| | 8| 9| | | |10|11| |            | |10|11| |
		| +--+--+ | | +--+--+ |            | +--+--+ |
		| |12|13| | | |14|15| |            | |14|15| |
		| +--+--+ | | +--+--+ | .......... | +--+--+ |
		+---------+ +---------+.           +---------+

		gridDim  {2, 2}

	*/

	myindex<<<blocks, threads_in_block>>>();

	cudaDeviceSynchronize();
}
