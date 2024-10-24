#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <numeric>

// Algorytm Hillis/Steele inclusive scan na 1 bloku
template<typename T, typename Func>
__global__ void hills_steele_scan(T * const result, const T * const source, Func func) {

	assert(gridDim.x == 1);

	auto const tid = threadIdx.x;

	result[tid] = source[tid];
	__syncthreads();

	for (auto stride = 1U; stride < blockDim.x; stride <<= 1) {
		if (tid >= stride) {
			result[tid] = func(result[tid], result[tid - stride]);
		}
		__syncthreads();
	}
}

template<typename T>
void showData(const T * const data, const size_t size) {
	// Dla prostoty przykładu, pozostawiam jedynie format %d dla obsługiwanych danych
	std::for_each(data, data + size, [](T a) { printf("%3d ", a); });
	putchar('\n');
}

int main() {
	using DataType_t = int;

	constexpr static size_t Elements = 16;
	constexpr static size_t MemElements = Elements * sizeof(DataType_t);
	constexpr static size_t ThreadsInBlock = 16;
	constexpr static size_t Block = (Elements + ThreadsInBlock - 1) / ThreadsInBlock;

	auto hSource = static_cast<DataType_t *>(malloc(MemElements));
	auto hDestination = static_cast<DataType_t *>(malloc(MemElements));

	DataType_t * dSource = nullptr;
	DataType_t * dDestination = nullptr;

	cudaMalloc(&dSource, MemElements);
	cudaMalloc(&dDestination, MemElements);

	std::iota(hSource, hSource + Elements, 1);

	showData(hSource, Elements);

	cudaMemcpy(dSource, hSource, MemElements, cudaMemcpyHostToDevice);

	hills_steele_scan<<<Block, ThreadsInBlock>>>(dDestination, dSource, 
			[] __device__ (auto a, auto b) { return a + b; });

	cudaMemcpy(hDestination, dDestination, MemElements, cudaMemcpyDeviceToHost);

	showData(hDestination, Elements);

	cudaFree(dDestination);
	cudaFree(dSource);

	free(hDestination);
	free(hSource);

	cudaDeviceSynchronize();
}
