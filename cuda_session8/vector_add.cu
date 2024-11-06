#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <numeric>

// Add 2 vector values...
template<typename T>
__global__ void vector_add(
	T * const result, const T * const source1, const T * const source2) {

	auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

	result[idx] = source1[idx] + source2[idx];
}

// Verify result on CPU
template<typename T>
void verify_add(const T * const result,
	const T * const source1, const T * const source2, const size_t size) {

	printf("Vector operation verified: ");

	for (auto i = 0UL; i < size; ++i) {
		if (result[i] != (source1[i] + source2[i])) {
			printf("---->BAD on index %lu!!!\n", i);
			return;
		}
	}

	printf("OK\n");
}

int main() {
	using DataType_t = int;

	constexpr static size_t Elements = 57344 * 1024;
	constexpr static size_t MemElements = Elements * sizeof(DataType_t);
	constexpr static size_t ThreadsInBlock = 1024;
	constexpr static size_t Block = (Elements + ThreadsInBlock - 1) / ThreadsInBlock;

	DataType_t * hSource1 = nullptr;
	DataType_t * hSource2 = nullptr;
	DataType_t * hResult = nullptr;

	cudaMallocHost(&hSource1, MemElements);
	cudaMallocHost(&hSource2, MemElements);
	cudaMallocHost(&hResult, MemElements);

	std::iota(hSource1, hSource1 + Elements, 1);
	std::iota(hSource2, hSource2 + Elements, -8192);

	DataType_t * dSource1 = nullptr;
	DataType_t * dSource2 = nullptr;
	DataType_t * dResult = nullptr;

	cudaMalloc(&dSource1, MemElements);
	cudaMalloc(&dSource2, MemElements);
	cudaMalloc(&dResult, MemElements);

	cudaMemcpyAsync(dSource1, hSource1, MemElements, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(dSource2, hSource2, MemElements, cudaMemcpyHostToDevice);

	vector_add<<<Block, ThreadsInBlock>>>(dResult, dSource1, dSource2);

	cudaMemcpyAsync(hResult, dResult, MemElements, cudaMemcpyDeviceToHost);

	cudaFree(dResult);
	cudaFree(dSource2);
	cudaFree(dSource1);

	cudaDeviceSynchronize();

	verify_add(hResult, hSource1, hSource2, Elements);

	cudaFreeHost(hResult);
	cudaFreeHost(hSource2);
	cudaFreeHost(hSource1);

}
