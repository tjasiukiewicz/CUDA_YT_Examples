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

	constexpr static size_t NumStreams = 8;

	constexpr static size_t Elements = 57344 * 1024;
	constexpr static size_t MemElements = Elements * sizeof(DataType_t);

	constexpr static size_t ElementsChunk = Elements / NumStreams;
	constexpr static size_t MemChunk = MemElements / NumStreams;

	constexpr static size_t ThreadsInBlock = 1024;
	constexpr static size_t Block = (Elements + ThreadsInBlock - 1) / ThreadsInBlock;

	constexpr static size_t BlockChunk = Block / NumStreams;

	cudaStream_t streams[NumStreams];

	for (auto i = 0U; i < NumStreams; ++i) {
		cudaStreamCreate(&streams[i]);
	}

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

	// Split by streams

	for (auto i = 0U; i < NumStreams; ++i) {
		size_t elementsOffset = ElementsChunk * i;

		cudaMemcpyAsync(&dSource1[elementsOffset], &hSource1[elementsOffset],
				MemChunk, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(&dSource2[elementsOffset], &hSource2[elementsOffset],
				MemChunk, cudaMemcpyHostToDevice, streams[i]);

		vector_add<<<BlockChunk, ThreadsInBlock, 0, streams[i]>>>
			(&dResult[elementsOffset], &dSource1[elementsOffset], &dSource2[elementsOffset]);

		cudaMemcpyAsync(&hResult[elementsOffset], &dResult[elementsOffset],
				MemChunk, cudaMemcpyDeviceToHost, streams[i]);
	}

	// End split by streams

	for (auto i = 0U; i < NumStreams; ++i) {
		cudaStreamDestroy(streams[i]);
	}

	cudaFree(dResult);
	cudaFree(dSource2);
	cudaFree(dSource1);

	cudaDeviceSynchronize();

	verify_add(hResult, hSource1, hSource2, Elements);

	cudaFreeHost(hResult);
	cudaFreeHost(hSource2);
	cudaFreeHost(hSource1);

}
