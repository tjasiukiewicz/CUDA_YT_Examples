#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <numeric>

// Rozwiązanie poprawne dla 1 bloku.
template<typename T, typename Func>
__global__ void reduce(T * const result, T * const source, const T ident, Func func) {
	auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
	auto const tid = threadIdx.x;

	for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			source[idx] = func(source[idx], source[idx + stride]);
		}
		__syncthreads();
	}

	if (tid == 0) {
		*result = ident;
		*result = func(*result, source[tid]);
	}
}

template<typename T>
void showData(const T * const data, const size_t size) {
	// Dla prostoty przykładu, pozostawiam jedynie format %d dla obsługiwanych danych
	std::for_each(data, data + size, [](T a) { printf("%3lu ", a); });
	putchar('\n');
}

int main() {
	using DataType_t = unsigned long;

	constexpr static size_t Elements = 16;
	constexpr static size_t MemElements = Elements * sizeof(DataType_t);
	constexpr static size_t MemResult = sizeof(DataType_t);
	constexpr static size_t ThreadsInBlock = 16;
	constexpr static size_t Block = (Elements + ThreadsInBlock - 1) / ThreadsInBlock;

	auto hSource = static_cast<DataType_t *>(malloc(MemElements));
	auto hResult = static_cast<DataType_t *>(malloc(MemResult));

	DataType_t * dSource = nullptr;
	DataType_t * dResult = nullptr;

	cudaMalloc(&dSource, MemElements);
	cudaMalloc(&dResult, MemResult);

	std::iota(hSource, hSource + Elements, 1);

	showData(hSource, Elements);

	cudaMemcpy(dSource, hSource, MemElements, cudaMemcpyHostToDevice);

	reduce<<<Block, ThreadsInBlock>>>(dResult, dSource, 1UL,
		[] __device__ (auto a, auto b) {
			return a * b;
	});

	cudaMemcpy(hResult, dResult, MemResult, cudaMemcpyDeviceToHost);

	showData(hResult, 1);

	cudaFree(dResult);
	cudaFree(dSource);

	free(hResult);
	free(hSource);

	cudaDeviceSynchronize();
}
