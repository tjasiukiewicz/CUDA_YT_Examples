#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "helper_cuda.cuh"
#include "matrix_helper.hpp"

using DataType_t = float;

constexpr static size_t Width = 4096;
constexpr static size_t Height = 4096;

constexpr static dim3 ThreadsInBlock = {64, 1};

// Kernel: Naive matrix transpose, threads mapped value
template<typename T>
__global__ void matrix_transpose_naive_full_mapped(
		T * const data_dst, const T * const data_src,
		const std::size_t height, const std::size_t width) {
	
	auto const threadsInRow = gridDim.x * blockDim.x;
	auto const threadOffsetX = blockIdx.x * blockDim.x;
	auto const sIdx = threadsInRow * threadIdx.y + threadOffsetX + threadIdx.x;
	auto const elements = height * width;

	if (sIdx < elements) [[likely]] {
		// Transpose, col <-> row
		auto const rowDst = sIdx % width; // <- transposed: col -> row
		auto const colDst = sIdx / width; // <- transpoded: row -> col
		auto const dIdx = rowDst * height + colDst;

		data_dst[dIdx] = data_src[sIdx];
	}
}

int main() {

	constexpr static size_t Elements = Width * Height;
	constexpr static size_t MemElements = Elements * sizeof(DataType_t);
	constexpr static size_t Threads = ThreadsInBlock.x * ThreadsInBlock.y;
	constexpr static size_t Blocks = (Elements + Threads - 1) / Threads;

	DataType_t * hMtrxSrc = static_cast<DataType_t *>(malloc(MemElements));
	DataType_t * hMtrxDst = static_cast<DataType_t *>(malloc(MemElements));
	if ((hMtrxSrc == nullptr) or (hMtrxDst == nullptr)) {
		std::cerr << "Host matrix allocation error.\n";
		exit(EXIT_FAILURE);
	}

	fill_sequential_matrix(hMtrxSrc, Height, Width);

	DataType_t * dMtrxSrc = nullptr;
	cudaCheckError(cudaMalloc(&dMtrxSrc, MemElements));

	DataType_t * dMtrxDst = nullptr;
	cudaCheckError(cudaMalloc(&dMtrxDst, MemElements));

	cudaCheckLastErrorCont();
	cudaMemcpy(dMtrxSrc, hMtrxSrc, MemElements, cudaMemcpyHostToDevice);
	cudaCheckLastError();

	cudaCheckLastErrorCont();
	matrix_transpose_naive_full_mapped<<<Blocks, ThreadsInBlock>>>(dMtrxDst, dMtrxSrc, Height, Width);
	cudaCheckLastError();

	cudaCheckLastErrorCont();
	cudaMemcpy(hMtrxDst, dMtrxDst, MemElements, cudaMemcpyDeviceToHost);
	cudaCheckLastError();

	cudaCheckError(cudaFree(dMtrxDst));
	cudaCheckError(cudaFree(dMtrxSrc));

	// Verify matrix correct: Height <-> Width, Matrix transposed and data flatten !!
	check_transposed_matrix(hMtrxDst, hMtrxSrc, Width, Height);

	//std::cout << "Matrix src:\n";
	//show_matrix(hMtrxSrc, Height, Width);

	//std::cout << "Matrix transposed:\n";
	// Height <-> Width, Matrix transformed and data flatten !!
	//show_matrix(hMtrxDst, Width, Height);

	free(hMtrxDst);
	free(hMtrxSrc);
}
