#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <chrono>
#include <iostream>
#include "helper_cuda.cuh"

using DataType_t = float;

constexpr static size_t Length = 33554432; // 32 MB elements

constexpr static size_t ThreadsInBlock = 1024;

// Kernel
// Fake, compute intensive kernel...
template<typename T>
__global__ void calculate(
		const T * const x, const T * const y,
		T * const distance, T * const sine,
		const std::size_t length
) {

	auto const tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < length) {
		auto & mx = x[tid];
		auto & my = y[tid];
		auto & md = distance[tid];
		md = sqrtf(mx * mx + my * my);
	}
}

int main() {

	constexpr static size_t Elements = Length;
	constexpr static size_t MemElements = Elements * sizeof(DataType_t);
	constexpr static size_t Blocks = (Elements + ThreadsInBlock - 1) / ThreadsInBlock;

	// Start for CPU
	auto hStart = std::chrono::high_resolution_clock::now();

	DataType_t * x, * y, * distance, * sine;
	x = y = distance = sine = nullptr;

	for(auto ptr: {&x, &y, &distance, &sine}) {
		cudaCheckError(cudaMallocManaged(ptr, MemElements, cudaMemAttachGlobal));
	}

	// Fill X, Y coord
	/*
	srand(0);
	for (auto i = 0UL; i < Elements; ++i) {
		// [-1, 1]
		x[i] = ((rand() * 2.0F) / RAND_MAX) - 1.0F;
		y[i] = ((rand() * 2.0F) / RAND_MAX) - 1.0F;
	}
	*/

	// Start for GPU
	auto dStart = std::chrono::high_resolution_clock::now();

	cudaStream_t s1;

	cudaCheckError(cudaStreamCreate(&s1));

	cudaCheckLastErrorCont();
	calculate<<<Blocks, ThreadsInBlock, 0, s1>>>(x, y, distance, sine, Elements);
	cudaCheckLastError();

	cudaCheckError(cudaStreamSynchronize(s1));

	cudaCheckError(cudaStreamDestroy(s1));

	for (auto ptr: {x, y, distance, sine}) {
		cudaCheckError(cudaFree(ptr));
	}

	// End for GPU
	auto dStop = std::chrono::high_resolution_clock::now();

	// End for CPU
	auto hStop = std::chrono::high_resolution_clock::now();

	std::cout << "GPU Execution time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(dStop - dStart).count()
		<< " ms.\n";

	std::cout << "CPU Execution time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(hStop - hStart).count()
		<< " ms.\n";
}
