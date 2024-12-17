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

	DataType_t * hX, * hY, * hDistance, * hSine;
	hX = hY = hDistance = hSine = nullptr;

	cudaCheckError(cudaHostAlloc(&hX, MemElements, cudaHostAllocWriteCombined));
	cudaCheckError(cudaHostAlloc(&hY, MemElements, cudaHostAllocWriteCombined));
	cudaCheckError(cudaHostAlloc(&hDistance, MemElements, cudaHostRegisterPortable));
	cudaCheckError(cudaHostAlloc(&hSine, MemElements, cudaHostRegisterPortable));

	// Fill X, Y coord
	/*
	srand(0);
	for (auto i = 0UL; i < Elements; ++i) {
		// [-1, 1]
		hX[i] = ((rand() * 2.0F) / RAND_MAX) - 1.0F;
		hY[i] = ((rand() * 2.0F) / RAND_MAX) - 1.0F;
	}
	*/

	// Start for GPU
	auto dStart = std::chrono::high_resolution_clock::now();

	DataType_t * dX, * dY, * dDistance, * dSine;
	dX = dY = dDistance = dSine = nullptr;

	for (auto ptr: {&dX, &dY, &dDistance, &dSine}) {
		cudaCheckError(cudaMalloc(ptr, MemElements));
	}

	cudaCheckLastErrorCont();
	cudaMemcpyAsync(dX, hX, MemElements, cudaMemcpyHostToDevice);
	cudaCheckLastError();
	cudaMemcpyAsync(dY, hY, MemElements, cudaMemcpyHostToDevice);
	cudaCheckLastError();

	cudaCheckLastErrorCont();
	calculate<<<Blocks, ThreadsInBlock>>>(dX, dY, dDistance, dSine, Elements);
	cudaCheckLastError();

	cudaCheckLastErrorCont();
	cudaMemcpyAsync(hDistance, dDistance, MemElements, cudaMemcpyDeviceToHost);
	cudaCheckLastError();
	cudaMemcpyAsync(hSine, dSine, MemElements, cudaMemcpyDeviceToHost);
	cudaCheckLastError();

	for (auto ptr: {dX, dY, dDistance, dSine}) {
		cudaCheckError(cudaFree(ptr));
	}

	// End for GPU
	auto dStop = std::chrono::high_resolution_clock::now();

	for (auto ptr: {hX, hY, hDistance, hSine}) {
		cudaCheckError(cudaFreeHost(ptr));
	}

	// End for CPU
	auto hStop = std::chrono::high_resolution_clock::now();

	std::cout << "GPU Execution time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(dStop - dStart).count()
		<< " ms.\n";

	std::cout << "CPU Execution time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(hStop - hStart).count()
		<< " ms.\n";
}
