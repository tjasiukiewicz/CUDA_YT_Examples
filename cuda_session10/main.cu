#include <cuda_runtime.h>
#include <chrono>
#include <limits>
#include <cstdint>
#include <fstream>
#include <vector>
#include <cstddef>
#include <string>
#include <iostream>

using DataType_t = uint8_t;
using RuleType_t = uint8_t;

constexpr static size_t Width = 1024;
constexpr static size_t Height = 1024;

constexpr static size_t ThreadsInBlock = 1024;

constexpr static RuleType_t RulesCount = 100;

constexpr static DataType_t LiveCell = std::numeric_limits<DataType_t>::max();
constexpr static DataType_t DeadCell = std::numeric_limits<DataType_t>::min();

// Return PGM image string
template<typename T>
std::string buffer_to_pgm_string(
		const T * buffer, const std::size_t width, const std::size_t height,
		const T color_depth
);

// Save buffer as PGM image
template<typename T>
void save_buffer_to_file(
		const T * buffer, const std::size_t width, const std::size_t height,
		const char * file_name, const T color_depth
);

// Kernel
template<typename T, typename RuleType>
__global__ void calculate_generations(
		T * const generations,
		const std::size_t width, const std::size_t height,
		const RuleType rule
) {

	auto const tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Set live one cell in the middle.
	generations[tid] = (tid == (width / 2)) ? LiveCell: DeadCell;
	__syncthreads();

	// Calculate next generations
	for (auto gen_count = 1U; gen_count < height; ++gen_count) {
		const auto prev_generation = &(generations[(gen_count - 1) * width]);

		uint8_t neighs = 0;

		const auto left_id = (width + tid - 1) % width;
		const auto right_id = (width + tid + 1) % width;
		
		neighs |= prev_generation[left_id] == LiveCell? 0x4: 0x0;
		neighs |= prev_generation[tid] == LiveCell ? 0x2: 0x0;
		neighs |= prev_generation[right_id] == LiveCell ? 0x1: 0x0;

		generations[gen_count * width + tid] =
			rule & (1 << neighs) ? LiveCell: DeadCell;

		__syncthreads();
	}
}

template<typename T, typename RuleType>
const T * make_generation(const RuleType rule) {

	constexpr static size_t MemElements = Width * Height * sizeof(T);
	constexpr static size_t Block = (Width + ThreadsInBlock - 1) / ThreadsInBlock;

	auto hResults = static_cast<T *>(malloc(MemElements));
	
	T * dResults = nullptr;

	cudaMalloc(&dResults, MemElements);

	calculate_generations<<<Block, ThreadsInBlock>>>(dResults, Width, Height, rule);

	cudaMemcpy(hResults, dResults, MemElements, cudaMemcpyDeviceToHost);

	cudaFree(dResults);

	return hResults;
}

int main() {

	// Start for CPU
	auto hStart = std::chrono::high_resolution_clock::now();

	std::vector<const DataType_t *> results;
	results.reserve(RulesCount);

	// Start for GPU
	auto dStart = std::chrono::high_resolution_clock::now();

	// GPU calculate
	for (RuleType_t rule = 0U; rule < RulesCount; ++rule) {
		const DataType_t * result = make_generation<DataType_t, RuleType_t>(rule);
		results.emplace_back(result);
	}

	cudaDeviceSynchronize();

	// End for GPU
	auto dStop = std::chrono::high_resolution_clock::now();

	// CPU calculate -> save buffer to file
	for (RuleType_t rule = 0U; rule < RulesCount; ++rule) {
		std::string file_name = std::string("image_") + std::to_string(rule)
			+ std::string(".pgm");

		save_buffer_to_file(results[rule], Width, Height, file_name.c_str(), LiveCell);

		// bad design !! allocated in make_generation(..)
		free((void *)(results[rule]));
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
   
template<typename T>
std::string buffer_to_pgm_string(
	const T * buffer, const std::size_t width, const std::size_t height,
	const T color_depth
) {    
        // PGM header    
        std::string result = "P2\n"    
                + std::to_string(width) + " " + std::to_string(height) + "\n"    
                + std::to_string(color_depth) + "\n";    
    
        // PGM values (pixels)    
        for (std::size_t row = 0; row < height; ++row) {    
                for (std::size_t col = 0; col < width; ++col) {    
                        result += std::to_string(*(buffer + row * width + col)) + " ";    
                }    
                result += "\n";    
        }    
    
        return result;    
}    
  
template<typename T>
void save_buffer_to_file(
	const T * buffer, const std::size_t width, const std::size_t height, const char * file_name,
	const T color_depth
) {    
        std::ofstream ofile(file_name);    
        if (!ofile) {    
                std::cerr << "Error open file: " << file_name << '\n';    
                exit(EXIT_FAILURE);    
        }    
        ofile << buffer_to_pgm_string(buffer, width, height, color_depth);    
}
