#pragma once

#include <numeric>
#include <iostream>
#include <iomanip>

constexpr static size_t FieldWidth = 4; // data field size - iomanip

// Sequentialy fill matrix
template<typename T>
void fill_sequential_matrix(T * data, const std::size_t height, const std::size_t width) {
	const size_t elements = height * width;
	std::iota(data, data + elements, T(0));
}

// Naive transpose matrix on CPU
template<typename T>
void transpose_matrix_cpu(T * const data_dst, const T * const data_src,
		const std::size_t height, const std::size_t width) {
	for (auto row = 0UL; row < height; ++row) {
		for(auto col = 0UL; col < width; ++col) {
			data_dst[col * height + row] = data_src[row * width + col];
		}
	}
}

// Check transposed matrix
template<typename T>
bool check_transposed_matrix(const T * const data_src, const T * const data_dst,
		const std::size_t height, const std::size_t width) {
	for (auto row = 0UL; row < height; ++row) {
		for(auto col = 0UL; col < width; ++col) {
			auto & src = data_src[row * width + col];
			auto & dst = data_dst[col * height + row];
			if (src != dst) {
				std::cerr << "Check transpose matrix error at: row = "
					<< row << ", col = " << col << ", expected = "
					<< src << ", obtained = " << dst << '\n';
				return false;
			}
		}
	}
	return true;
}

// Show matric for 'printed type'
template<typename T>
void show_matrix(const T * data, const std::size_t height, const std::size_t width) {
	for (auto row = 0UL; row < height; ++row) {
		for(auto col = 0UL; col < width; ++col) {
			std::cout << std::setw(FieldWidth) << data[row * width + col] << ' ';
		}
		std::cout << '\n';
	}
}
