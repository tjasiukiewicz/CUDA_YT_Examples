#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <cmath>
#include <map>
#include <cstdio>

template<typename T, typename Func>
std::map<T, unsigned long> compute_numeric_instability(std::vector<T> & vec, float ident, Func func) {

	std::map<T, unsigned long> results;

	do {
		T sum = std::accumulate(vec.cbegin(), vec.cend(), ident, func);
		results[sum] += results.find(sum) == results.end() ? 0: 1;
	} while(std::next_permutation(vec.begin(), vec.end()));

	return results;
}

template<typename T, typename Func>
void test_numeric_instability(const std::string & name, std::vector<T> & vec, T ident, Func func) {

	auto results = compute_numeric_instability(vec, ident, func);

	printf("%s\n", name.c_str());
	puts("Result\t\t\tCount");

	for (auto & pr: results) {
		printf("%20.12f\t%lu\n", pr.first, pr.second);
	}
	puts("");
}

int main() {
	std::vector<float> vec{3.01F, 1.031F, 6.095F, 9.05112F, 0.145F, 1.0131F};

	test_numeric_instability("Test: float (a + b)", vec, 0.0F,
		[](float a, float b) {
			return a + b;
	});

	test_numeric_instability("Test: float (a * b)", vec, 1.0F,
		[](float a, float b) {
			return a * b;
	});

	test_numeric_instability("Test: float (a / b) * (b / a)", vec, 1.0F,
		[](float a, float b) {
			return (a / b) * (b / a);
	});

	test_numeric_instability("Test: float (2.0 * a * 2.0 * b)", vec, 1.0F,
		[](float a, float b) {
			return 2.0F * a * 2.0F * b;
	});

}
