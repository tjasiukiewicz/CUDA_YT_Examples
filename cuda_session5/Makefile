.PHONY := clean

NVCC := nvcc
NVCC_FLAGS := --expt-extended-lambda

CXX := g++
CXX_FLAGS := -Wall -Wextra -pedantic

all: main numeric_unstable

main: main.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

main.o: main.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

numeric_unstable: numeric_unstable.o
	$(CXX) $(CXX_FLAGS) -o $@ $<

numeric_unstable.o: numeric_unstable.cpp Makefile
	$(CXX) $(CXX_FLAGS) -c -o $@ $<


clean:
	@rm -f main main.o numeric_unstable numeric_unstable.o >/dev/null 2>&1

