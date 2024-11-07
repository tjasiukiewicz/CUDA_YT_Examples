.PHONY := clean

NVCC := nvcc
NVCC_FLAGS := -std=c++11 -O3 

all: vector_add

profile: vector_add
	nvprof -fo $<_streams.nvvp ./$<

vector_add: vector_add.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

vector_add.o: vector_add.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

clean:
	@rm -f vector_add vector_add.o vector_add_streams.nvvp >/dev/null 2>&1

