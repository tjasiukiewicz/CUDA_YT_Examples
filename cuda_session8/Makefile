.PHONY := clean

NVCC := nvcc
NVCC_FLAGS := -std=c++11 -O3 

all: vector_add

vector_add: vector_add.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

vector_add.o: vector_add.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

clean:
	@rm -f vector_add vector_add.o >/dev/null 2>&1

