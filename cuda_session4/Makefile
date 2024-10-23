.PHONY := clean

NVCC := nvcc
NVCC_FLAGS := --expt-extended-lambda

all: index

index: index.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

index.o: index.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

clean:
	@rm -f index index.o >/dev/null 2>&1

