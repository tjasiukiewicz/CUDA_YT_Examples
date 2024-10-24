.PHONY := clean

NVCC := nvcc
NVCC_FLAGS := --expt-extended-lambda

all: hills_steele

hills_steele: hills_steele.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

hills_steele.o: hills_steele.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

clean:
	@rm -f hills_steele hills_steele.o >/dev/null 2>&1

