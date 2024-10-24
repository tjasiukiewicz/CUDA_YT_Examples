.PHONY := clean

NVCC := nvcc
NVCC_FLAGS := --expt-extended-lambda

all: hillis_steele

hillis_steele: hillis_steele.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

hillis_steele.o: hillis_steele.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

clean:
	@rm -f hillis_steele hillis_steele.o >/dev/null 2>&1

