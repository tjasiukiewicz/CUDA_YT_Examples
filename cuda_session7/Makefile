.PHONY := clean

NVCC := nvcc
NVCC_FLAGS := --expt-extended-lambda

all: belloch_scan

belloch_scan: belloch_scan.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

belloch_scan.o: belloch_scan.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

clean:
	@rm -f belloch_scan belloch_scan.o >/dev/null 2>&1

