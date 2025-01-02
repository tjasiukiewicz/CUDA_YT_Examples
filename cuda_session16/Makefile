.PHONY := clean

NVCC := nvcc
NVCC_FLAGS := --cudart shared --std=c++11 -O3 -Xcompiler="-O3,-march=native,-Wall,-Wextra"
NVCC_FLAGS += --ptxas-options=-v
#NVCC_FLAGS += --gpu-architecture=native

all: main

profile_occupancy: main
	ncu --rule TheoreticalOccupancy ./main

profile_mem: main
	ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld,smsp__sass_average_data_bytes_per_sector_mem_global_op_st \
		./main 

main: main.o Makefile
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

main.o: main.cu matrix_helper.hpp Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $< 2>&1 > /dev/null | cu++filt

clean:
	@rm -f main main.o main.nvvp >/dev/null 2>&1

