.PHONY := clean

NVCC := nvcc
NVCC_FLAGS := -std=c++11 -O3 -Xcompiler="-O3,-march=native,-Wall,-Wextra"

all: main

profile: main Makefile
	nvprof -fo $<.nvvp ./$<

main: main.o Makefile
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

main.o: main.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

clean:
	@rm -f main main.o image_*.pgm main.nvvp >/dev/null 2>&1

