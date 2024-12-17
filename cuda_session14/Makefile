.PHONY := clean

NVCC := nvcc
NVCC_FLAGS := -std=c++11 -O3 -Xcompiler="-O3,-march=native,-Wall,-Wextra"

all: main main_um

profile: main Makefile
	nvprof -fo $<.nvvp ./$<

profile_um: main_um Makefile
	nvprof -fo $<.nvvp ./$<

main: main.o Makefile
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

main.o: main.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

main_um: main_um.o Makefile
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

main_um.o: main_um.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

clean:
	@rm -f main main.o main_um main_um.o main.nvvp main_um.nvvp >/dev/null 2>&1

