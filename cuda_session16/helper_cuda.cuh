#pragma once

#include <cstdio>
#include <cuda_device_runtime_api.h>

/* CUDA error handle macro, *exit* on any error */        
#define cudaCheckError(status)  do {                    	\
        if (status != cudaSuccess) {				\
        	showCudaError(status, __FILE__, __LINE__);      \
                exit (EXIT_FAILURE);        			\
        }							\
} while(0)        

/* CUDA error handle macro, *continue* on any error */        
#define cudaCheckErrorCont(status) do {				\
        if (status != cudaSuccess) {				\
        	showCudaError(status, __FILE__, __LINE__);      \
        }							\
} while(0)        

/* CUDA last error handle macro, *exit* on any error */        
#define cudaCheckLastError() do {	\
	cudaError_t status = cudaGetLastError();		\
	if (status != cudaSuccess) {				\
		showCudaError(status, __FILE__, __LINE__);	\
		exit (EXIT_FAILURE);				\
	}							\
} while(0)

/* CUDA last error handle macro, *continue* on any error */        
#define cudaCheckLastErrorCont() do {				\
	cudaError_t status = cudaGetLastError();		\
	if (status != cudaSuccess) {				\
		showCudaError(status, __FILE__, __LINE__);	\
	}							\
} while(0)


/* Show error for CUDA call */        
void showCudaError (cudaError_t status, const char *file, int line) {        
        if (status != cudaSuccess) {        
                fprintf(stderr, "ERROR: CUDA Call error at %s:%d %s: %s\n",    
                        file, line, cudaGetErrorName(status), cudaGetErrorString(status));        
        }        
}
