#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

#ifdef PE_USE_CUDA
#define PE_CUDA_BLOCKS 256

PHYS_NAMESPACE_BEGIN

inline void cudaInit(int argc, char **argv){
    int devID;
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaDevice(argc, (const char **)argv);
    if (devID < 0){
        printf("No CUDA Capable devices found, exiting...\n");
        exit(EXIT_SUCCESS);
    }
}

//// follow helper_cuda.h

#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)
inline void __checkCudaError(cudaError_t err, const char *file, const int line){
    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : CudaError()"
                " %s : (%d) %s.\n",
                file, line, static_cast<int>(err),
                cudaGetErrorString(err));
        exit(-1);
    }
}

#define checkCuda(msg) __checkCuda(msg, __FILE__, __LINE__)
inline void __checkCuda(const char *errMsg, const char *file, const int line){
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file, line, errMsg, static_cast<int>(err),
                cudaGetErrorString(err));
        exit(-1);
    }
}

inline void computeCudaThread(unsigned int n, unsigned int bs, unsigned int &nb, unsigned int &nt){
    nt = min(bs, n);//// #thread
    nb = (n % nt != 0) ? (n / nt + 1) : (n / nt);//// #block
}

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

PHYS_NAMESPACE_END
#endif
