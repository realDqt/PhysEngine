#pragma once
#include <functional>
#include "common/array.h"


PHYS_NAMESPACE_BEGIN

//// after

#define DECLARE_GPU_KERNEL_TEMP(func_name,...)\
    template __device__ void _k_##func_name<MemType::GPU>(int i, __VA_ARGS__);

#define FILL_CALL_GPU_DEVICE_CODE(func_name,...)\
        {\
        unsigned int nb=0, nt=0;\
        computeCudaThread(size, PE_CUDA_BLOCKS, nb, nt);\
        _g_##func_name<<<nb,nt>>>(size, __VA_ARGS__);\
        getLastCudaError("arrayFill kernel failed");\
        }
    
    
#define DECLARE_KERNEL_TEMP(func_name,...)\
    template __host__ __device__ void _k_##func_name<MemType::CPU>(int i, __VA_ARGS__);\
    template __host__ __device__ void _k_##func_name<MemType::GPU>(int i, __VA_ARGS__);

#define FILL_CALL_DEVICE_CODE(func_name,...)\
    if constexpr(MT==MemType::GPU){\
        unsigned int nb=0, nt=0;\
        computeCudaThread(size, PE_CUDA_BLOCKS, nb, nt);\
        _g_##func_name<<<nb,nt>>>(size, __VA_ARGS__);\
        getLastCudaError("arrayFill kernel failed");\
    } else {\
        __pragma(omp parallel for) \
        for(unsigned int i=0;i<size;i++) _k_##func_name<MemType::CPU>(i, __VA_ARGS__);}

// fix bug here
#define IF_IDX_VALID(size)\
    int i = blockIdx.x * blockDim.x + threadIdx.x;if (i < size)

#define DECLARE_CALL_CPU_KERNEL(func_name,...)\
template void c_##func_name<MemType::CPU>(int size, __VA_ARGS__);
#define DECLARE_CALL_GPU_KERNEL(func_name,...)\
template void c_##func_name<MemType::GPU>(int size, __VA_ARGS__);


//// some common kernel function
template<MemType MT>
void callFillArray(int size, vec3r tar, VecArray<vec3r,MT>& vx);

template<MemType MT>
void callAddArray(int size, VecArray<vec3r,MT>& a, VecArray<vec3r,MT>& b);

template<MemType MT>
void callAddArray(int size, vec3r tar, VecArray<vec3r,MT>& vx);


PHYS_NAMESPACE_END