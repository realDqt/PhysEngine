#include "common/allocator.h"

PHYS_NAMESPACE_BEGIN

// #ifdef PE_USE_CUDA

template<typename T>
__global__ void arrayFillKernel(T* ptr, const T t, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) ptr[i] = t;
};

template<typename T>
void fillArrayCuda(T** ptr, const T& t, unsigned int size){
    unsigned int nb=0, nt=0;
    computeCudaThread(size, PE_CUDA_BLOCKS, nb, nt);
    arrayFillKernel<T><<<nb,nt>>>(*ptr, t, size);
    getLastCudaError("arrayFill kernel failed");
};

template void fillArrayCuda<int>(int** ptr, const int& t, unsigned int size);
template void fillArrayCuda<uint>(uint** ptr, const uint& t, unsigned int size);
template void fillArrayCuda<Real>(Real** ptr, const Real& t, unsigned int size);
template void fillArrayCuda<vec4r>(vec4r** ptr, const vec4r& t, unsigned int size);
template void fillArrayCuda<vec3r>(vec3r** ptr, const vec3r& t, unsigned int size);
template void fillArrayCuda<vec2r>(vec2r** ptr, const vec2r& t, unsigned int size);

template void fillArrayCuda<int4>(int4** ptr, const int4& t, unsigned int size);
template void fillArrayCuda<int3>(int3** ptr, const int3& t, unsigned int size);
template void fillArrayCuda<int2>(int2** ptr, const int2& t, unsigned int size);

// #endif

PHYS_NAMESPACE_END
