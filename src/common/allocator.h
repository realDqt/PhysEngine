#pragma once

#ifdef _WIN32
#include <stdlib.h>
#else
#include <cstdlib>
#endif

#include "math/math.h"
#include "common/logger.h"
#include "common/cuda/cuda_util.h"
#include "common/cuda/cuda_math.h"

#ifdef PE_USE_CUDA
#include "helper_cuda.h"
#endif

PHYS_NAMESPACE_BEGIN

inline void* alignedAllocC11Std(size_t size, size_t align) {
    size_t mask = align - 1;
    const size_t remainder = size & mask;
    if(remainder != 0)
        size = size & !mask + align;
#ifdef _WIN32
    void* ptr = _aligned_malloc(size, align);
#else
    // TODO FIX ME segment fault using aligned alloc, fallback to normal one for now
    void* ptr = malloc(size);//, align);
#endif
    // LOG_OSTREAM_WARN << "malloc " << size << " bytes at " << ptr << std::endl;
    return ptr;
}

inline void alignedFreeC11Std(void* ptr, size_t align) {
    // LOG_OSTREAM_WARN << "free at " << ptr << std::endl;
#ifdef _WIN32
	_aligned_free(ptr);
#else
    free(ptr);
#endif    
}

//// alloc
template<typename T, MemType MT> 
void allocArray(T** ptr, unsigned int size){
    if constexpr(MT==MemType::CPU)
        *ptr = (T*)malloc(size * sizeof(T));
    #ifdef PE_USE_CUDA
    if constexpr(MT==MemType::GPU)
        physeng::checkCudaError(cudaMalloc(ptr, size*sizeof(T)));
    #else
    if constexpr(MT==MemType::GPU)
        LOG_OSTREAM_ERROR<<"No CUDA Module"<<std::endl;
    #endif
};

//// free
template<typename T, MemType MT> 
void freeArray(T** ptr){
    if constexpr(MT==MemType::CPU){
        free(*ptr); ptr=nullptr; 
        return;
    }
    #ifdef PE_USE_CUDA
    if constexpr(MT==MemType::GPU){
        physeng::checkCudaError(cudaFree(*ptr)); ptr=nullptr; 
        return;
    }
    #else
    if constexpr(MT==MemType::GPU){
        LOG_OSTREAM_ERROR<<"No CUDA Module"<<std::endl;
        return;
    }
    #endif
};


//// fill
//// fill helper
// #ifdef PE_USE_CUDA
template<typename T>
void fillArrayCuda(T** ptr, const T& t, unsigned int size);


template<typename T, MemType MT>
void fillArray(T** ptr, const T& t, int size){
    if constexpr(MT==MemType::CPU){
        for (unsigned int i = 0; i < size; i++) (*ptr)[i] = t;
        return;
    }
    #ifdef PE_USE_CUDA
    if constexpr(MT==MemType::GPU){
        fillArrayCuda(ptr,t,size);
        return;
    }
    #else
    if constexpr(MT==MemType::GPU){
        LOG_OSTREAM_ERROR<<"No CUDA Module"<<std::endl;
        return;
    }
    #endif
}

//// copy
template<typename T, MemType MT1, MemType MT2>
void copyArray(T** ptr1, T** ptr2, unsigned int size){
    if constexpr(MT1==MemType::CPU && MT2==MemType::CPU){
        // LOG_OSTREAM_DEBUG<<"copyArray h->h"<<std::endl;
        memcpy(*(ptr1),*(ptr2), size*sizeof(T)); 
        return;
    }
    #ifdef PE_USE_CUDA
    if constexpr(MT1==MemType::GPU && MT2==MemType::CPU){
        // LOG_OSTREAM_DEBUG<<"copyArray h->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1),*(ptr2), size*sizeof(T), cudaMemcpyHostToDevice)); 
        return;
    }
    if constexpr(MT1==MemType::CPU && MT2==MemType::GPU){
        // LOG_OSTREAM_DEBUG<<"copyArray d->h"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1),*(ptr2), size*sizeof(T), cudaMemcpyDeviceToHost)); 
        return;
    }
    if constexpr(MT1==MemType::GPU && MT2==MemType::GPU){
        // LOG_OSTREAM_DEBUG<<"copyArray d->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1),*(ptr2), size*sizeof(T), cudaMemcpyDeviceToDevice));
        return;
    }
    #else
    { LOG_OSTREAM_ERROR<<"No CUDA Module"<<std::endl; return; }
    #endif
    
};

template<typename T, MemType MT1, MemType MT2>
void copyArray(T** ptr1, T** ptr2, unsigned int start, unsigned int size){
    if constexpr(MT1==MemType::CPU && MT2==MemType::CPU){
        LOG_OSTREAM_DEBUG<<"copyArray h->h"<<std::endl;
        memcpy(*(ptr1)+start,*(ptr2)+start, size*sizeof(T)); 
        return;
    }
    #ifdef PE_USE_CUDA
    if constexpr(MT1==MemType::GPU && MT2==MemType::CPU){
        // LOG_OSTREAM_DEBUG<<"copyArray h->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start,*(ptr2)+start, size*sizeof(T), cudaMemcpyHostToDevice)); 
        return;
    }
    if constexpr(MT1==MemType::CPU && MT2==MemType::GPU){
        // LOG_OSTREAM_DEBUG<<"copyArray d->h"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start,*(ptr2)+start, size*sizeof(T), cudaMemcpyDeviceToHost)); 
        return;
    }
    if constexpr(MT1==MemType::GPU && MT2==MemType::GPU){
        // LOG_OSTREAM_DEBUG<<"copyArray d->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start,*(ptr2)+start, size*sizeof(T), cudaMemcpyDeviceToDevice)); 
        return;
    }
    #else
    { LOG_OSTREAM_ERROR<<"No CUDA Module"<<std::endl; return; }
    #endif
    
};

template<typename T, MemType MT1, MemType MT2>
void copyArray(T** ptr1, T** ptr2, unsigned int start1, unsigned int start2, unsigned int size) {
    if constexpr (MT1 == MemType::CPU && MT2 == MemType::CPU) {
        LOG_OSTREAM_DEBUG << "copyArray h->h" << std::endl;
        memcpy(*(ptr1)+start1, *(ptr2)+start2, size * sizeof(T));
        return;
    }
#ifdef PE_USE_CUDA
    if constexpr (MT1 == MemType::GPU && MT2 == MemType::CPU) {
        // LOG_OSTREAM_DEBUG<<"copyArray h->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start1, *(ptr2)+start2, size * sizeof(T), cudaMemcpyHostToDevice));
        return;
    }
    if constexpr (MT1 == MemType::CPU && MT2 == MemType::GPU) {
        // LOG_OSTREAM_DEBUG<<"copyArray d->h"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start1, *(ptr2)+start2, size * sizeof(T), cudaMemcpyDeviceToHost));
        return;
    }
    if constexpr (MT1 == MemType::GPU && MT2 == MemType::GPU) {
        // LOG_OSTREAM_DEBUG<<"copyArray d->d"<<std::endl;
        physeng::checkCudaError(cudaMemcpy(*(ptr1)+start1, *(ptr2)+start2, size * sizeof(T), cudaMemcpyDeviceToDevice));
        return;
    }
#else
    { LOG_OSTREAM_ERROR << "No CUDA Module" << std::endl; return; }
#endif

};


PHYS_NAMESPACE_END