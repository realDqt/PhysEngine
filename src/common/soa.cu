#include "common/soa.h"

PHYS_NAMESPACE_BEGIN

////fill array
template<MemType MT>
__host__ __device__ void _k_fill_array(int i, vec3r tar, vec3r* vx){
    vx[i].x=tar.x;
    vx[i].y=tar.y;
    vx[i].z=tar.z;
}

DECLARE_KERNEL_TEMP(fill_array, vec3r tar, vec3r* vx);
__global__ void _g_fill_array(int size, vec3r tar, vec3r* vx){
    IF_IDX_VALID(size) _k_fill_array<MemType::GPU>(i, tar, vx);
}

template<MemType MT>
void callFillArray(int size, vec3r tar, VecArray<vec3r,MT>& vx){
    // if constexpr(MT==MemType::GPU){
    //     unsigned int nb=0, nt=0;
    //     computeCudaThread(size, PE_CUDA_BLOCKS, nb, nt);
    //     _g_fill_array<<<nb,nt>>>(size, vx.m_data);
    //     getLastCudaError("arrayFill kernel failed");
    // }
    // if constexpr(MT==MemType::CPU){
    //     LOG_OSTREAM_DEBUG<<"define call kernel "<<MT<<std::endl;
    //     #pragma omp parallel for
    //     for(unsigned int i=0;i<size;i++)
    //         _k_fill_array<MemType::CPU>(i, vx.m_data);
    // }
    FILL_CALL_DEVICE_CODE(fill_array, tar, vx.m_data);
}
template void callFillArray<MemType::CPU>(int, vec3r, VecArray<vec3r,MemType::CPU>&);
template void callFillArray<MemType::GPU>(int, vec3r, VecArray<vec3r,MemType::GPU>&);

////add array
////define the kernel
template<MemType MT>
__host__ __device__ void _k_add_array(int i, vec3r* a, vec3r* b){ a[i]+=b[i]; }
////explicit declare the CPU/GPU version
DECLARE_KERNEL_TEMP(add_array, vec3r* a, vec3r* b);
////define GPU global function
__global__ void _g_add_array(int size, vec3r* a, vec3r* b){
    IF_IDX_VALID(size) _k_add_array<MemType::GPU>(i, a, b);
}
////define host function
template<MemType MT>
void callAddArray(int size, VecArray<vec3r,MT>& a, VecArray<vec3r,MT>& b){
    FILL_CALL_DEVICE_CODE(add_array, a.m_data, b.m_data);
}
////explicit declare the CPU/GPU version
template void callAddArray<MemType::CPU>(int, VecArray<vec3r,MemType::CPU>&, VecArray<vec3r,MemType::CPU>&);
template void callAddArray<MemType::GPU>(int, VecArray<vec3r,MemType::GPU>&, VecArray<vec3r,MemType::GPU>&);


template<MemType MT>
__host__ __device__ void _k_add_vec_to_array(int i, vec3r v, vec3r* a){ a[i]+=v; }
////explicit declare the CPU/GPU version
DECLARE_KERNEL_TEMP(add_vec_to_array, vec3r v, vec3r* a);
////define GPU global function
__global__ void _g_add_vec_to_array(int size, vec3r v, vec3r* a){
    IF_IDX_VALID(size) _k_add_vec_to_array<MemType::GPU>(i, v, a);
}
////define host function
template<MemType MT>
void callAddArray(int size, vec3r v, VecArray<vec3r,MT>& a){
    FILL_CALL_DEVICE_CODE(add_vec_to_array, v, a.m_data);
}
////explicit declare the CPU/GPU version
template void callAddArray<MemType::CPU>(int, vec3r, VecArray<vec3r,MemType::CPU>&);
template void callAddArray<MemType::GPU>(int, vec3r, VecArray<vec3r,MemType::GPU>&);

PHYS_NAMESPACE_END