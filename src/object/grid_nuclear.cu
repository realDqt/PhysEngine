#include "object/grid_nuclear.h"
#include "curand_kernel.h"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

PHYS_NAMESPACE_BEGIN

inline __host__ __device__ double getPollutionC(int3 center, int3 position, Real time){
    //parameters
    double a,b,c; 
    //half-life period
    double T5 = 456;
    //set according to the data
    a=0.22;
    b=0.0001;
    c=0.20;

    //frequency of the source about releasing the pollution
    //Time interval for release
    double frequency = 1500,deltaT = 2;  
    //release rate * time interval * correction factor
    double Qi = frequency*deltaT;
    double factor = exp(-0.693 * time / 4);
    //diffusion coefficient
    double sigmax,sigmay,sigmaz;
    // double z_inv = 0.09 * 6 / (REAL_PI/(12 * 3600));
    //Inversion Height
    double z_inv = 500; 
    // printf("time %lf\n",time);

    sigmax = sigmaz =  a*2 *time*pow(1+b*5*time, 0.5); 
    sigmay =  c*2 * time;
    // sigmaz = d * pow((double)position.z,e+f*log((double)position.z));
    // sigmax =  5;
    // sigmay =  5;
    // sigmaz = 2;
    double z_part = 0;
    for(int i = -3;i < 4; ++i){
    z_part += exp(-0.5*pow(((position.y-center.y+2*i*z_inv)/sigmaz),2))
    + exp(-0.5*pow(((position.y+center.y+2*i*z_inv)/sigmaz),2));
    }
    
    //calculate according to the nuclear model
    double result = Qi/(pow((2*REAL_PI),1.5)*sigmax * sigmay*sigmaz)* 
    exp(-0.5 * (pow((position.x - center.x)/sigmax,2))-0.5 * (pow((position.z - center.z)/sigmay,2)))
    *z_part ;
    return result;
}

///////////////////////////////////////////////////

//calculate pollution for each grid
//this is related to the source position and diffusion time
template<MemType MT>
__device__ void _k_addConcentration(int index, int3 source,Real time, Real* __restrict__ Concentration)
{
    int3 gridIdx = _d_getGridIdx(index);
    double newConcentration = 0;
    newConcentration = getPollutionC(source,gridIdx,time);
    
    Concentration[index] += newConcentration;
}
DECLARE_KERNEL_TEMP(addConcentration, int3 source,Real time,Real* __restrict__ Concentration);
__global__ void _g_addConcentration(int size, int blocksize, uint3 totalsize,Real *  height, int3 source,Real time, Real* __restrict__ Concentration)
{
    // printf("size %i\n",size);
    IF_IDX_VALID(size)
    {
        int3 position = make_int3(source.x+ i % blocksize- blocksize/2, source.y+i/blocksize%blocksize- blocksize/2,source.z+ i/blocksize/blocksize - blocksize/2);
        //limit the grid that is effected by the source
        //accelerate computing
        if(position.x >=0 && position.x < totalsize.x && position.z >= 0 && position.z < totalsize.z
        && position.y >= height[position.x + position.z * totalsize.x] && position.y < totalsize.y && position.y >= 0){
            int index = _d_getIdx(position);
            // printf("source : %i %i %i ,  index %i %i %i\n",source.x,source.y,source.z,position.x,position.y,position.z);            
            _k_addConcentration<MemType::GPU>(index, source,time,  Concentration);
        }
    }
}

template<MemType MT>
void callAddConcentration(int size, int blocksize, uint3 totalSize,VecArray<Real,MT> height, int3 source,Real time,VecArray<Real, MT>& Concentration)
{
    FILL_CALL_GPU_DEVICE_CODE(addConcentration,blocksize,totalSize,height.m_data, source,time,  Concentration.m_data);
}
template void callAddConcentration<MemType::GPU>(int,int,uint3,VecArray<Real,MemType::GPU>, int3, Real,  VecArray<Real, MemType::GPU>&);
///////////////////////////////////////////////////////////

///////////////////////////////////////////////////

//this is the second way to accelerate the computation 
//pass all the sources to GPU
//so number of times in and out of kernel can be reduced
template<MemType MT>
__device__ void _k_addConcentration2(int index,int sources_number, VecArray<Source,MemType::GPU> sources, Real* __restrict__ Concentration)
{ 
    int3 gridIdx = _d_getGridIdx(index);
    double newConcentration = 0;
    for(int i =0;i<sources_number;++i){
        if(gridIdx.x >= sources[i].center.x -15 && gridIdx.x <= sources[i].center.x+15 &&
        gridIdx.y >= sources[i].center.y -15 && gridIdx.y <= sources[i].center.y+15 &&
        gridIdx.z >= sources[i].center.z -15 && gridIdx.z <= sources[i].center.z+15){
            newConcentration = getPollutionC(make_int3(sources[i].center),gridIdx,sources[i].time);
            Concentration[index] += newConcentration;
        }
    }
    
}
DECLARE_KERNEL_TEMP(addConcentration2,int sources_number,VecArray<Source,MemType::GPU> sources,Real* __restrict__ Concentration);

__global__ void _g_addConcentration2(int size,int sources_number, VecArray<Source,MemType::GPU> sources, Real* __restrict__ Concentration)
{
    IF_IDX_VALID(size)
        _k_addConcentration2<MemType::GPU>(i,sources_number, sources,  Concentration);
}

template<MemType MT>
void callAddConcentration2(int size,int sources_number, VecArray<Source,MT> sources,VecArray<Real, MT>& Concentration)
{
    FILL_CALL_GPU_DEVICE_CODE(addConcentration2,sources_number, sources,  Concentration.m_data);
}
template void callAddConcentration2<MemType::GPU>(int,int, VecArray<Source, MemType::GPU> ,  VecArray<Real, MemType::GPU>&);
///////////////////////////////////////////////////////////

///////////////////////////////////////////////////

//clear the concentration of each grid
template<MemType MT>
__device__ void _k_clearC(int index, Real* __restrict__ Concentration)
{
    int3 gridIdx = _d_getGridIdx(index);
    
    Concentration[index] = 0;
}
DECLARE_KERNEL_TEMP(clearC, Real* __restrict__ Concentration);
__global__ void _g_clearC(int size,  Real* __restrict__ Concentration)
{
    IF_IDX_VALID(size)
        _k_clearC<MemType::GPU>(i, Concentration);
}

template<MemType MT>
void callClearC(int size, VecArray<Real, MT>& Concentration)
{
    FILL_CALL_GPU_DEVICE_CODE(clearC,  Concentration.m_data);
}
template void callClearC<MemType::GPU>(int, VecArray<Real, MemType::GPU>&);
///////////////////////////////////////////////////////////
PHYS_NAMESPACE_END
