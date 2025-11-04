#pragma once

#include <vector>
#include <algorithm>
#include "allocator.h"

//// cpu array for objects, AOS
#define ObjectArray std::vector

//// cpu/gpu array for vector/matrix/real, , SOA
PHYS_NAMESPACE_BEGIN

template<typename T, MemType MT>
class VecArray
{
public:
    VecArray(){
        m_size=0;
        m_data=nullptr;
    }

    VecArray(unsigned int size){
        m_size=size;
        m_data=nullptr;
        if(m_size!=0) this->alloc(size);
    }

    VecArray(unsigned int size, const T& t){
        m_size=size;
        m_data=nullptr;
        if(m_size!=0){
            this->alloc(size);
            this->fill(size, t);
        }
    }
    
    int size() const { return m_size; }
    void resize(unsigned int size, bool bKeepData=true){
        //// new array
        T* data;
        allocArray<T,MT>(&data, size);
        //// copy from old array
        if(bKeepData && m_data) copyArray<T,MT,MT> (&data, &m_data, min(size, m_size)); 
        //// replace
        if(m_data) freeArray<T,MT>(&m_data);
        m_data=data;
        m_size=size;
    }
    
    void release() { if(m_data) freeArray<T,MT>(&m_data); m_data=nullptr;}
    
    void swap(VecArray<T,MT>& va) {
        if(m_size==0 || va.m_size==0){
            LOG_OSTREAM_WARN<<"Swap array: empty array"<<std::endl;
            return;
        }
        if(this->m_size!=va.size()){
            LOG_OSTREAM_ERROR<<"Swap array: different size"<<std::endl;
            return;
        }
        T* tmp=this->m_data;
        this->m_data=va.m_data;
        va.m_data=tmp;
    }

    void fill(unsigned int size, const T& t){ fillArray<T,MT>(&m_data, t, size); }

    void alloc(unsigned int size){
        allocArray<T,MT>(&m_data, size);
        m_size=size;
    }
    
    PE_CUDA_FUNC inline unsigned int size() { return m_size; }
    PE_CUDA_FUNC inline T& operator[](unsigned int i){ return m_data[i]; }
    PE_CUDA_FUNC inline T operator[](unsigned int i) const { return m_data[i]; }

protected:
    int m_size;
public:
    T* m_data;
};

#define DEFINE_SOA_SET_GET(T,MT,name,Name)\
    protected:\
    VecArray<T,MT> name;\
    public:\
    void set##Name(int idx, const T& t){ name[idx] = t; }\
    const T& get##Name(int idx) const { return name[idx]; }\
    VecArray<T,MT>& get##Name##Ref() { return name; }


#define DEFINE_SOA_GET(T,MT,name,Name)\
    protected:\
    VecArray<T,MT> name;\
    public:\
    const T& get##Name(int idx) const { return name[idx]; }\
    VecArray<T,MT>& get##Name##Ref() { return name; }

#define USING_BASE_SOA_SET_GET(name,Name)\
    using Base::name; using Base::set##Name; using Base::get##Name; using Base::get##Name##Ref;

#define USING_BASE_SOA_GET(name,Name)\
    using Base::name; using Base::get##Name; using Base::get##Name##Ref;


#define MAX_NEIGHBOR_NUM 126

class NbrList {
public:
    __host__ __device__ NbrList() { cnt = 0; }
    // NbrList(const NbrList& nbrlist) {cnt=nbrlist.cnt; nbr_pair=nblist.nbrpair;}
    // __host__ __device__ void add(int idx, Real dist) { if (cnt < MAX_NEIGHBOR_NUM) { nbr_pair[cnt] = make_vec2r(Real(idx), dist); cnt++; } }
    inline __host__ __device__ void add(int idx, Real dist) { nbr_pair[cnt] = make_vec2r(Real(idx), dist); cnt++; }

    // __host__ __device__ vec2r get(int off) { return nbr_pair[off]; }
    __host__ __device__ void get(int off, int& idx, Real& dist) const { const vec2r& pair = nbr_pair[off]; idx = int(pair.x); dist = pair.y; }

    vec2r nbr_pair[MAX_NEIGHBOR_NUM];
    int cnt;
};



//// Read and write API for vert_and_slot combination
__inline__ __host__ __device__ uint getVertSlot(uint v, uint s){
    // return (v<<5)|(s&0x1F);
    return (v<<6)|(s&0x3F);
}

__inline__ __host__ __device__ uint getVertFromInfo(uint vs){
    // return vs>>5;
    return vs>>6;
}

__inline__ __host__ __device__ uint getSlotFromInfo(uint vs){
    // return vs&0x1F;
    return vs&0x3F;
}


#define CONSTRAINT_SLOT_SIZE 64
class ConstraintWriteSlot{
//// The write buffer for each particle to avoid write conflict on the GPU
public:
    __host__ __device__ ConstraintWriteSlot(){
        for(int i=0;i<CONSTRAINT_SLOT_SIZE;i++) slots[i]=make_vec3r(0.0);
    }
    vec3r slots[CONSTRAINT_SLOT_SIZE];
};

class ClothNbrList {
//// The connected particle(vertex) neighbors for each cloth particle(vertex)
public:
    int nbrs[16];  // neighbor index
    Real lens[16]; // rest length between them
};

class TriangleInfo {
//// Current triangle status, including the positions of triangle vertices, the center and the normal
public:
    __host__ __device__ TriangleInfo(){;}
    __host__ __device__ TriangleInfo(const vec3r& p0, const vec3r& p1, const vec3r& p2, const vec3r& cc, const vec3r& nn):c(cc),n(nn),padding(0){
        p[0]=p0;
        p[1]=p1;
        p[2]=p2;
    }
    vec3r p[3];   // current position of each triangle
    vec3r c;      // center of the triangle
    vec3r n;      // normal of the triangle (unused)
    Real padding; // padding for alignment
};


class RestTriangleInfo {
//// Triangle information including the index and slot offset of each triangle vertex and the inverse of the constant reference shape matrix $\mathbf{D}_m$ used in deformation gradient.
public:
    uint4 vert_and_slot;    //// the combination of vertex index and its slot offset of each vertex. Each int includes the vert index[0-27], and the vert TetraWriteSlot's slot idx[27-32]. read/write through getVertSlot/getVertFromInfo/getSlotFromInfo
    vec4r inv_rest_mat;     //// inverse of the constant reference shape matrix D_m used in deformation gradient
};

class DeformNbrList {
public:
    int nbrs[16];
    Real lens[16];
};

class TetInfo {
public:
    __host__ __device__ TetInfo(){;}
    __host__ __device__ TetInfo(const vec3r& p0, const vec3r& p1, const vec3r& p2, const vec3r& p3, const vec3r& cc):c(cc),padding(0) {
        p[0] = p0;
        p[1] = p1;
        p[2] = p2;
        p[3] = p3;
    }
    vec3r p[4];
    vec3r c;
    Real padding;
};

class RestTetInfo {
public:
    //// todo:merge vert and vert_slot??
    uint4 vert_and_slot;    //// 4 vertices, each int includes vert index[0-27], vert TetraWriteSlot's slot idx[27-32]
    vec3r inv_rest_mat[3]; //// mat3 (inverse and transpose of the constant reference shape matrix D_m used in deformation gradient)
    Real vol;              //// rest volume
    Real padding[2];       //// padding
};


PHYS_NAMESPACE_END