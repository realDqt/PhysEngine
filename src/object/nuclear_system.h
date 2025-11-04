#pragma once
#include "math/math.h"
#include <helper_functions.h>
#include "vector_functions.h"
#include "object/grid_nuclear.h"

using namespace physeng;
// Particle system class



class NuclearSystem
{
public:
    NuclearSystem(uint3 gridSize, Real cellLength, vec3r worldMin, vec3r worldMax);
    ~NuclearSystem();

    
    //// simulation loop
    void update(Real deltaTime);
    
    //// return particle radius
    Real getCellLength(){ return m_params.cellLength; }

    //// clear    
    void clear();

    //// sample materials
    void addDam(vec3r center, vec3r scale, Real spacing);

    //// submit parameters to device
    void updateParams(){ setGridParameters(&m_params); }

    //// submit host arrays to device
    void submitToDevice(uint start, uint count);

    // void setM_hc(int idx,vec3r color);

protected: // methods
    NuclearSystem() {}
    void _initialize();
    void _finalize();

public:
    physeng::Nuclear<MemType::GPU> gf = physeng::Nuclear<MemType::GPU>(make_uint3(0, 0, 0), 0);
    GridSimParams m_params; //!< simulation parameters

    DEFINE_SOA_SET_GET(vec3r, MemType::GPU, m_c, Color); //!< particle color array for rendering
    DEFINE_MEMBER_SET_GET(uint, m_solverIterations, SolverIterations); //!< solver iteration
    DEFINE_MEMBER_SET_GET(uint, m_subSteps, SubSteps); //!< solver substeps
    DEFINE_MEMBER_GET(bool, m_isInited, IsInited);
    DEFINE_MEMBER_SET_GET(Real, norm_wind, NormWind);
    DEFINE_SOA_SET_GET(Source, MemType::GPU, sources_G, Source_G);

    bool start_pollution;
    VecArray<Real, MemType::CPU>& getDensity() { return m_hd; }
// data
protected: 
    
    // CPU data
    VecArray<Real, MemType::CPU> m_hd;                   //!< grid density
    VecArray<Real, MemType::CPU> m_ht;                   //!< grid temperature
    VecArray<vec3r, MemType::CPU> m_hv;                   //!< particle velocity
    VecArray<vec3r, MemType::CPU> m_hc;                   //!< particle color

    // grid params
    uint3 m_gridSize;
    Real m_cellLength;
    uint m_numCells;
    Real time_counter;
    // VecArray<Source,MemType::CPU> sources;
    std::vector<Source> sources;
    float3 wind_velocity;
    //float norm_wind;
    float wind_angle;
    Real r_interval;//释放间隔
    float3 Vs;
    VecArray<Real,MemType::GPU> height;    
	VecArray<Real,MemType::CPU> temp_height;//给更新速度部分使用

    void resize(int num);
};






