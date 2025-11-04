#pragma once
#include "math/math.h"
#include <helper_functions.h>
#include "vector_functions.h"
#include "object/grid_fluid.h"

using namespace physeng;

// Particle system class
class SmokeSystem
{
public:
    // Constructor
    SmokeSystem(uint3 gridSize, Real cellLength, vec3r worldMin, vec3r worldMax);

    // Destructor
    ~SmokeSystem();

    //// simulation loop
    void update(Real deltaTime);

    //// return particle radius
    Real getCellLength() { return m_params.cellLength; }

    //// clear    
    void clear();

    //// sample materials
    void addDam(vec3r center, vec3r scale, Real spacing);

    //// submit parameters to device
    void updateParams() { setGridParameters(&m_params); }

    //// submit host arrays to device
    void submitToDevice(uint start, uint count);

    /// calulate the lifetime of smoke;
    bool calcLifetime() { return m_lifetime > m_maxlife; }


protected:
    // Methods
    SmokeSystem();
    void _initialize();
    void _finalize();

public:
    physeng::GridFluid<MemType::GPU> gf = physeng::GridFluid<MemType::GPU>(make_uint3(0, 0, 0), 0);
    GridSimParams m_params; //!< simulation parameters

    // Particle color array for rendering
    DEFINE_SOA_SET_GET(vec3r, MemType::GPU, m_c, Color); 

    // Number of solver iterations
    DEFINE_MEMBER_SET_GET(uint, m_solverIterations, SolverIterations); 

    // Number of solver substeps
    DEFINE_MEMBER_SET_GET(uint, m_subSteps, SubSteps); 

    // Flag indicating if the system is initialized
    DEFINE_MEMBER_GET(bool, m_isInited, IsInited);

    DEFINE_MEMBER_GET(Real, m_lifetime, Lifetime);
    DEFINE_MEMBER_SET_GET(Real, m_maxlife, MaxLife);

    DEFINE_MEMBER_SET_GET(vec3r, m_wind, Wind);

    Real getDensity(int x, int y, int z) {
        copyArray<Real, MemType::CPU, MemType::GPU>(&m_hd.m_data, &gf.getDensityRef().m_data, 0, m_gridSize.x * m_gridSize.y * m_gridSize.z);
        int index = x + y * m_params.gridHashMultiplier.y + z * m_params.gridHashMultiplier.z;
        //for (int i = 0; i < m_gridSize.x; i++) {
        //    for (int j = 0; j < m_gridSize.y; j++) {
        //        for (int k = 0; k < m_gridSize.z; k++) {
        //            int index2 = i + j * m_params.gridHashMultiplier.y + k * m_params.gridHashMultiplier.z;
        //            if (m_hd[index2] > 0.0f)
        //                LOG_OSTREAM_INFO << i << ' ' << j << ' ' << k << std::endl;
        //        
        //        }
        //    }
        //}
        return m_hd[index];
    }


protected: 
    // CPU data
    VecArray<Real, MemType::CPU> m_hd; // grid density
    VecArray<Real, MemType::CPU> m_ht; // grid temperature
    VecArray<vec3r, MemType::CPU> m_hv; // particle velocity
    VecArray<vec3r, MemType::CPU> m_hc; // particle color

    // grid params
    uint3 m_gridSize;
    Real m_cellLength;
    uint m_numCells;

    void resize(int num);
};






