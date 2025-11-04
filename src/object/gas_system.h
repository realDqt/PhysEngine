#pragma once
#include "math/math.h"
#include <vector>
#include <helper_functions.h>
#include "vector_functions.h"
#include "object/grid_gas.h"

using namespace physeng;

struct GasSource {
    int3 source;
    int radius;
    vec3r velocity;
    Real density;
};

// Particle system class
class GasSystem
{
public:
    // Constructor
    GasSystem(uint3 gridSize, Real cellLength, vec3r worldMin, vec3r worldMax);

    // Destructor
    ~GasSystem();

	int getCellNum() const { return m_numCells; }

    //// simulation loop
    void update(Real deltaTime);

    //// return particle radius
    Real getCellLength() { return m_params.cellLength; }

    //// clear    
    void clear();

    //// submit parameters to device
    void updateParams() { setGridParameters(&m_params); }

    //// submit host arrays to device
    void submitToDevice(uint start, uint count);

    /// calulate the lifetime of smoke;
    bool calcLifetime() { return m_lifetime > m_maxlife; }

    VecArray<unsigned char, MemType::CPU>& getTexture() {
        return m_texture;
    }
protected:
    // Methods
    GasSystem();
    void _initialize();
    void _finalize();
    std::vector<GasSource> gasSources;

public:
    physeng::GasGrid<MemType::GPU> gg = physeng::GasGrid<MemType::GPU>(make_uint3(0, 0, 0), 0);
    GridSimParams m_params; //!< simulation parameters

    // Number of solver iterations
    DEFINE_MEMBER_SET_GET(uint, m_solverIterations, SolverIterations); 

    // Number of solver substeps
    DEFINE_MEMBER_SET_GET(uint, m_subSteps, SubSteps); 

    // Flag indicating if the system is initialized
    DEFINE_MEMBER_GET(bool, m_isInited, IsInited);

    DEFINE_MEMBER_GET(Real, m_lifetime, Lifetime);
    DEFINE_MEMBER_SET_GET(Real, m_maxlife, MaxLife);

    DEFINE_MEMBER_SET_GET(vec3r, m_windDirection, WindDirection);
    DEFINE_MEMBER_SET_GET(Real, m_windStrength, WindStrength);
    DEFINE_MEMBER_SET_GET(bool, m_ifGenerate, IfGenerate);


    DEFINE_MEMBER_SET_GET(vec3r, m_color, Color);
    DEFINE_MEMBER_SET_GET(Real, m_decay, Decay);
    DEFINE_MEMBER_SET_GET(Real, m_decreaseDensity, DecreaseDensity);
    DEFINE_MEMBER_SET_GET(UCHAR, m_ambient, Ambient);
    DEFINE_MEMBER_SET_GET(vec3r, m_lightDir, LightDir);
    DEFINE_MEMBER_SET_GET(Real, m_alpha, Alpha);

    // Real getDensity(int x, int y, int z) {
    //     copyArray<Real, MemType::CPU, MemType::GPU>(&m_hd.m_data, &gf.getDensityRef().m_data, 0, m_gridSize.x * m_gridSize.y * m_gridSize.z);
    //     int index = x + y * m_params.gridHashMultiplier.y + z * m_params.gridHashMultiplier.z;
    //     return m_hd[index];
    // }
    void generateRayTemplate();
    void calcRenderData();
    bool addGasSource(vec3r source, Real radius, vec3r velocity, Real density);
    void addBox(vec3r origin, vec3r size);

    Real getAverageDensity();
protected: 
    // CPU data
    VecArray<Real, MemType::CPU> m_hd;  // grid density
    VecArray<UCHAR, MemType::CPU> m_hi; // grid intensity
    VecArray<bool, MemType::CPU> m_rflag; //grid rigid flag

    VecArray<unsigned char, MemType::CPU> m_texture;

    VecArray<int3, MemType::CPU> rayTemplate; //sample point light

    // grid params
    uint3 m_gridSize;
    Real m_cellLength;
    uint m_numCells;

    //void resize(int num);

    void fillTexture();
    void castLight();
    void lightRay(int x, int y, int z, int n, Real decay);
};






