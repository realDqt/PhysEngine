#pragma once
#include "math/math.h"
#include <helper_functions.h>
#include "vector_functions.h"
#include "object/particle_system_util.h"
#include "object/particle_fluid.h"

using namespace physeng;


extern vec3r g_phaseColor[5];

vec3r getPhaseColor(const PhaseType& phase);

// Particle system class
class FluidSystem
{
public:
    FluidSystem(Real radius, uint3 gridSize, vec3r worldMin, vec3r worldMax, bool useFoam=false, bool useSurfaceTension=false);
    ~FluidSystem();

    
    //// simulation loop
    void update(Real deltaTime);
    
    //// return particle radius
    Real getParticleRadius(){ return m_params.particleRadius; }

    //// update column obstacles
    void updateColumn(int idx, Real r, vec3r x){
        if(idx<0 || idx>3) return;
        if(r>0) m_params.ocol_r[idx]=r;
        if(length(x)>0) m_params.ocol_x[idx]=x;
    }

    //// update sphere obstacles
    void updateSphere(int idx, Real r, vec3r x){
        if(idx<0 || idx>3) return;
        if(r>0) m_params.osph_r[idx]=r;
        if(length(x)>0) m_params.osph_x[idx]=x;
    }

    //// clear    
    void clear();

    //// sample materials
    void addDam(vec3r center, vec3r scale, Real spacing, PhaseType phase);
    void addSandpile(vec3r center, vec3r scale, Real spacing, PhaseType phase);
    void addCubeFromMesh(vec3r center, vec3r scale, Real spacing, PhaseType phase);
    void addModel(const char* str, vec3r center, vec3r scale, Real spacing, PhaseType phase);
    void addTerrain();
    void addTerrain(const char* filePath);

    //// submit parameters to device
    void updateParams(){ setParameters(&m_params); }

    //// submit host arrays to device
    void submitToDevice(uint start, uint count);


    //// function for phase flags
    bool getPhaseFlag(PhaseType ph){ return (m_phaseFlag&(1<<ph))!=0; }
    void setPhaseFlag(PhaseType ph){ m_phaseFlag |= (1<<ph); }
    void resetPhaseFlag(PhaseType ph){ m_phaseFlag &= ~(1<<ph); }
    void clearPhaseFlag(){ m_phaseFlag = 0; }

    void completeInit();
    

protected: // methods
    FluidSystem() {}
    void _initialize();
    void _finalize();

public:
    physeng::ParticleFluid<MemType::GPU> pf = physeng::ParticleFluid<MemType::GPU>(0, 0);
    SimParams m_params; //!< simulation parameters

    DEFINE_SOA_SET_GET(vec3r, MemType::GPU, m_c, Color); //!< particle color array for rendering
    DEFINE_MEMBER_SET_GET(uint, m_solverIterations, SolverIterations); //!< solver iteration
    DEFINE_MEMBER_SET_GET(uint, m_subSteps, SubSteps); //!< solver substeps
    DEFINE_MEMBER_GET(uint, m_numParticles, NumParticles); //<! max particle number
    DEFINE_MEMBER_GET(uint, m_curNumParticles, CurNumParticles); //!< the number of used particles
    DEFINE_MEMBER_GET(bool, m_isInited, IsInited); //!< the number of used particles
    int getFoamParticleCount(){ return *m_foamParticleCount; }

    void CreateRigidFromMesh(std::vector<vec4r>& particles, std::vector<vec3r> vertices, std::vector<int> indices, float spacing, float expand, PhaseType phase);
    void addParticles(std::vector<vec4r>& particles, PhaseType phase);
// data
protected: 
    uint m_phaseFlag; //!< mark the phase used in simulation
    bool ifStartUpdate = false;

    //Rigid Constraint number -> use to locate arrays
    uint m_rigidConstraintCount;
    uint m_rigidParticleCount;
    
    // CPU data
    VecArray<vec3r, MemType::CPU> m_hx;                   //!< particle position
    VecArray<vec4r, MemType::CPU> m_hv;                   //!< particle velocity
    VecArray<Real, MemType::CPU> m_hm;                    //!< particle mass
    VecArray<uint, MemType::CPU> m_hp;                    //!< particle phase
    VecArray<uint, MemType::CPU> m_hRigidParticleSign;    //!< sign the rigid particles to avoid self-collide
    VecArray<vec3r, MemType::CPU> m_hc;                   //!< particle color
    VecArray<Real, MemType::CPU> m_hTerrainHeight;         //!< terrain height

    VecArray<uint, MemType::GPU> m_dConstraintStart;      //!< for rigidbody
    VecArray<uint, MemType::GPU> m_dConstraintCnt;        //!< for rigidbody
    VecArray<uint, MemType::GPU> m_dConstraintParticleMap;//!< for rigidbody

    VecArray<vec3r, MemType::GPU> m_dr;                   //!< for rigidbody, 3x3, last R in shape matching
    VecArray<vec3r, MemType::GPU> m_dq;                   //!< for rigidbody, q in Apq

    VecArray<uint, MemType::GPU> m_dCellStart;            //!< index of start of each cell in sorted list
    VecArray<uint, MemType::GPU> m_dCellEnd;              //!< index of end of cell
    VecArray<Real, MemType::GPU> m_dSortedPos;            //!< sorted position array
    VecArray<Real, MemType::GPU> m_dTerrainHeight;         //!< terrain height

    // Diffuse Particle Data
    int *m_foamParticleCount;
    VecArray<vec3r, MemType::GPU> m_fpx;                  //!< foam particles position
    VecArray<vec3r, MemType::GPU> m_tfpx;                 //!< temp foam particles position
    VecArray<vec4r, MemType::GPU> m_fpv;                  //!< foam particles velocity
    VecArray<vec4r, MemType::GPU> m_tfpv;                  //!< temp foam particles velocity
    VecArray<Real, MemType::GPU> m_fplife;                //!< foam particles life
    VecArray<uint, MemType::GPU> S20FoamParticleIndex;

    // VecArray<NbrList, MemType::GPU> m_dNbrs;              //!< try to store nbr list
    
    // grid params
    uint m_gridSortBits;
    uint3 m_gridSize;
    uint m_numCells;

    void resize(int num);
};