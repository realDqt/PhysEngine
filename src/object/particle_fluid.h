#pragma once
#include "object/particle_system.h"
#include "object/particle_system_util.h"
#include "common/soa.h"

PHYS_NAMESPACE_BEGIN

//// init particle fluid
template<MemType MT>
void initParticleFluid();

// template<MemType MT>
// void callAdvect(int size, Real dt, VecArray<vec3r,MT>& vx,  VecArray<vec3r,MT>& vv);

// template<MemType MT>
// void callStackParticle(int size, int x, int y, int z, Real gap, VecArray<vec3r,MT>& vx);

// template<MemType MT>
// void callComputeAttractForce(int size, vec3r center, Real scale, VecArray<vec3r,MT>& vf, VecArray<vec3r,MT>& vx);




void reorderDataAndFindCellStart(uint size, uint* cellStart, uint* cellEnd, vec4r* sortedPositionPhase, vec4r* sortedVel, Real* sortedMass, uint* sortedPhase, uint* gridParticleHash, uint* O2SParticleIndex, uint* S2OparticleIndex, vec3r* oldPos, vec4r* oldVel, Real* oldMass, uint* oldPhase, uint numCells, uint numParticles);

// template<MemType MT>
// void callEnforceFrictionBoundary(int size, VecArray<vec3r,MT>& newPos, VecArray<vec3r,MT>& oldPos, VecArray<vec3r,MT>& sortedPos, VecArray<uint,MT>& sortedPhase, VecArray<uint,MT>& S2OParticleIndex);

template<MemType MT>
void callResolvePenetration(int size, VecArray<vec3r,MT>& tempPosition, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd, VecArray<uint,MT>& rigidParticleSign);

template<MemType MT>
void callResolveFriction(int size, VecArray<vec4r,MT>& tempStarPositionPhase, VecArray<vec3r,MT>& tempPosition, VecArray<vec3r,MT>& oldPosition, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd, VecArray<uint,MT>& rigidParticleSign);

template<MemType MT>
void callUpdateLambda(int size, VecArray<Real,MT>& lambda, VecArray<Real,MT>& InvDensity, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

template<MemType MT>
void callSolveFluid(int size, VecArray<vec4r,MT>& normal, VecArray<vec4r,MT>& newPositionPhase, VecArray<Real,MT>& lambda, VecArray<Real,MT>& InvDensity, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

template<MemType MT>
void callSolveFluidAndViscosity(int size, VecArray<vec4r,MT>& normal, VecArray<vec4r,MT>& dv, VecArray<vec4r,MT>& newPositionPhase, VecArray<Real,MT>& lambda, VecArray<Real,MT>& InvDensityy, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<vec4r,MT>& sortedVel, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

// template<MemType MT>
// void callSolveFluidAndViscosity(int size, VecArray<vec3r,MT>& normal, VecArray<vec3r,MT>& dv, VecArray<vec3r,MT>& newPosStar, VecArray<Real,MT>& lambda, VecArray<Real,MT>& InvDensity, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<vec3r,MT>& sortedVel, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

template<MemType MT>
void callUpdateSurfaceTension(int size, VecArray<vec4r,MT>& force, VecArray<vec4r,MT>& normal, VecArray<Real,MT>& InvDensity, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<Real,MT>& sortedMass, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

template<MemType MT>
void callUpdatePositionVelocity(int size, Real dt, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<vec3r,MT>& oldPos, VecArray<vec4r,MT>& vel, VecArray<vec4r,MT>& dvel, VecArray<vec4r,MT>& force, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

template<MemType MT>
void callSolveShapeMatching(int size, uint rigidConstraintCount, VecArray<vec4r,MT>& tempStarPositionPhase, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<uint,MT>& constraintStartIndex, VecArray<uint,MT>& constraintParticleCount, VecArray<vec3r,MT>& r, VecArray<uint,MT>& constraintParticleMap, VecArray<vec3r,MT>& q, VecArray<Real,MT>& mass, VecArray<uint,MT>& O2SParticleIndex, VecArray<uint,MT>& particleIndex);

template<MemType MT>
void callUpdateVorticity(int, VecArray<vec4r,MT>&, VecArray<Real,MT>&, VecArray<vec4r,MT>&, VecArray<vec4r,MT>&, VecArray<Real,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&);

template<MemType MT>
void callApplyVorticity(int, Real, VecArray<vec4r,MT>&, VecArray<vec4r,MT>&, VecArray<Real,MT>&, VecArray<vec4r,MT>&, VecArray<Real,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&, VecArray<uint,MT>&);


template<MemType MT>
void callUpdateNbrList(int size, VecArray<NbrList,MT>& nbrLists, VecArray<vec4r,MT>& sortedPositionPhase,VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

template<MemType MT>
void callUpdateLambdaFast(int size, VecArray<Real,MT>& lambda, VecArray<Real,MT>& invDensity, VecArray<NbrList,MT>& nbrLists, VecArray<vec4r,MT>& sortedPositionPhase);

template<MemType MT>
void callSolveFluidFast(int size, VecArray<vec4r,MT>& normal, VecArray<vec4r,MT>& newPositionPhase, VecArray<Real,MT>& lambda, VecArray<Real,MT>& invDensity, VecArray<NbrList,MT>& nbrLists, VecArray<vec4r,MT>& sortedPositionPhase);

template<MemType MT>
void callUpdateColorField(int size, VecArray<Real,MT>& colorField, VecArray<Real, MT>& invDensity, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);

template<MemType MT>
void callUpdateNormal(int size, VecArray<vec4r,MT>& normal, VecArray<Real, MT>& colorField, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd);
// void setParameters(SimParams* hostParams);

template<MemType MT>
void generateFoamParticle(int size, int rigidbodySize, Real dt, int* foamParticleCount, VecArray<vec3r, MT> &foamParticlePositionPhase, VecArray<vec4r, MT> &foamParticleVelocity, VecArray<Real, MT> &foamParticleLifetime, VecArray<vec4r,MT>& sortedPositionPhase, VecArray<vec4r,MT>& normal, VecArray<vec4r,MT>& vel, VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd, VecArray<vec3r, MT>& color);

template<MemType MT>
void advectFoamParticle(int size, Real dt, VecArray<vec3r, MT>& foamParticlePositionPhase, VecArray<vec4r, MT>& foamParticleVelocity, VecArray<Real, MT>& foamParticleLifetime, VecArray<vec4r, MT>& sortedPositionPhase, VecArray<vec4r, MT>& vel, VecArray<uint, MT>& S2OParticleIndex, VecArray<uint, MT>& cellStart, VecArray<uint, MT>& cellEnd);


template<MemType MT>
void sortFoamParticle(int size, VecArray<uint, MemType::GPU> S20FoamParticleIndex, VecArray<vec3r, MT>& foamParticlePositionPhase, VecArray<vec3r, MT>& tempFoamParticlePositionPhase, VecArray<vec4r, MT>& foamParticleVelocity, VecArray<vec4r, MT>& tempFoamParticleVelocity, VecArray<Real, MT>& foamParticleLifetime);

template<MemType MT>
void removeFoamParticle(int size, int *foamParticleCount, VecArray<vec3r, MT>& foamParticlePositionPhase, VecArray<vec4r, MT>& foamParticleVelocity, VecArray<Real, MT>& foamParticleLifetime);

template<MemType MT>
void initUselessParticle(int size, int *foamParticleCount, VecArray<vec3r, MT>& foamParticlePositionPhase, VecArray<vec4r, MT>& foamParticleVelocity, VecArray<Real, MT>& foamParticleLifetime);

template<MemType MT>
void callCollideTerrain(int size, VecArray<vec4r, MT>& sortedPositionPhase, VecArray<Real, MT>& terrainHeight, vec3r originPos, uint edgeCellNum, Real cellSize);

template<MemType MT>
void callCollideTerrainFoam(int size, VecArray<vec3r, MT>& foamParticlePosition, VecArray<Real, MT>& terrainHeight, vec3r originPos, uint edgeCellNum, Real cellSize);

void checkConstraint(int,VecArray<uint,MemType::GPU>&,VecArray<uint,MemType::GPU>&);

template<MemType MT>
class ParticleFluid : public ParticleSystem<MT>{
public:
    using Base=ParticleSystem<MT>;

    //// inherent variables
    USING_BASE_SOA_SET_GET(m_x, Position);                  //!< Base::position
    USING_BASE_SOA_SET_GET(m_tx, TempPosition);             //!< Base::temp position
    USING_BASE_SOA_SET_GET(m_v, Velocity);                  //!< Base::velocity
    USING_BASE_SOA_SET_GET(m_tv, TempVelocity);             //!< Base::temp velocity
    USING_BASE_SOA_SET_GET(m_m, Mass);                      //!< Base::mass

    DEFINE_SOA_SET_GET(vec4r, MT, m_txph, TempPositionPhase);          //!< Base::temp position
    DEFINE_SOA_SET_GET(vec4r, MT, m_tsxph, TempStarPositionPhase);          //!< Base::temp position

    //// new variables for particle pbd
    DEFINE_SOA_SET_GET(uint, MT, m_ph, Phase);              //!< material phase
    DEFINE_SOA_SET_GET(uint, MT, m_h, ParticleHash);        //!< particle hash for space sort
    DEFINE_SOA_SET_GET(uint, MT, m_rs, RigidParticleSign);  //!< rigid body sign

    //// index mapping
    DEFINE_SOA_SET_GET(uint, MT, m_s2o, S2OParticleIndex);  //!< map sorted index to origin index
    DEFINE_SOA_SET_GET(uint, MT, m_o2s, O2SParticleIndex);  //!< map origin index to sorted index

    //// sorted variables
    DEFINE_SOA_SET_GET(Real, MT, m_sm, SortedMass);         //!< sorted mass
    // DEFINE_SOA_SET_GET(vec4r, MT, m_sx, SortedPosition);    //!< sorted position
    DEFINE_SOA_SET_GET(vec4r, MT, m_sv, SortedVelocity);    //!< sorted velocity
    DEFINE_SOA_SET_GET(uint, MT, m_sph, SortedPhase);       //!< sorted phase
    DEFINE_SOA_SET_GET(vec4r, MT, m_sxph, SortedPositionPhase);    //!< sorted position and phase
    DEFINE_SOA_SET_GET(vec4r, MT, m_dv, DeltaVelocity);    //!< delta velocity

    //// intermediate variables
    // DEFINE_SOA_SET_GET(Real, MT, m_rho, Density);           //!< fluid density
    DEFINE_SOA_SET_GET(Real, MT, m_irho, InvDensity);           //!< fluid density inverse
    DEFINE_SOA_SET_GET(Real, MT, m_cf, ColorField);         //!< color field
    DEFINE_SOA_SET_GET(Real, MT, m_l, Lambda);              //!< lambda in fluid density constraint
    DEFINE_SOA_SET_GET(vec4r, MT, m_vort, Vorticity);       //!< fluid vorticity (vort, |vort|)
    // DEFINE_SOA_SET_GET(Real, MT, m_o, Omega);               //!< the norm2 of fluid vorticity
    DEFINE_SOA_SET_GET(vec4r, MT, m_n, Normal);             //!< fluid normal
    DEFINE_SOA_SET_GET(vec4r, MT, m_f, Force);              //!< force
    

    ParticleFluid(Real r, unsigned int size, bool useFoam=false):Base(r,size){
        if (size != 0) {
            if (useFoam) {
                m_x.resize(size * 2, false);
            }
            m_txph.resize(size,false);

            m_ph.resize(size, false);
            m_h.resize(size, false);            
            m_rs.resize(size, false);
            m_tsxph.resize(size, false);

            m_s2o.resize(size, false);
            m_o2s.resize(size, false);

            m_sm.resize(size, false);
            // m_sx.resize(size, false);
            m_sv.resize(size, false);
            m_sph.resize(size, false);
            m_sxph.resize(size, false);
            m_dv.resize(size, false);
            
            // m_rho.resize(size, false);
            m_irho.resize(size, false);
            m_cf.resize(size, false);
            m_l.resize(size, false);
            m_vort.resize(size, false);
            m_n.resize(size, false);
            m_f.resize(size, false);
        }
    }

    virtual ~ParticleFluid() {
        LOG_OSTREAM_DEBUG << "release particle fluid" << std::hex << &m_x << std::dec << std::endl;
        m_txph.release();
        
        m_ph.release();
        m_h.release();            
        m_rs.release();
        m_tsxph.release();

        m_s2o.release();
        m_o2s.release();

        m_sm.release();
        // m_sx.release();
        m_sv.release();
        m_sph.release();
        m_sxph.release();
        m_dv.release();
        
        // m_rho.release();
        m_irho.release();
        m_cf.release();
        m_l.release();
        m_vort.release();
        m_n.release();
        m_f.release();
        // m_gradc.release();
        LOG_OSTREAM_DEBUG << "release ParticleFluid finished"<<std::endl;
    }

    void init() {};

    void stack(vec3r length, Real gap){
        vec3r count=length/gap;
        callStackParticle<MT>(m_x.size(), (int)count.x, (int)count.y, (int)count.z, gap, m_x);
    }

    void setVelocity(vec3r vel){
        callFillArray<MT>(m_v.size(), vel, m_v);
    }

    void advect(Real dt){
        callAdvect<MT>(m_x.size(), dt, m_x, m_v);
    }

    void resizePF(unsigned int size, bool useFoam = false){
        if (size != 0) {
            if (useFoam)
                m_x.resize(size * 2);
            else
                m_x.resize(size);

            m_tx.resize(size);
            m_txph.resize(size);
            m_v.resize(size);
            m_tv.resize(size);
            m_m.resize(size);

            m_ph.resize(size);
            m_h.resize(size);            
            m_rs.resize(size);
            m_tsxph.resize(size);

            m_s2o.resize(size);
            m_o2s.resize(size);

            m_sm.resize(size);
            // m_sx.resize(size);
            m_sv.resize(size);
            m_sph.resize(size);
            m_sxph.resize(size);
            m_dv.resize(size);

            // m_rho.resize(size);
            m_irho.resize(size);
            m_cf.resize(size);
            m_l.resize(size);
            m_vort.resize(size);
            m_n.resize(size);
            m_f.resize(size);
        }
    };

    void clearVelocity(){
        callFillArray<MT>(m_v.size(), make_vec3r(0.,0.,0.), m_v);
    }

    void applyBodyForce(vec3r bodyForce, Real dt){
        callAddArray<MT>(m_v.size(), bodyForce*dt, m_v);
    }

    void applyAttractForce(vec3r center, Real scale, Real dt){
        callComputeAttractForce<MT>(m_v.size(), center, scale, m_tx, m_x);
        callAdvect<MT>(m_v.size(), dt, m_v, m_tx);
    }

};
PHYS_NAMESPACE_END