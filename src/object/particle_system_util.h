#pragma once
#include "object/particle_system.h"
#include "common/soa.h"

PHYS_NAMESPACE_BEGIN

typedef unsigned int uint;

//// material phase
// Rigid - rigid body
// Liquid - liquid with larger density
// Oil - another liquid with smaller density
// Sand - granular material
// Cloth - cloth
// Deform - deform
// Hair - hair
// Fixed - fixed point used in cloth simulation or deformable simulation
enum PhaseType
{
    Rigid = 0,  
    Liquid = 1, 
    Oil = 2,    
    Sand = 3,   
    Cloth = 4,  
    Deform = 5, 
    Hair = 6,   
    Fixed = 7   

};

struct SimParams
{
    //// simulation world
    // worldOrigin - origin of the world
    // worldMin - left lower corner of the world
    // worldMax - right upper corner of the world
    vec3r worldOrigin;
    vec3r worldMin;    
    vec3r worldMax;    

    //// particle radius
    // particleRadius - radius of each particle
    // maxNeighbours - max #neighbor
    Real particleRadius; 
    uint maxNeighbours;  

    //// kernel radius
    // h - smooth kernel radius
    // h2 - h*h
    // halfh - h*0.5
    // volume - volume of each particle, 4/3*pi*r^3
    Real h;      
    Real h2;    
    Real halfh;  
    Real volume; 

    //// particle grid
    // gridSize - grid size
    // gridHashMask - mask for hash
    // gridHashMultiplier - multiplier for hash
    // numCells - #cells
    // cellLength - length of each cell
    // invCellLength - inverse of length of each cell
    uint3 gridSize; 
    uint3 gridHashMask;
    uint3 gridHashMultiplier;
    uint numCells;       
    vec3r cellLength;    
    vec3r invCellLength; 

    //// density/mass
    // density0 - default rest density(deprecated)
    // invDensity0 - default 1.0/rest density(deprecated)
    // rho0 - rest density of each phase
    // invRho0 - 1.0/rest density of each phase
    // pmass - particle mass of each phase
    Real density0;    
    Real invDensity0; 
    Real rho0[5];     
    Real invRho0[5];  
    Real pmass[5];    

    //// kernel coef
    // poly6Coef - poly6 kernel
    // zeroPoly6 - poly6 kernel
    // spikyGradCoef - spiky gradient kernel
    // cohesionCoef - cohesion kernel
    // cohesionConstCoef - cohesion kernel
    Real poly6Coef;         
    Real zeroPoly6;         
    Real spikyGradCoef;     
    Real cohesionCoef;      
    Real cohesionConstCoef; 

    //// physics parameter
    // gravity - gravity
    // globalDamping - global damping for each timestep
    // boundaryDamping - boundary restitution
    vec3r gravity;        
    Real globalDamping;   
    Real boundaryDamping; 
    //// --------
    //// rigid/granular material
    // peneDist - collision distance for granular material
    // peneDist2 - collision distance for granular material
    // staticFriction - static friction coefficient
    // dynamicFriction - dynamic friction coefficient
    Real peneDist;        
    Real peneDist2;       
    Real staticFriction;  
    Real dynamicFriction; 
    // Real invStaticFriction;

    //// --------
    //// fluid
    //// fluid density constraint

    //if solid are sampled more densely than fluids then it should be set < 1
    Real solidPressure;
    // user specified relaxation parameter for lambda
    Real lambdaRelaxation; 
    Real pressure;
    Real invPressure;

    //// vorticity/viscosity/surface tension
    // kvorticity - vorticity coefficient
    // kviscosity - viscosity coefficient
    // useSurfaceTension - use surface tension?
    // kcurvature - curvature force coefficient
    // kcohesion - cohesion force coefficient
    Real kvorticity;       
    Real kviscosity;        
    bool useSurfaceTension; 
    Real kcurvature;        
    Real kcohesion;         

    //// --------
    //// foam
    bool useFoam;
    
    //// --------
    //// cloth
    // triCollisionThres - threshold for triangle collision
    // triLength - length of each triangle
    Real triCollisionThres;
    Real triLength;

    //// --------
    //// deform
    //(strain based solver)
    /// stretchStiffness - strech stiffness
    /// shearStiffness - shear stiffness
    vec3r stretchStiffness; 
    vec3r shearStiffness;   
    //(FEM based solver)
    /// youngsModulus - Young's modulus
    /// poissonRatio - Poisson ratio
    /// deformMu - mu determined by young's modulus and poisson ratio
    /// deformLambda - lambda determined by young's modulus and poisson ratio
    /// volumeStiffness - volume constraint stiffness
    /// collisionStiffness - collision stiffness
    Real youngsModulus; 
    Real poissonRatio;  
    Real deformMu;     
    Real deformLambda;  
    Real volumeStiffness; 
    Real collisionStiffness;

    //// control param
    // sleepVelocity - velocity threshold for particle sleeping, smaller velocity will be clamped to 0
    // sleepVelocity2 - velocity threshold for particle sleeping, smaller velocity will be clamped to 0
    // maxVelocity - velocity threshold for CFL, larger velocity will be clamped to maxVelocity
    // maxVelocity2 - velocity threshold for CFL, larger velocity will be clamped to maxVelocity
    Real sleepVelocity;  
    Real sleepVelocity2; 
    Real maxVelocity;    
    Real maxVelocity2;   

    //// obstacles
    // useColumnObstacle - use column obstacle?
    // ocol_r - the radii of columns
    // ocol_x - the positions of columns
    // useSphereObstacle - use sphere obstacle?
    // osph_r - the radii of sphere
    // osph_x - the positions of sphere
    bool useColumnObstacle; 
    Real ocol_r[3];         
    vec3r ocol_x[3];        
    bool useSphereObstacle; 
    Real osph_r[3];         
    vec3r osph_x[3];      

    //// terrain
    bool useTerrain;
};

// global variable on GPU
extern __constant__ SimParams params;
// global variable on CPU
extern SimParams hparams;

// force the position of particle to be inside the boundary
inline __host__ __device__ vec3r _d_enforceBoundaryLocal(vec3r pos)
{
    vec3r npos = pos;
    npos.x = (max(min(npos.x, params.worldMax.x), params.worldMin.x) + pos.x) * 0.5f;
    npos.y = (max(min(npos.y, params.worldMax.y), params.worldMin.y) + pos.y) * 0.5f;
    npos.z = (max(min(npos.z, params.worldMax.z), params.worldMin.z) + pos.z) * 0.5f;
    return npos;
}

// calculate grid hash value for each particle
inline __host__ __device__ int3 _d_calcGridPos(vec3r p)
{
    return make_int3(
        floorf((p.x - params.worldOrigin.x) * params.invCellLength.x),
        floorf((p.y - params.worldOrigin.y) * params.invCellLength.y),
        floorf((p.z - params.worldOrigin.z) * params.invCellLength.z));
}

// calculate address in grid from position (clamping to edges)
inline __host__ __device__ uint _d_calcGridHash(int3 gridPos)
{
    uint x = (uint)gridPos.x & params.gridHashMask.x; // wrap grid, assumes size is power of 2
    uint y = (uint)gridPos.y & params.gridHashMask.y;
    uint z = (uint)gridPos.z & params.gridHashMask.z;
    return z * params.gridHashMultiplier.z + y * params.gridHashMultiplier.y + x;
}

// calculate grid hash value for each particle on CPU
inline __host__ int3 _h_calcGridPos(vec3r p)
{
    return make_int3(
        floorf((p.x - hparams.worldOrigin.x) * hparams.invCellLength.x),
        floorf((p.y - hparams.worldOrigin.y) * hparams.invCellLength.y),
        floorf((p.z - hparams.worldOrigin.z) * hparams.invCellLength.z));
}

// calculate address in grid from position (clamping to edges)
inline __host__ uint _h_calcGridHash(int3 gridPos)
{
    uint x = (uint)gridPos.x & hparams.gridHashMask.x; // wrap grid, assumes size is power of 2
    uint y = (uint)gridPos.y & hparams.gridHashMask.y;
    uint z = (uint)gridPos.z & hparams.gridHashMask.z;
    return z * hparams.gridHashMultiplier.z + y * hparams.gridHashMultiplier.y + x;
}

// check if the position is inside the triangle
// p0, p1, p2 - the vertices of the triangle
// x - the position to be checked
inline __host__ __device__ bool _d_isPointInTriangle(const vec3r &p0, const vec3r &p1, const vec3r &p2, const vec3r &x)
{
    //calculate the edge vectors of the triangle 0->1, 1->2, 2->0
    vec3r r0 = p0 - p2; ////d1
    vec3r r1 = p1 - p2; ////d2
    vec3r r2 = x - p2;  ////pp0

    // calculate the temperary variables
    Real dot00 = dot(r0, r0); ////a
    Real dot01 = dot(r0, r1); ////b d
    Real dot02 = dot(r0, r2); ////c
    Real dot11 = dot(r1, r1); ////e
    Real dot12 = dot(r1, r2); ////f

    Real inverDeno = 1.0f / (dot00 * dot11 - dot01 * dot01); //// 1/det

    Real u = (dot11 * dot02 - dot01 * dot12) * inverDeno; /////ec-bf s
    // if u out of range, return directly
    if (u < -0.01 || u > 1.01)                           
        return false;

    Real v = (dot00 * dot12 - dot01 * dot02) * inverDeno; ////af-dc t
    // if v out of range, return directly
    if (v < -0.01 || v > 1.01)                            
        return false;

    return u + v <= 1.01;
}

inline __host__ __device__ bool _d_projectPointInTriangle(const vec3r &p0, const vec3r &p1, const vec3r &p2, const vec3r &p, Real &b0, Real &b1, Real &b2, vec3r &n)
{
    // find barycentric coordinates of closest point on triangle
    b0 = static_cast<Real>(1.0 / 3.0); // for singular case
    b1 = b0;
    b2 = b0;
    // find the closest point on the triangle
    vec3r d1 = p1 - p0;
    vec3r d2 = p2 - p0;
    vec3r pp0 = p - p0;
    Real a = dot(d1, d1);
    Real b = dot(d2, d1);
    Real c = dot(pp0, d1);
    Real d = b;
    Real e = dot(d2, d2);
    Real f = dot(pp0, d2);
    Real det = a * e - b * d;

    //vec3r np0,np1,np2,edge;
    if (det != 0.0)
    {
        Real s = (c * e - b * f) / det;
        Real t = (a * f - c * d) / det;

        // inside triangle
        b0 = static_cast<Real>(1.0) - s - t; 
        b1 = s;
        b2 = t;

        if (b0 < 0.0)
        { // on edge 1-2
            vec3r d = p2 - p1;
            Real d2 = dot(d, d);
            Real t = (d2 == static_cast<Real>(0.0)) ? static_cast<Real>(0.5) : dot(d, p - p1) / d2;
            t = min(1.0, max(0.0, t));
            b0 = 0.0;
            b1 = (static_cast<Real>(1.0) - t);
            b2 = t;
        }
        else if (b1 < 0.0)
        { // on edge 2-0
            vec3r d = p0 - p2;
            Real d2 = dot(d, d);
            Real t = (d2 == static_cast<Real>(0.0)) ? static_cast<Real>(0.5) : dot(d, p - p2) / d2;
            t = min(1.0, max(0.0, t));
            b1 = 0.0;
            b2 = (static_cast<Real>(1.0) - t);
            b0 = t;
        }
        else if (b2 < 0.0)
        { // on edge 0-1
            vec3r d = p1 - p0;
            Real d2 = dot(d, d);
            Real t = (d2 == static_cast<Real>(0.0)) ? static_cast<Real>(0.5) : dot(d, p - p0) / d2;
            t = min(1.0, max(0.0, t));
            b2 = 0.0;
            b0 = (static_cast<Real>(1.0) - t);
            b1 = t;
        }
    }

    n = p0 * b0 + p1 * b1 + p2 * b2;
    n = p - n;
    return true;
}

void setParameters(SimParams *hostParams);
/**
 * @brief sort triangle data based on the grid index
 * 
 * @param size 
 * @param cellStart 
 * @param cellEnd 
 * @param sortedTriangleInfo 
 * @param sortedTriangleCenter 
 * @param sortedTriangleNormal 
 * @param sortedTriangleVertex 
 * @param gridTriangleHash 
 * @param O2STriangleIndex 
 * @param S2OTriangleIndex 
 * @param triangleInfo 
 * @param triangleCenter 
 * @param triangleNormal 
 * @param triangleVertex 
 * @param numCells 
 */
void reorderDataAndFindCellStart(uint size, uint *cellStart, uint *cellEnd, TriangleInfo *sortedTriangleInfo, vec4r *sortedTriangleCenter, vec4r *sortedTriangleNormal, int4 *sortedTriangleVertex, uint *gridTriangleHash, uint *O2STriangleIndex, uint *S2OTriangleIndex, TriangleInfo *triangleInfo, vec4r *triangleCenter, vec4r *triangleNormal, int4 *triangleVertex, uint numCells);
/**
 * @brief sort particle data based on the grid index
 * 
 * @param cellStart 
 * @param cellEnd 
 * @param sortedPositionPhase 
 * @param gridParticleHash 
 * @param O2SParticleIndex 
 * @param S2OparticleIndex 
 * @param oldPos 
 * @param numCells 
 * @param numParticles 
 */
void reorderParticleDataAndFindCellStart(uint *cellStart, uint *cellEnd, vec4r *sortedPositionPhase, uint *gridParticleHash, uint *O2SParticleIndex, uint *S2OparticleIndex, vec4r *oldPos, uint numCells, uint numParticles);
/**
 * @brief sort particle and object data based on the grid index
 * 
 * @param cellStart 
 * @param cellEnd 
 * @param sortedPositionPhase 
 * @param sortedObjectIdx 
 * @param gridParticleHash 
 * @param O2SParticleIndex 
 * @param S2OparticleIndex 
 * @param oldPos 
 * @param oldObjectIdx 
 * @param numCells 
 * @param numParticles 
 */
void reorderParticleObjectDataAndFindCellStart(uint *cellStart, uint *cellEnd, vec4r *sortedPositionPhase, uint *sortedObjectIdx, uint *gridParticleHash, uint *O2SParticleIndex, uint *S2OparticleIndex, vec4r *oldPos, uint *oldObjectIdx, uint numCells, uint numParticles);
/**
 * @brief update the sorted position and velocity
 * 
 * @tparam MT 
 * @param size 
 * @param dt 
 * @param sortedPositionPhase 
 * @param pos 
 * @param vel 
 * @param S2OParticleIndex 
 */
template <MemType MT>
void callUpdateSortedPositionVelocity(int size, Real dt, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<vec3r, MT> &pos, VecArray<vec4r, MT> &vel, VecArray<uint, MT> &S2OParticleIndex);
/**
 * @brief resolve the penetration of rigidbody particles
 * 
 * @tparam MT 
 * @param size 
 * @param tempPosition 
 * @param sortedPositionPhase 
 * @param S2OParticleIndex 
 * @param cellStart 
 * @param cellEnd 
 */
template <MemType MT>
void callResolvePenetration(int size, VecArray<vec4r, MT> &tempPosition, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd);
/**
 * @brief resolve the penetration of different objects
 * 
 * @tparam MT 
 * @param size 
 * @param tempPosition 
 * @param sortedPositionPhase 
 * @param sortedObjectIdx 
 * @param S2OParticleIndex 
 * @param cellStart 
 * @param cellEnd 
 */
template <MemType MT>
void callResolvePenetrationDiffObj(int size, VecArray<vec4r, MT> &tempPosition, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<uint, MT> &sortedObjectIdx, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd);
/**
 * @brief integrate the position and velocity
 * 
 * @tparam MT 
 * @param size 
 * @param dt 
 * @param vtx 
 * @param vx 
 * @param vv 
 */
template <MemType MT>
void callIntegrate(int size, Real dt, VecArray<vec3r, MT> &vtx, VecArray<vec3r, MT> &vx, VecArray<vec4r, MT> &vv);
/**
 * @brief integrate the position and velocity
 * 
 * @tparam MT 
 * @param size 
 * @param dt 
 * @param vtxph 
 * @param vxph 
 * @param vv 
 */
template <MemType MT>
void callIntegrate(int size, Real dt, VecArray<vec4r, MT> &vtxph, VecArray<vec4r, MT> &vxph, VecArray<vec4r, MT> &vv);
/**
 * @brief integrate the position and velocity
 * 
 * @tparam MT 
 * @param size 
 * @param dt 
 * @param vtxph 
 * @param vxph 
 * @param vv 
 * @param vf 
 */
template <MemType MT>
void callIntegrate(int size, Real dt, VecArray<vec4r, MT> &vtxph, VecArray<vec4r, MT> &vxph, VecArray<vec4r, MT> &vv, VecArray<vec3r, MT> &vf);
/**
 * @brief integrate the position and velocity use the quaternion
 * 
 * @tparam MT 
 * @param size 
 * @param dt 
 * @param tq 
 * @param q 
 * @param w 
 * @param torque 
 * @param im 
 */
template <MemType MT>
void callIntegrateQuaternion(int size, Real dt, VecArray<quaternionf, MT> &tq, VecArray<quaternionf, MT> &q, VecArray<vec3r, MT> &w, VecArray<vec3r, MT> &torque, VecArray<mat3r, MT> &im);
/**
 * @brief merge the position and phase into one array
 * 
 * @tparam MT 
 * @param size 
 * @param start 
 * @param vxph 
 * @param vx 
 * @param vph 
 */
template <MemType MT>
void callMergePositionPhase(int size, int start, VecArray<vec4r, MT> &vxph, VecArray<vec3r, MT> &vx, VecArray<uint, MT> &vph);
/**
 * @brief update the hash value of the particles
 * 
 * @tparam MT 
 * @param size 
 * @param gridParticleHash 
 * @param S2OParticleIndex 
 * @param pos 
 */
template <MemType MT>
void callUpdateHash(int size, VecArray<uint, MT> &gridParticleHash, VecArray<uint, MT> &S2OParticleIndex, VecArray<vec3r, MT> &pos);
/**
 * @brief update the hash value of the particles
 * 
 * @tparam MT 
 * @param size 
 * @param gridParticleHash 
 * @param S2OParticleIndex 
 * @param pos 
 */
template <MemType MT>
void callUpdateHash(int size, VecArray<uint, MT> &gridParticleHash, VecArray<uint, MT> &S2OParticleIndex, VecArray<vec4r, MT> &pos);
// 
/**
 * @brief sort particles based on the hash value
 * 
 * @tparam MT 
 * @param size 
 * @param phash 
 * @param pidx 
 */
template <MemType MT>
void sortParticles(int size, VecArray<uint, MT> &phash, VecArray<uint, MT> &pidx);
/**
 * @brief obstacles collide
 * 
 * @param size 
 * @param vx 
 * @return template <MemType MT> 
 */
template <MemType MT>
void callCollideStaticObstacle(int size, VecArray<vec4r, MT> &vx);
/**
 * @brief spheres collide
 * 
 * @tparam MT 
 * @param size 
 * @param vx 
 */
template <MemType MT>
void callCollideStaticSphere(int size, VecArray<vec4r, MT> &vx);
/**
 * @brief update the position and velocity
 * 
 * @tparam MT 
 * @param size 
 * @param dt 
 * @param sortedPositionPhase 
 * @param pos 
 * @param vel 
 */
template <MemType MT>
void callUpdatePositionVelocity(int size, Real dt, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<vec3r, MT> &pos, VecArray<vec4r, MT> &vel);
/**
 * @brief update the quaternion and angular velocity
 * 
 * @tparam MT 
 * @param size 
 * @param dt 
 * @param vu 
 * @param vq 
 * @param vw 
 */
template <MemType MT>
void callUpdateQuaternionAndAngularVelocity(int size, Real dt, VecArray<quaternionf, MT> &vu, VecArray<quaternionf, MT> &vq, VecArray<vec3r, MT> &vw);
/**
 * @brief correct the position and velocity
 * 
 * @tparam MT 
 * @param size 
 * @param dx 
 * @param xph 
 */
template <MemType MT>
void callCorrectPositionAverage(int size, VecArray<vec4r, MT> &dx, VecArray<vec4r, MT> &xph);
/**
 * @brief update the collision between particles and triangles
 * 
 * @tparam MT 
 * @param size 
 * @param particlePositionPhase 
 * @param sortedTriangleInfo 
 * @param sortedTriangleCenter 
 * @param sortedTriangleVertex 
 * @param cellStart 
 * @param cellEnd 
 */
template <MemType MT>
void callUpdateParticleTriangleCollision(int size, VecArray<vec4r, MT> &particlePositionPhase, VecArray<TriangleInfo, MT> &sortedTriangleInfo, VecArray<vec4r, MT> &sortedTriangleCenter, VecArray<int4, MT> &sortedTriangleVertex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd);
/**
 * @brief update the collision between particles and triangles atomically
 * 
 * @tparam MT 
 * @param size 
 * @param newParticlePositionPhase 
 * @param particlePositionPhase 
 * @param sortedTriangleInfo 
 * @param sortedTriangleCenter 
 * @param sortedTriangleVertex 
 * @param cellStart 
 * @param cellEnd 
 */
template <MemType MT>
void callUpdateParticleTriangleCollisionAtom(int size, VecArray<vec4r, MT> &newParticlePositionPhase, VecArray<vec4r, MT> &particlePositionPhase, VecArray<TriangleInfo, MT> &sortedTriangleInfo, VecArray<vec4r, MT> &sortedTriangleCenter, VecArray<int4, MT> &sortedTriangleVertex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd);
/**
 * @brief update the triangle info
 * 
 * @tparam MT 
 * @param size 
 * @param triangleInfo 
 * @param triangleCenter 
 * @param triangleNormal 
 * @param particlePositionPhase 
 * @param triangleVertex 
 */
template <MemType MT>
void callUpdateTriangleInfo(int size, VecArray<TriangleInfo, MT> &triangleInfo, VecArray<vec4r, MT> &triangleCenter, VecArray<vec4r, MT> &triangleNormal, VecArray<vec4r, MT> &particlePositionPhase, VecArray<int4, MT> &triangleVertex);

PHYS_NAMESPACE_END