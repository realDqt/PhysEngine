#pragma once
#include "fluid_system.h"
#include "gas_system.h"
#include "config.h"
#include <vector>
#include "cuda_viewer/cuda_viewer.h"

#define MAX_PARTICLE 200000
#define FLUID_GRID_SIZE 64
class FluidWorld
{
private:
    std::vector<FluidSystem*> fluidArray;
    DEFINE_MEMBER_SET_GET(Real, dt, Dt);
    DEFINE_MEMBER_SET_GET(vec3r, origin, Origin);
    DEFINE_MEMBER_SET_GET(vec3r, worldMin, WorldMin);
    DEFINE_MEMBER_SET_GET(vec3r, worldMax, WorldMax);
public:
    FluidWorld(vec3r worldMin, vec3r worldMax);
    ~FluidWorld();

    int initFluidSystem(vec3r center, vec3r scale, Real kvorticity, Real kviscosity, bool useFoam = false);
    int initFluidSystem(const char* config_file, bool useFoam = false);
    void completeInit(int index);
    void setWorldBoundary(vec3r worldMin, vec3r worldMax);
    FluidSystem* getFluid(int index);
    void updateColumn(int idx, Real r, vec3r x);
    void addCube(vec3r center, vec3r scale);
    void addSandpile();

    void initViewer(int index, CudaViewer& viewer);
    void updateViewer(int index, CudaViewer& viewer);
    void update(int index);
};