#include "object/fluid_system.h"
#include "object/particle_fluid.h"
#include "cuda_viewer/cuda_viewer.h"
#include "object/config.h"
#include "common/timer.h"
#include "common/logger.h"
#include "object/fluid_world.h"
using namespace physeng;
#define GRID_SIZE 64
uint numParticles = 0;
uint3 gridSize;

StopWatchInterface *timer = NULL;

int main(int argc, char** argv) {
    cudaInit(argc, argv);

    FluidWorld* fluidWorld = new FluidWorld(make_vec3r(-15, 0, -15), make_vec3r(15, 25, 15));
    int fluidIndex = fluidWorld->initFluidSystem("E:/tem/fluid.txt");

    printf("Fluid Symbol:%d\n", fluidIndex);
    if (fluidIndex < 0) {
        exit(0);
    }
}