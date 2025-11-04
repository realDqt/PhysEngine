#include "object/gas_system.h"
#include "object/grid_gas.h"
#include "cuda_viewer/cuda_viewer.h"
#include "object/config.h"
#include "common/timer.h"
#include "object/gas_world.h"

using namespace physeng;

uint numParticles = 0;
uint3 gridSize;

StopWatchInterface *timer = NULL;

int main(int argc, char **argv) {
    if (argc < 9) {
        printf("lack params\n");
        return -1;
    }
    vec3r origin;
    origin.x = atof(argv[1]);
    origin.y = atof(argv[2]);
    origin.z = atof(argv[3]);

    Real vorticity = atof(argv[4]), diffusion = atof(argv[5]), buoyancy = atof(argv[6]), vcEpsilon = atof(argv[7]), decreaseDensity = atof(argv[8]);
    int scene = 0;

    cudaInit(argc, argv);

    GasWorld* gasWorld = new GasWorld();

    int gasIndex = gasWorld->initGasSystem(origin, vorticity, diffusion, buoyancy, vcEpsilon, decreaseDensity);

    printf("Gas Symbol:%d\n", gasIndex);
    if(gasIndex < 0){
        exit(0);
    }
}