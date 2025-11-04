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
    int scene = 0;

    cudaInit(argc, argv);

    GasWorld* gasWorld = new GasWorld();

    int gasIndex = gasWorld->initGasSystem("E:/tem/gas.txt");

    printf("Gas Symbol:%d\n", gasIndex);
    if(gasIndex < 0){
        exit(0);
    }
}