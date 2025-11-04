#include "object/gas_system.h"
#include "object/grid_gas.h"
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

    int gasIndex = gasWorld->initGasSystem(make_vec3r(0.0f), 0.000002f, 0.000000f, 4.0f, 5.0f, 0.001f);

    printf("气体标识：%d\n", gasIndex);
    if(gasIndex < 0){
        exit(0);
    }

    gasWorld->getGas(gasIndex)->addGasSource(make_vec3r(-1.2f, -1.0f, 0.0f), 0.5f, make_vec3r(1.0f, 0.0f, 0.0f), 1.0f);
    gasWorld->getGas(gasIndex)->addBox(make_vec3r(0, -0.7, 0), make_vec3r(0.8, 1.5, 0.8));
    gasWorld->setRenderData(gasIndex, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);


    int frame=0;
    while (true) {
        gasWorld->update(gasIndex);
        frame++;
        std::cout << "current frame = " << frame << std::endl;
    }
}