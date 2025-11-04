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

    int gasIndex = gasWorld->initGasSystem(make_vec3r(0.0f), 0.000002f, 0.000000f, 4.0f, 5.0f, 0.001f);

    printf("气体标识：%d\n", gasIndex);
    if(gasIndex < 0){
        exit(0);
    }
    gasWorld->getGas(gasIndex)->addGasSource(make_vec3r(-1.2f, -1.0f, 0.0f), 0.5f, make_vec3r(1.0f, 0.0f, 0.0f), 1.0f);
    gasWorld->setRenderData(gasIndex, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);


    CudaViewer viewer;
    viewer.init(argc, argv);
    viewer.bindFunctions();
    viewer.camera_trans[2] = -5;

    gasWorld->initViewer(gasIndex, viewer);

    sdkCreateTimer(&timer);

    int frame=0;

    viewer.updateCallback = [&]() {
        sdkStartTimer(&timer);
        {
            // PHY_PROFILE("grid system update");
            gasWorld->update(gasIndex);
            frame++;
        }

        gasWorld->updateViewer(gasIndex, viewer);
        sdkStopTimer(&timer);


        if(frame%100==0){
            float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
            LOG_OSTREAM_INFO<<"frame "<<frame<<", fps="<<ifps<<", time="<<sdkGetTimerValue(&timer) / 1000.f<<std::endl;
            BENCHMARK_REPORT();
        }

        return true;
    };

    viewer.closeCallback = [&](){
        sdkDeleteTimer(&timer);
        delete gasWorld;
        return true;
    };
    viewer.isPause=true;
    viewer.run();
}