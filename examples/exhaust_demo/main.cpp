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
        printf("Lack parameters!");
        logger.Log(LogType::Error, "缺少环境参数");
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

    gasWorld->setRenderData(gasIndex, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);

    gasWorld->setExhaust(gasIndex);

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