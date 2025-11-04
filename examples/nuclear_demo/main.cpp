#include "object/nuclear_system.h"
#include "object/particle_fluid.h"
#include "cuda_viewer/cuda_viewer.h"
#include "object/gas_world.h"
#include "common/timer.h"

using namespace physeng;
//#define GRID_SIZE 64
uint numParticles = 0;
uint3 gridSize;

StopWatchInterface *timer = NULL;

int main(int argc, char **argv) {
    int scene = 0;
    if (argc < 3) {
        printf("Lack parameters!");
        logger.Log(LogType::Error, "Lack parameters!");
        return -1;
    }
    cudaInit(argc, argv);
    
    //total number of the grid
    GasWorld* gasWorld = new GasWorld();
    int gasIndex = gasWorld->initNuclearSystem(make_vec3r(0.0f), 5);

    //set wind
    gasWorld->setWind(atof(argv[2]), make_vec3r(1, 0, 0));


    LOG_OSTREAM_DEBUG<<"finish add nuclear"<<std::endl;
    CudaViewer viewer;
    viewer.init(argc, argv);
    viewer.bindFunctions();

    viewer.camera_trans[2] = -60;

    gasWorld->initViewer(gasIndex, viewer);

    sdkCreateTimer(&timer);
    viewer.keyCallback = [&](unsigned int key, int, int) {
        switch (key)
        {
        case 's':
            //psystem->start_pollution = ! psystem->start_pollution;
            break;
        }
        LOG_OSTREAM_DEBUG << "Key down " << (char)key << std::endl;
        return false;
    };

    int frame=0;

    viewer.updateCallback = [&]() {        
        Real dt=0.016;


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