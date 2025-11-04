#pragma once

#include "cuda_viewer/particle_renderer.h"
#include "cuda_viewer/mesh_renderer.h"
#include "cuda_viewer/hair_renderer.h"
#include "cuda_viewer/gas_renderer.h"
#include "common/timer.h"

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <functional>

#include <chrono>
#include <thread>


void display();
void reshape(int,int);
void mouse(int,int,int,int);
void motion(int, int);
void key(unsigned char, int, int);
void idle();
void cleanup();

class CudaViewer{
  public:
    CudaViewer(){}
    ~CudaViewer(){
        if(prender) delete prender;
        if(nrender) delete nrender;
        if(mrender) delete mrender;
        if(m2render) delete m2render;
        if(hrender) delete hrender;
        if (grender) delete grender;
    }

    void init(int argc, char **argv);

    void initGL(int *argc, char **argv)
    {
        glutInit(argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE | GLUT_MULTISAMPLE);
        glutInitWindowSize(width, height);
        glutCreateWindow("CUDA Particles");
 
        if (!isGLVersionSupported(2,0) ||
            !areGLExtensionsSupported("GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
        {
            fprintf(stderr, "Required OpenGL extensions missing.");
            exit(EXIT_FAILURE);
        }

    #if defined (WIN32)

        if (wglewIsSupported("WGL_EXT_swap_control"))
        {
            // disable vertical sync
            wglSwapIntervalEXT(0);
        }

    #endif
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.25, 0.25, 0.25, 1.0);
        glutReportErrors();
    }

    void initCuda(int argc, char **argv){
        int devID;
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);
        if (devID < 0){
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void bindFunctions(){

        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutKeyboardFunc(key);
        // glutSpecialFunc(special);
        glutIdleFunc(idle);
        glutCloseFunc(cleanup);

        // glutMainLoop();
    }

    void run(){
        glutMainLoop();
    }

    static unsigned int createVbo(unsigned int size){
        GLuint vbo;
        // std::cout<<"create vbo size"<<size<<std::endl;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return vbo;
    }

    void setWorldBoundary(float3 wmin, float3 wmax){
        world_min_x = wmin.x;
        world_min_y = wmin.y;
        world_min_z = wmin.z;
        world_max_x = wmax.x;
        world_max_y = wmax.y;
        world_max_z = wmax.z;
    }

    void useNormalRenderer() {
        nrender = new ParticleRenderer(1);
    }

    void useParticleRenderer() {
        prender = new ParticleRenderer();
    }

    void useGasRenderer() {
        grender = new GasRenderer();
    }

    void useWorldBoundary() {
        renderWorldBoundary = true;
    }

    //// callBacks

    //// camera
  public:
    unsigned int width = 640;
    unsigned int height = 480;

    // view params
    int mouse_x, mouse_y;
    int buttonState = 0;
    float camera_trans[3] = {0, 0, -3};
    float camera_rot[3]   = {0, 0, 0};
    float camera_trans_lag[3] = {0, 0, -3};
    float camera_rot_lag[3] = {0, 0, 0};
    const float inertia = 0.1f;
    // ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

    float world_min_x = -10;
    float world_min_y = 0;
    float world_min_z = -10;
    float world_max_x = 10;
    float world_max_y = 10;
    float world_max_z = 10;

    bool isPause = false;
    bool advanceOneStep = false;
    bool renderWorldBoundary = false;
    // bool demoMode = false;
    // int idleCounter = 0;
    // int demoCounter = 0;
    // const int idleDelay = 2000;
    float modelView[16];
    
    ParticleRenderer* prender;
    ParticleRenderer* nrender;
    MeshRenderer* mrender;
    MeshRenderer* m2render;
    HairRenderer* hrender;
    GasRenderer* grender;

    int mouseButton[3]={0,0,0};
  public:
    // std::function<void()> drawSimulationMenuCallback; //!< customized menu callback
    std::function<bool(unsigned int, int, int)> keyCallback; //!< customized key callback
    std::function<bool()> drawCallback; //!< customized viewer draw callback
    std::function<bool()> updateCallback; //!< customized viewer update callback
    std::function<bool()> closeCallback; //!< customized viewer close callback
    //test
    float3 startPos;
    int voxelPerSide;
    float length;

};