#include "cuda_viewer/cuda_viewer.h"
#include "object/particle_system_util.h"
static CudaViewer * __viewer;

void CudaViewer::init(int argc, char **argv){
    initGL(&argc, argv);
    initCuda(argc, argv);
    __viewer=this;
    prender=nullptr;
    nrender = nullptr;
    mrender=new MeshRenderer();
    m2render=new MeshRenderer();
    m2render->m_useCpuData=true;
    hrender=new HairRenderer();
    //grender = new GasRenderer();
    grender = nullptr;
}

void display()
{
    PHY_PROFILE("mainLoop");
    // std::cout<<"begin display"<<std::endl;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for (int c = 0; c < 3; ++c)
    {
        __viewer->camera_trans_lag[c] += (__viewer->camera_trans[c] - __viewer->camera_trans_lag[c]) * __viewer->inertia;
        __viewer->camera_rot_lag[c] += (__viewer->camera_rot[c] - __viewer->camera_rot_lag[c]) * __viewer->inertia;
    }

    glTranslatef(__viewer->camera_trans_lag[0], __viewer->camera_trans_lag[1], __viewer->camera_trans_lag[2]);
    glRotatef(__viewer->camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(__viewer->camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, __viewer->modelView);

    // cube

    // glPushMatrix();
    // glTranslatef(
    //     0.5f*(__viewer->world_min_x+__viewer->world_max_x),
    //     0.5f*(__viewer->world_min_y+__viewer->world_max_y),
    //     0.5f*(__viewer->world_min_z+__viewer->world_max_z));
    // glScalef(0.5f*(__viewer->world_max_x-__viewer->world_min_x),
    //     0.5f*(__viewer->world_max_y-__viewer->world_min_y),
    //     0.5f*(__viewer->world_max_z-__viewer->world_min_z));
    // glColor3f(1.0, 1.0, 1.0);
    // glutWireCube(2.0);
    // glPopMatrix();
    
    if (__viewer->renderWorldBoundary) {
        //// bounding box
        glPointSize(3.0);
        glColor3f(1.0, 1.0, 1.0);
        glBegin(GL_LINES);
        //// x line
        glVertex3d(__viewer->world_min_x, __viewer->world_min_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_min_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_min_x, __viewer->world_max_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_max_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_min_x, __viewer->world_min_y, __viewer->world_max_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_min_y, __viewer->world_max_z);
        glVertex3d(__viewer->world_min_x, __viewer->world_max_y, __viewer->world_max_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_max_y, __viewer->world_max_z);
        //// y line
        glVertex3d(__viewer->world_min_x, __viewer->world_min_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_min_x, __viewer->world_max_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_min_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_max_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_min_x, __viewer->world_min_y, __viewer->world_max_z);
        glVertex3d(__viewer->world_min_x, __viewer->world_max_y, __viewer->world_max_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_min_y, __viewer->world_max_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_max_y, __viewer->world_max_z);

        //// z line
        glVertex3d(__viewer->world_min_x, __viewer->world_min_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_min_x, __viewer->world_min_y, __viewer->world_max_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_min_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_min_y, __viewer->world_max_z);
        glVertex3d(__viewer->world_min_x, __viewer->world_max_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_min_x, __viewer->world_max_y, __viewer->world_max_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_max_y, __viewer->world_min_z);
        glVertex3d(__viewer->world_max_x, __viewer->world_max_y, __viewer->world_max_z);
        glEnd();
    }


    // coordinate
    glPointSize(3.0);
    glColor3f(1.0,0.0,0.0); 
    glBegin(GL_LINES);
    glVertex3d(0, 0, 0);
    glVertex3d(1, 0, 0);
    glEnd();
    glColor3f(0.0,1.0,0.0); 
    glBegin(GL_LINES);
    glVertex3d(0, 0, 0);
    glVertex3d(0, 1, 0);
    glEnd();
    glColor3f(0.0,0.0,1.0); 
    glBegin(GL_LINES);
    glVertex3d(0, 0, 0);
    glVertex3d(0, 0, 1);
    glEnd();
    //add for nuclear rendering
    
    if(__viewer->nrender && __viewer->nrender->use == true){
        glPointSize(3.0);
        glColor3f(1.0,1.0,0.0);
        glBegin(GL_QUADS);
        glVertex3f(2.5, 0, __viewer->world_min_z);
        glVertex3f(2.5, 0, __viewer->world_max_z);
        glVertex3f(__viewer->world_max_x,(__viewer->world_max_x - 2.5) / pow(3,0.5), __viewer->world_max_z);
        glVertex3f(__viewer->world_max_x,  (__viewer->world_max_x - 2.5) / pow(3,0.5), __viewer->world_min_z);
        glEnd();
    }

    //// customized draw callback
    if (!__viewer->isPause || __viewer->advanceOneStep){
        if(__viewer->updateCallback) __viewer->updateCallback();
        __viewer->advanceOneStep=false;
    }

    if(__viewer->drawCallback) __viewer->drawCallback();
    

    //// particle renderer
    // std::cout<<"prender"<<std::endl;

    
    if(__viewer->prender) __viewer->prender->draw();

    if(__viewer->nrender) __viewer->nrender->draw();
    //// mesh renderer
    // std::cout<<"mrender"<<std::endl;
    if(__viewer->mrender) __viewer->mrender->draw();
    if(__viewer->m2render) __viewer->m2render->draw();
    if (__viewer->hrender) {
        PHY_PROFILE("render");
        mat3r rotateMat;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                rotateMat[i * 3 + j] = __viewer->modelView[j * 4 + i];
        vec3r transform = make_vec3r(__viewer->modelView[12], __viewer->modelView[13], __viewer->modelView[14]);
        rotateMat = rotateMat.inverse();
        vec3r cameraPos = rotateMat * (make_vec3r(0, 0, 0) - transform);
        __viewer->hrender->setViewPos(cameraPos);
        __viewer->hrender->draw();
    }

    if (__viewer->grender) {
        __viewer->grender->setViewDir(-make_vec3r(__viewer->modelView[2], __viewer->modelView[6], __viewer->modelView[10]));
        __viewer->grender->draw();
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    // auto duration = start.time_since_epoch();

    auto sleepTime = std::chrono::milliseconds(int(0.016f*1000)) - std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    auto durationTime=std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    // std::this_thread::sleep_for(sleepTime);

    glutSwapBuffers();
    glutReportErrors();
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 400.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (__viewer->prender) {
        __viewer->prender->m_winw = w;
        __viewer->prender->m_winh = h;
    }

    // __viewer->nrender->m_winw = w;
    // __viewer->nrender->m_winh = h;

}

void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
    {
        __viewer->mouseButton[button]=1;
        __viewer->buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        __viewer->mouseButton[button]=0;
        __viewer->buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT)
    {
        __viewer->buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL)
    {
        __viewer->buttonState = 3;
    }

    __viewer->mouse_x = x;
    __viewer->mouse_y = y;

    // __viewer->demoMode = false;
    // __viewer->idleCounter = 0;

    glutPostRedisplay();
}

void motion(int x, int y){
    float dx, dy;
    dx = (float)(x - __viewer->mouse_x);
    dy = (float)(y - __viewer->mouse_y);

    if (__viewer->mouseButton[2]==1)
    {
        // right = zoom
        __viewer->camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(__viewer->camera_trans[2]);
    }
    else if (__viewer->mouseButton[1]==1)
    {
        // middle = translate
        __viewer->camera_trans[0] += dx / 100.0f;
        __viewer->camera_trans[1] -= dy / 100.0f;
    }
    else if (__viewer->mouseButton[0]==1)
    {
        // left = rotate
        __viewer->camera_rot[0] += dy / 5.0f;
        __viewer->camera_rot[1] += dx / 5.0f;
    }

    __viewer->mouse_x = x;
    __viewer->mouse_y = y;
}



// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case ' ':
            __viewer->isPause = !__viewer->isPause;
            break;
        case 'a':
            __viewer->advanceOneStep = true;
            break;
        case '\033':
        case 'q':
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
        default:
            if(__viewer->keyCallback) __viewer->keyCallback(key, 0, 0);
            break;
    }

    // __viewer->demoMode = false;
    // __viewer->idleCounter = 0;
    glutPostRedisplay();
}

void idle(void){
    // if ((__viewer->idleCounter++ > __viewer->idleDelay) && (__viewer->demoMode==false)){
    //     __viewer->demoMode = true;
    //     printf("Entering demo mode\n");
    // }

    // if (__viewer->demoMode)
    // {
    //     __viewer->camera_rot[1] += 0.1f;

    //     if (__viewer->demoCounter++ > 1000)
    //     {
    //         // ballr = 10 + (rand() % 10);
    //         // addSphere();
    //         __viewer->demoCounter = 0;
    //     }
    // }

    glutPostRedisplay();
}

void cleanup(){
    if(__viewer->closeCallback) __viewer->closeCallback();
    // if (psystem) delete psystem;
    return;
}