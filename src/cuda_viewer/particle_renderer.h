#pragma once

#include "cuda_viewer/shader.h"

extern const char *vertexShader;
extern const char *spherePixelShader;
extern const char *vertexShader2;

class ParticleRenderer {
public:
    ParticleRenderer(int i = 0) {
        use = false;
        if(i == 0)
            initShaders();
        else if (i == 1){
            initShaders2();
        }
#if !defined(__APPLE__) && !defined(MACOSX)
        glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
        glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
    }

    ~ParticleRenderer() {
        delete m_shader;
        m_pos = 0;
        unregisterVbo();
    }

    void registerVbo(){
        registerGLBufferObject(m_vbo, &m_cudaVbo);
        if(m_colorVbo) registerGLBufferObject(m_colorVbo, &m_cudaColorVbo);
    }

    void unregisterVbo(){
        unregisterGLBufferObject(m_cudaVbo);
        if(m_cudaColorVbo) unregisterGLBufferObject(m_cudaColorVbo);
    }


    void draw() {
        glEnable(GL_POINT_SPRITE_ARB);
        glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);

        m_shader->use();
        m_shader->setFloat("pointScale", float(m_winh) / tanf(m_fov * 0.5f * (float)3.1415926 / 180.0f));
        // m_shader->setVec2("screenSize", m_winw,m_winh);
        m_shader->setFloat("pointRadius", m_radius);

        glColor3f(1, 1, 1);
        drawParticles();

        glUseProgram(0);
        glDisable(GL_POINT_SPRITE_ARB);
    }


private:
    void initShaders() {
        m_shader = new Shader(vertexShader, spherePixelShader);
    };

    void initShaders2(){
        m_shader = new Shader(vertexShader2,spherePixelShader);
    };

    void drawParticles() {
        // std::cout<<"vbo="<<m_vbo<<",cvbo="<<m_colorVbo<<",size="<<m_size<<" "<<std::endl;
        if (!m_vbo) {
            glBegin(GL_POINTS);
            {
                int k = 0;
                for (int i = 0; i < m_size; ++i) {
                    glVertex3fv(&m_pos[k]);
                    k += 4;
                }
            }
            glEnd();
        }
        else {
            glBindBuffer(GL_ARRAY_BUFFER, m_vbo);////TODO: 4->3?
            // glVertexPointer(4, GL_FLOAT, 0, 0);
            glVertexPointer(3, GL_FLOAT, 0, 0);
            glEnableClientState(GL_VERTEX_ARRAY);

            if (m_colorVbo) {
                
                glBindBuffer(GL_ARRAY_BUFFER, m_colorVbo);
                // glColorPointer(4, GL_FLOAT, 0, 0);
                glColorPointer(3, GL_FLOAT, 0, 0);
                glEnableClientState(GL_COLOR_ARRAY);
            }

            glDrawArrays(GL_POINTS, 0, m_size);

            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);
        }
    }


public:
    Shader* m_shader;

    float* m_pos;
    //add den to control color
    float* den;
    int m_size = 0;

    float m_radius = 0.05f;
    float m_fov = 60.0f;

    int m_winw, m_winh;

    GLuint m_vbo=0;
    GLuint m_colorVbo=0;
    bool use;
    
    struct cudaGraphicsResource *m_cudaVbo=nullptr;
    struct cudaGraphicsResource *m_cudaColorVbo=nullptr;


};

