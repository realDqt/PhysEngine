#pragma once
#include "cuda_viewer/shader.h"

extern const char *hairVertexShader;

extern const char* hairFragmentShaderMarschner;
extern const char *hairGeometryShader;
extern const char *hairTessControlShader;
extern const char *hairTessEvaluationShader;

extern const char* hairTessControlShader2;
extern const char* hairGeometryShader2;
extern const char* hairFragmentShader2;
class HairRenderer {
public:
    HairRenderer() {
        initShaders();
#if !defined(__APPLE__) && !defined(MACOSX)
        glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
        glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
        initBuffers();
        m_vertices=nullptr;
        m_tangents=nullptr;
        m_shadows = nullptr;
        // m_elements=nullptr;
        m_vertCnt=0;
        // m_elemCnt=0;
    }

    ~HairRenderer() {
        delete m_shader;
        freeBuffers();
        if(m_vertices) delete[] m_vertices;
        if(m_tangents) delete[] m_tangents;
        if (m_shadows) delete[] m_shadows;
        // if(m_elements) delete[] m_elements;
        unregisterVbo();
    }

    void registerVbo(){
        // registerGLBufferObject(m_vbo, &m_cudaVbo);
        if(m_vertVbo) registerGLBufferObject(m_vertVbo, &m_cudaVertVbo);
        if(m_tangVbo) registerGLBufferObject(m_tangVbo, &m_cudaTangVbo);
        if (m_shadowVbo) registerGLBufferObject(m_shadowVbo, &m_cudaShadowVbo);
        // if(m_elemVbo) registerGLBufferObject(m_elemVbo, &m_cudaElemVbo);
        // if(m_colorVbo) registerGLBufferObject(m_colorVbo, &m_cudaColorVbo);
    }

    void unregisterVbo(){
        // unregisterGLBufferObject(m_cudaVbo);
        if(m_cudaVertVbo) unregisterGLBufferObject(m_cudaVertVbo);
        if(m_cudaTangVbo) unregisterGLBufferObject(m_cudaTangVbo);
        if (m_cudaShadowVbo) unregisterGLBufferObject(m_cudaShadowVbo);
        // if(m_cudaElemVbo) unregisterGLBufferObject(m_cudaElemVbo);
        // if(m_cudaColorVbo) unregisterGLBufferObject(m_cudaColorVbo);
    }

    void beginInitBuffers(){
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);
    }

    void endInitBuffers(){
        glBindVertexArray(0);
    }

    ////// deprecated
    //void setHair(const float *vertices, const float *tangents, int vertCnt, const unsigned int *elements, int elemCnt){
    //    if(vertices && vertCnt>=0){
    //        if(m_vertices) delete[] m_vertices;
    //        m_vertices=new float[vertCnt*3];
    //        memcpy(m_vertices,vertices, vertCnt*3*sizeof(float)); 
    //        m_vertCnt=vertCnt;
    //        m_vertDirty=true;
    //    }else m_vertDirty=false;

    //    if(tangents && vertCnt>=0){
    //        if(m_tangents) delete[] m_tangents;
    //        m_tangents=new float[vertCnt*3];
    //        memcpy(m_tangents,tangents, vertCnt*3*sizeof(float)); 
    //        m_vertDirty=true;
    //    }else m_vertDirty=false;
    //    
    //    glBindVertexArray(m_vao);

    //    glBindBuffer(GL_ARRAY_BUFFER, m_vertVbo);
    //    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_vertices, GL_DYNAMIC_DRAW);
    //    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	   // glEnableVertexAttribArray(0);

    //    glBindBuffer(GL_ARRAY_BUFFER, m_tangVbo);
    //    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_tangents, GL_DYNAMIC_DRAW);
    //    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	   // glEnableVertexAttribArray(1);

    //    glBindVertexArray(0);
    //}

    void draw() {
        
        if(m_vertCnt==0) return;

        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
        //// for line
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_LINE_SMOOTH);
        // glLineWidth(2.0);

        m_shader->use();

        // bindBuffers();//// bind buffer manually
        glBindVertexArray(m_vao);

        glBindBuffer(GL_ARRAY_BUFFER, m_vertVbo);
        if(m_useCpuData)
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_vertices, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	    glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, m_tangVbo);
        if(m_useCpuData)
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_tangents, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	    glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, m_shadowVbo);
        if (m_useCpuData)
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * m_vertCnt, m_shadows, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(2);

        // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elemVbo);
        // if(m_useCpuData)
        //     glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*3*m_elemCnt, m_elements, GL_DYNAMIC_DRAW);


        // m_shader->setElementArray(m_elemVbo, m_elements, 3, m_elemCnt, true);
        m_shader->setFloat3("lightPos", make_float3(0.0,5.0,100.0));
        m_shader->setFloat3("viewPos", m_viewPos);
        m_shader->setFloat4("fixed_color", m_fixedColor);
        drawHairs();

        glUseProgram(0);
        glBindVertexArray(0);
        // std::cout<<"Hair draw end"<<std::endl;
    }

    void setColor(float4 color){
        m_fixedColor=color;
    }

    void setViewPos(float3 viewPos) {
        m_viewPos = viewPos;
    }

    void setPositionOffset(float4 off){
        m_positionOffset=off;
    }

    void setScale(float3 scale){
        m_scale=scale;
    }

private:
    void initShaders() {
//#define Test
#ifndef Test
        m_shader = new Shader(hairVertexShader, hairFragmentShaderMarschner, hairGeometryShader, hairTessControlShader, hairTessEvaluationShader);
#else
        m_shader = new Shader(hairVertexShader, hairFragmentShader2, hairGeometryShader2, hairTessControlShader2, hairTessEvaluationShader);
#endif // !Test

    };

    void initBuffers(){
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);
        glGenBuffers(1, &m_vertVbo);
        glGenBuffers(1, &m_tangVbo);
        glGenBuffers(1, &m_shadowVbo);
        // glGenBuffers(1, &m_elemVbo);
        glBindVertexArray(0);
    }

    void freeBuffers(){
        glDeleteVertexArrays(1, &m_vao);
        glDeleteBuffers(1, &m_vertVbo);
        glDeleteBuffers(1, &m_tangVbo);
        glDeleteBuffers(1, &m_shadowVbo);
        // glDeleteBuffers(1, &m_elemVbo);
    }
    
    // void bindBuffers(){
        
    //     glBindVertexArray(m_vao);

    //     m_shader->setVertexArray("position", m_vertVbo, m_vertices, 3, m_vertCnt, true);
    //     // m_shader->setVertexArray("tangent", m_tangVbo, m_tangents, 3, m_vertCnt, true);
    //     // m_shader->setElementArray(m_elemVbo, m_elements, 3, m_elemCnt, true);
    //     // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elemVbo);
    //     // // F_vbo is data
    //     // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned)*F_vbo.size(), F_vbo.data(), GL_DYNAMIC_DRAW);
    // }

    void drawHairs(bool solid = true){
        
        // glUniform....
        glPolygonMode(GL_FRONT_AND_BACK, solid ? GL_FILL : GL_LINE);

        // /* Avoid Z-buffer fighting between filled triangles & wireframe lines */
        // if (solid)
        // {
        //     glEnable(GL_POLYGON_OFFSET_FILL);
        //     glPolygonOffset(1.0, 1.0);
        // }
        // glDrawElements(GL_TRIANGLES, m_elemCnt*3, GL_UNSIGNED_INT, 0);

        // glBindVertexArray(VAO);
        glPatchParameteri(GL_PATCH_VERTICES, 4);
        glDrawArrays(GL_PATCHES, 0, m_vertCnt);

        glUseProgram(0);
        glDisable(GL_POLYGON_OFFSET_FILL);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

public:
    Shader* m_shader;

    float* m_vertices;
    float* m_tangents;
    float* m_shadows;
    // unsigned int* m_elements;

    int m_vertCnt;
    // int m_elemCnt;

    bool m_vertDirty;
    // bool m_elemDirty;
    bool m_useCpuData=false;

    int m_size;

    // float m_radius = 0.05f;
    // float m_fov = 60.0f;

    float4 m_fixedColor=make_float4(104/255.0f, 77/255.0f, 57/255.0f, 1.0);
    float4 m_positionOffset=make_float4(0,0,0,0);
    float3 m_scale=make_float3(1,1,1);
    float3 m_viewPos;

    int m_winw, m_winh;

    GLuint m_vao=0;
    GLuint m_vertVbo=0;
    GLuint m_tangVbo=0;
    GLuint m_shadowVbo = 0;
    // GLuint m_elemVbo=0;
    // GLuint m_colorVbo=0;

    struct cudaGraphicsResource *m_cudaVertVbo=nullptr;
    struct cudaGraphicsResource *m_cudaTangVbo=nullptr;
    struct cudaGraphicsResource* m_cudaShadowVbo = nullptr;
    // struct cudaGraphicsResource *m_cudaElemVbo=nullptr;
    // struct cudaGraphicsResource *m_cudaColorVbo=nullptr;

    // struct cudaGraphicsResource *m_cudaVbo;
    // struct cudaGraphicsResource *m_cudaColorVbo;
};

