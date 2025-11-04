#pragma once
#include "cuda_viewer/shader.h"

extern const char *meshVertexShader;
extern const char *meshFragmentShader;

extern const float g_sphere_vertices[];
extern const float g_sphere_normals[];
extern const unsigned int g_sphere_faces[];

extern const float g_box_vertices[];
extern const float g_box_normals[];
extern const unsigned int g_box_faces[];

extern const float g_cylinder_vertices[];
extern const float g_cylinder_normals[];
extern const unsigned int g_cylinder_faces[];

extern const float g_boxquad_vertices[];
extern const float g_boxquad_normals[];
extern const unsigned int g_boxquad_faces[];

class MeshRenderer {
public:
    enum MeshType{ Triangle=0, Quad=1 };

    MeshRenderer() {
        initShaders();
#if !defined(__APPLE__) && !defined(MACOSX)
        glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
        glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
        initBuffers();
        m_vertices=nullptr;
        m_normals=nullptr;
        m_elements=nullptr;
        m_vertCnt=0;
        m_elemCnt=0;
    }

    ~MeshRenderer() {
        delete m_shader;
        freeBuffers();
        if(m_vertices) delete[] m_vertices;
        if(m_normals) delete[] m_normals;
        if(m_elements) delete[] m_elements;
        unregisterVbo();
    }

    void registerVbo(){
        // registerGLBufferObject(m_vbo, &m_cudaVbo);
        if(m_vertVbo) registerGLBufferObject(m_vertVbo, &m_cudaVertVbo);
        if(m_normVbo) registerGLBufferObject(m_normVbo, &m_cudaNormVbo);
        if(m_elemVbo) registerGLBufferObject(m_elemVbo, &m_cudaElemVbo);
        if(m_colorVbo) registerGLBufferObject(m_colorVbo, &m_cudaColorVbo);
    }

    void unregisterVbo(){
        // unregisterGLBufferObject(m_cudaVbo);
        // if(m_cudaColorVbo) unregisterGLBufferObject(m_cudaColorVbo);
        if(m_cudaVertVbo) unregisterGLBufferObject(m_cudaVertVbo);
        if(m_cudaNormVbo) unregisterGLBufferObject(m_cudaNormVbo);
        if(m_cudaElemVbo) unregisterGLBufferObject(m_cudaElemVbo);
        if(m_cudaColorVbo) unregisterGLBufferObject(m_cudaColorVbo);
    }

    void beginInitBuffers(){
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);
    }

    void endInitBuffers(){
        glBindVertexArray(0);
    }

    //// deprecated
    void setMesh(const float *vertices, const float *normals, int vertCnt, const unsigned int *elements, int elemCnt){
        if(vertices && vertCnt>=0){
            if(m_vertices) delete[] m_vertices;
            m_vertices=new float[vertCnt*3];
            memcpy(m_vertices,vertices, vertCnt*3*sizeof(float)); 
            m_vertCnt=vertCnt;
            m_vertDirty=true;
        }else m_vertDirty=false;

        if(normals && vertCnt>=0){
            if(m_normals) delete[] m_normals;
            m_normals=new float[vertCnt*3];
            memcpy(m_normals,normals, vertCnt*3*sizeof(float)); 
            m_vertDirty=true;
        }else m_vertDirty=false;
        
        if(elements && elemCnt>=0){
            if(m_elements) delete[] m_elements;
            m_elements=new unsigned int[elemCnt*m_vertPerElem];
            memcpy(m_elements,elements, elemCnt*m_vertPerElem*sizeof(unsigned int)); 
            m_elemCnt=elemCnt;
            m_elemDirty=true;
        }else m_elemDirty=false;

        glBindVertexArray(m_vao);

        glBindBuffer(GL_ARRAY_BUFFER, m_vertVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_vertices, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	    glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, m_normVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_normals, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	    glEnableVertexAttribArray(1);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elemVbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*m_vertPerElem*m_elemCnt, m_elements, GL_DYNAMIC_DRAW);

        glBindVertexArray(0);
    }

    void draw() {
        
        if(m_elemCnt==0) return;

        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
        m_shader->use();

        // bindBuffers();//// bind buffer manually
        glBindVertexArray(m_vao);

        glBindBuffer(GL_ARRAY_BUFFER, m_vertVbo);
        if(m_useCpuData)
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_vertices, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	    glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, m_normVbo);
        if(m_useCpuData)
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_normals, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	    glEnableVertexAttribArray(1);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elemVbo);
        if(m_useCpuData)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*m_vertPerElem*m_elemCnt, m_elements, GL_DYNAMIC_DRAW);


        // m_shader->setElementArray(m_elemVbo, m_elements, 3, m_elemCnt, true);
        m_shader->setFloat4("fixed_color", m_fixedColor);
        m_shader->setFloat4("position_offset", m_positionOffset);
        m_shader->setFloat3("scale", m_scale);
        drawMeshes();

        glUseProgram(0);
        glBindVertexArray(0);
        // std::cout<<"mesh draw end"<<std::endl;
    }

    void setColor(float4 color){
        m_fixedColor=color;
    }

    void setPositionOffset(float4 off){
        m_positionOffset=off;
    }

    void setScale(float3 scale){
        m_scale=scale;
    }

    void setMeshType(MeshType mt){
        m_meshType=mt;
        m_vertPerElem=mt+3;
    }

    MeshType getMeshType(){return m_meshType;}

private:
    void initShaders() {
        m_shader = new Shader(meshVertexShader, meshFragmentShader);
    };

    void initBuffers(){
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);
        glGenBuffers(1, &m_vertVbo);
        glGenBuffers(1, &m_normVbo);
        glGenBuffers(1, &m_elemVbo);
        glBindVertexArray(0);
    }

    void freeBuffers(){
        glDeleteVertexArrays(1, &m_vao);
        glDeleteBuffers(1, &m_vertVbo);
        glDeleteBuffers(1, &m_normVbo);
        glDeleteBuffers(1, &m_elemVbo);
    }
    
    void bindBuffers(){
        
        glBindVertexArray(m_vao);

        m_shader->setVertexArray("position", m_vertVbo, m_vertices, 3, m_vertCnt, true);
        m_shader->setVertexArray("normal", m_normVbo, m_normals, 3, m_vertCnt, true);
        m_shader->setElementArray(m_elemVbo, m_elements, m_vertPerElem, m_elemCnt, true);
        // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elemVbo);
        // // F_vbo is data
        // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned)*F_vbo.size(), F_vbo.data(), GL_DYNAMIC_DRAW);
    }

    void drawMeshes(bool solid = true){
        
        // glUniform....
        glPolygonMode(GL_FRONT_AND_BACK, solid ? GL_FILL : GL_LINE);

        /* Avoid Z-buffer fighting between filled triangles & wireframe lines */
        if (solid)
        {
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.0, 1.0);
        }
        // glDrawElements(GL_TRIANGLES, m_elemCnt*3, GL_UNSIGNED_INT, 0);
        // glDrawArrays(GL_TRIANGLES, 0, m_elemCnt*3);
        if(m_meshType==MeshType::Triangle)
        glDrawElements(GL_TRIANGLES, m_elemCnt*3, GL_UNSIGNED_INT, 0);
        else
        glDrawElements(GL_QUADS, m_elemCnt*4, GL_UNSIGNED_INT, 0);

        glUseProgram(0);
        glDisable(GL_POLYGON_OFFSET_FILL);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

public:
    Shader* m_shader;

    float* m_vertices;
    float* m_normals;
    unsigned int* m_elements;

    int m_vertCnt;
    int m_elemCnt;

    bool m_vertDirty;
    bool m_elemDirty;
    bool m_useCpuData=false;

    int m_size;

    // float m_radius = 0.05f;
    // float m_fov = 60.0f;

    float4 m_fixedColor=make_float4(1,0,0,0);
    float4 m_positionOffset=make_float4(0,0,0,0);
    float3 m_scale=make_float3(1,1,1);

    int m_winw, m_winh;

    GLuint m_vao=0;
    GLuint m_vertVbo=0;
    GLuint m_normVbo=0;
    GLuint m_elemVbo=0;
    GLuint m_colorVbo=0;

    struct cudaGraphicsResource *m_cudaVertVbo=nullptr;
    struct cudaGraphicsResource *m_cudaNormVbo=nullptr;
    struct cudaGraphicsResource *m_cudaElemVbo=nullptr;
    struct cudaGraphicsResource *m_cudaColorVbo=nullptr;
private:
    MeshType m_meshType=MeshType::Triangle;
    int m_vertPerElem=3;
    // struct cudaGraphicsResource *m_cudaVbo;
    // struct cudaGraphicsResource *m_cudaColorVbo;
};

