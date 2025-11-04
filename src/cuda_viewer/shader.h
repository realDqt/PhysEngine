#pragma once
#include "cuda_viewer/cuda_gl_helper.h"

class Shader {
public:
    unsigned int m_shaderId;
    // constructor generates the shader on the fly
    // ------------------------------------------------------------------------
    Shader(const char* vShaderCode, const char* fShaderCode, const char* gShaderCode = nullptr, const char* tcShaderCode = nullptr, const char* teShaderCode = nullptr) {
        // 2. compile shaders
        unsigned int vertex, fragment, geometry, tessControl, tessEvaluation;
        // int success;
        // char infoLog[512];
        // vertex shader
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");

        // fragment Shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");

        // geometry Shader
        if (gShaderCode) {
            geometry = glCreateShader(GL_GEOMETRY_SHADER);
            glShaderSource(geometry, 1, &gShaderCode, NULL);
            glCompileShader(geometry);
            checkCompileErrors(geometry, "GEOMETRY");
        }

        if (tcShaderCode && teShaderCode) {
            tessControl = glCreateShader(GL_TESS_CONTROL_SHADER);
            glShaderSource(tessControl, 1, &tcShaderCode, NULL);
            glCompileShader(tessControl);
            checkCompileErrors(tessControl, "TESSELLATION CONTROL");

            tessEvaluation = glCreateShader(GL_TESS_EVALUATION_SHADER);
            glShaderSource(tessEvaluation, 1, &teShaderCode, NULL);
            glCompileShader(tessEvaluation);
            checkCompileErrors(tessEvaluation, "TESSELLATION EVALUATION");
        }


        // shader Program
        m_shaderId = glCreateProgram();
        glAttachShader(m_shaderId, vertex);
        glAttachShader(m_shaderId, fragment);
        if (gShaderCode) glAttachShader(m_shaderId, geometry);
        if (tcShaderCode && teShaderCode) {
            glAttachShader(m_shaderId, tessControl);
            glAttachShader(m_shaderId, tessEvaluation);
        }

        glLinkProgram(m_shaderId);
        checkCompileErrors(m_shaderId, "PROGRAM");

        // delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        if (gShaderCode) glDeleteShader(geometry);
        if (tcShaderCode && teShaderCode) {
            glDeleteShader(tessControl);
            glDeleteShader(tessEvaluation);
        }
    }

    void use() { glUseProgram(m_shaderId); }
    void setBool(const std::string& name, bool value) const {
        glUniform1i(glGetUniformLocation(m_shaderId, name.c_str()), (int)value);
    }
    void setInt(const std::string& name, int value) const {
        glUniform1i(glGetUniformLocation(m_shaderId, name.c_str()), value);
    }
    void setFloat(const std::string& name, float value) const {
        glUniform1f(glGetUniformLocation(m_shaderId, name.c_str()), value);
    }
    void setFloat4(const std::string& name, float4 value) const {
        // std::cout<<"set "<<name<<"="<<value.x<<","<<value.y<<","<<value.z<<","<<value.w<<" at "<<glGetUniformLocation(m_shaderId, name.c_str())<<std::endl;
        glUniform4f(glGetUniformLocation(m_shaderId, name.c_str()), value.x, value.y, value.z, value.w);
    }
    void setFloat3(const std::string& name, float3 value) const {
        // std::cout<<"set "<<name<<"="<<value.x<<","<<value.y<<","<<value.z<<" at "<<glGetUniformLocation(m_shaderId, name.c_str())<<std::endl;
        glUniform3f(glGetUniformLocation(m_shaderId, name.c_str()), value.x, value.y, value.z);
    }

    // void setVec2(const std::string& name, float value0, float value1) const {
    //     glUniform2f(glGetUniformLocation(m_shaderId, name.c_str()), value0, value1);
    // }
    bool isInUse() const {
        GLint currentProgram = 0;
        glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgram);
        return (currentProgram == (GLint)m_shaderId);
    }

    GLint setVertexArray(const std::string& name, GLuint bufferId, float* data, int step, int size, bool isDirty){
        std::cout<<"  shaderId "<<m_shaderId<<"name"<<name<<std::endl;
        GLint id = glGetAttribLocation(m_shaderId, name.c_str());
        std::cout<<"  get ID "<<id<<" "<<size<<std::endl;
        if (id < 0) return id;
        if (size == 0) {
            glDisableVertexAttribArray(id);
            return id;
        }
        glBindBuffer(GL_ARRAY_BUFFER, bufferId);
        if (isDirty) 
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*step*size, data, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(id, step, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(id);
        return id;
    }

    GLint setElementArray(GLuint bufferId, unsigned int* data, int step, int size, bool isDirty){
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferId);
        // if (dirty & MeshGL::DIRTY_FACE)
        if (isDirty) 
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*step*size, data, GL_DYNAMIC_DRAW);
        return 0;
    }

private:
    void checkCompileErrors(unsigned int shader, std::string type) {
        int success;
        char infoLog[1024];
        if (type != "PROGRAM") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        if (!success) {
            exit(-1);
        }
    }
};