#pragma once

#include <vector>
#include "common/allocator.h"
#include "math/math.h"

struct Mesh
{
    void AddMesh(const Mesh& m);
    void Normalize();

    uint32_t GetNumVertices() const { return uint32_t(m_positions.size()); }
    uint32_t GetNumFaces() const { return uint32_t(m_indices.size()) / 3; }


	void DuplicateVertex(uint32_t i);

    std::vector<vec3r> m_positions;
    std::vector<vec3r> m_normals;
    std::vector<vec2r> m_texcoords[2];
    std::vector<vec3r> m_colours;

    std::vector<int> m_indices;    
};

// create mesh from file
Mesh* ImportMeshFromObj(const char* path);
Mesh* ImportMeshFromPly(const char* path);
Mesh* ImportMeshFromBin(const char* path);

// just switches on filename
Mesh* ImportMesh(const char* path);
std::string GetFileType(const char* path);