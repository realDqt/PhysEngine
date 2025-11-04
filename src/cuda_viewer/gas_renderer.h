#pragma once

#include "cuda_viewer/shader.h"

extern const char* vertexShader;
extern const char* spherePixelShader;
extern const char* vertexShader2;

class Convexcomp
{
private:
    const float3& p0, & up;
public:
    Convexcomp(const float3& p0, const float3& up) : p0(p0), up(up) {}

    bool operator()(const float3& a, const float3& b) const
    {
        float3 va = a - p0, vb = b - p0;
        return dot(up, cross(va,vb)) >= 0;
    }
};

class GasRenderer {
public:
    GasRenderer(int i = 0) {
        // cube vertices
        GLfloat cv[][3] = {
            {1.0f, 1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f}, {-1.0f, -1.0f, 1.0f}, {1.0f, -1.0f, 1.0f},
            {1.0f, 1.0f, -1.0f}, {-1.0f, 1.0f, -1.0f}, {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}
        };

        GLfloat ce[12][2][3] = {
            {{1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},
            {{-1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},
            {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},
            {{1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},

            {{1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
            {{-1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
            {{-1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
            {{1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},

            {{-1.0f, 1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
            {{-1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
            {{-1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
            {{-1.0f, 1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}}
        };

        memcpy(_cubeVertices, cv, sizeof(_cubeVertices));
        memcpy(_cubeEdges, ce, sizeof(_cubeEdges));

        initGL();
    }

    ~GasRenderer() {
    }

    void fillTexture(uint3 gridSize, unsigned char* textureData) {
        glActiveTextureARB(GL_TEXTURE0_ARB);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, gridSize.x, gridSize.y, gridSize.z, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData);
    }


    void draw() {
        GLfloat(*cv)[3] = _cubeVertices;
        int i;
        for (i = 0; i < 8; i++) {
            float x = cv[i][0] + viewDir.x;
            float y = cv[i][1] + viewDir.y;
            float z = cv[i][2] + viewDir.z;
            if ((x >= -1.0f) && (x <= 1.0f)
                && (y >= -1.0f) && (y <= 1.0f)
                && (z >= -1.0f) && (z <= 1.0f))
            {
                break;
            }
        }
        assert(i != 8);

        float SLICE_NUM = 64.0f;

        float d0 = -dot(viewDir, make_float3(cv[i][0], cv[i][1], cv[i][2]));
        float dStep = 2 * d0 / SLICE_NUM;
        int n = 0;
        for (float d = -d0; d < d0; d += dStep) {
            // IntersectEdges returns the intersection points of all cube edges with
            // the given plane that lie within the cube
            std::vector<float3> pt = intersectEdges(viewDir.x, viewDir.y, viewDir.z, d);

            if (pt.size() > 2) {
                // sort points to get a convex polygon
                std::sort(pt.begin() + 1, pt.end(), Convexcomp(pt[0], viewDir));

                glEnable(GL_TEXTURE_3D);
                glBegin(GL_POLYGON);
                for (i = 0; i < pt.size(); i++) {
                    glColor3f(1.0, 1.0, 1.0);
                    glTexCoord3d((pt[i].x + 1.0) / 2.0, (pt[i].y + 1.0) / 2.0, (pt[i].z + 1.0) / 2.0);//FIXME
                    glVertex3f(pt[i].x * gridLength.x / 2.0, pt[i].y * gridLength.y / 2.0, pt[i].z * gridLength.z / 2.0);
                }
                glEnd();

            }
            n++;
        }

        glDisable(GL_TEXTURE_3D);
    }

    void setViewDir(float3 viewdir) {
        this->viewDir = normalize(viewdir);
    }

    void setLightDir(float3 lightdir) {
        this->lightDir = normalize(lightdir);
    }

    void setGridSize(uint3 gridSize, float cellLength) {
        this->gridLength = make_float3(gridSize) * cellLength;
    }


private:
    float3 viewDir;
    float3 lightDir;
    float3 gridLength = make_float3(1.0f);

    unsigned int _hTexture;

    GLfloat _cubeVertices[8][3];
    GLfloat _cubeEdges[12][2][3];

    void initGL()
    {
        glEnable(GL_TEXTURE_3D);
        glDisable(GL_DEPTH_TEST);
        glCullFace(GL_FRONT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


        glGenTextures(2, &_hTexture);

        glActiveTextureARB(GL_TEXTURE0_ARB);
        glBindTexture(GL_TEXTURE_3D, _hTexture);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    }

    std::vector<float3> intersectEdges(float A, float B, float C, float D)
    {
        float t;
        float3 p;
        std::vector<float3> res;
        GLfloat(*edges)[2][3] = _cubeEdges;


        for (int i = 0; i < 12; i++) {
            t = -(A * edges[i][0][0] + B * edges[i][0][1] + C * edges[i][0][2] + D)
                / (A * edges[i][1][0] + B * edges[i][1][1] + C * edges[i][1][2]);
            if ((t > 0) && (t < 2)) {
                p.x = edges[i][0][0] + edges[i][1][0] * t;
                p.y = edges[i][0][1] + edges[i][1][1] * t;
                p.z = edges[i][0][2] + edges[i][1][2] * t;
                res.push_back(p);
            }
        }

        return res;
    }

public:
};

