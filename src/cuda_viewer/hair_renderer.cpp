#define STRINGIFY(A) #A

// TODO:UNIMPLEMENTED

const char* hairVertexShader = STRINGIFY(
    #version 410 compatibility \n

    in vec3 position;
    in vec3 tangent;
    in float shadowCoef;

    out vec4 vsColor;
    out vec3 vsTangent;
    out float vsShadowCoef;

    uniform vec4 fixed_color;

    void main()
    {
        gl_Position = vec4(position, 1.0);
        vsColor = fixed_color;
        //vsColor = vec4(0.1, 0.1, 0.1, 1.0);
        vsTangent = tangent;
        vsShadowCoef = shadowCoef;            
    }
);

const char* hairTessControlShader = STRINGIFY(
    #version 410 compatibility \n
    layout(vertices = 4) out;

    in vec4 vsColor[];
    in vec3 vsTangent[];
    in float vsShadowCoef[];

    out vec4 tcsColor[];
    out vec3 tcsTangent[];
    out float tcsShadowCoef[];

    void main()
    {
        // Pass along the vertex position unmodified
        gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
        tcsColor[gl_InvocationID] = vsColor[gl_InvocationID];
        tcsTangent[gl_InvocationID] = vsTangent[gl_InvocationID];
        tcsShadowCoef[gl_InvocationID] = vsShadowCoef[gl_InvocationID];

        gl_TessLevelOuter[0] = float(1);
        gl_TessLevelOuter[1] = float(16);
    }
);

const char* hairTessControlShader2 = STRINGIFY(
    #version 410 compatibility \n
    layout(vertices = 4) out;

in vec4 vsColor[];
in vec3 vsTangent[];
in float vsShadowCoef[];

out vec4 tcsColor[];
out vec3 tcsTangent[];
out float tcsShadowCoef[];

void main()
{
    // Pass along the vertex position unmodified
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    tcsColor[gl_InvocationID] = vsColor[gl_InvocationID];
    tcsTangent[gl_InvocationID] = vsTangent[gl_InvocationID];
    tcsShadowCoef[gl_InvocationID] = vsShadowCoef[gl_InvocationID];

    gl_TessLevelOuter[0] = float(1);
    gl_TessLevelOuter[1] = float(1);
}
);

const char* hairTessEvaluationShader = STRINGIFY(
    #version 410 compatibility \n
    layout(isolines) in;

    in vec4 tcsColor[];
    in vec3 tcsTangent[];
    in float tcsShadowCoef[];

    out TES_OUT{
        vec4 color;
        vec3 pos;
        vec3 tangent;
        float shadowCoef;
    }tes_out;

    vec3 cubicInterpolation(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float mu) {
        float mu2 = mu * mu;
        vec3 a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;

        vec3 a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
        vec3 a2 = -0.5 * p0 + 0.5 * p2;
        vec3 a3 = p1;

        return (a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3);
    }

    float cubicInterpolation2(float p0, float p1, float p2, float p3, float mu) {
        float mu2 = mu * mu;
        float a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;

        float a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
        float a2 = -0.5 * p0 + 0.5 * p2;
        float a3 = p1;

        return (a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3);
    }

    void main()
    {
        float u = gl_TessCoord.x;
        vec3 p0 = gl_in[0].gl_Position.xyz;
        vec3 p1 = gl_in[1].gl_Position.xyz;
        vec3 p2 = gl_in[2].gl_Position.xyz;
        vec3 p3 = gl_in[3].gl_Position.xyz;
        float leng = length(p1 - p0);
        // Linear interpolation

        vec3 p;
        p = cubicInterpolation(p0, p1, p2, p3, u);

        // Transform to clip coordinates
        gl_Position = gl_ModelViewProjectionMatrix * vec4(p, 1);

        tes_out.pos = p;
        tes_out.color = tcsColor[0];
        tes_out.tangent = cubicInterpolation(tcsTangent[0], tcsTangent[1], tcsTangent[2], tcsTangent[3], u);
        tes_out.shadowCoef = cubicInterpolation2(tcsShadowCoef[0], tcsShadowCoef[1], tcsShadowCoef[2], tcsShadowCoef[3],u);
    }
);

const char* hairGeometryShader = STRINGIFY(
    #version 410 compatibility \n
    layout(lines) in;
    layout(triangle_strip, max_vertices = 4) out;

    in TES_OUT{
        vec4 color;
        vec3 pos;
        vec3 tangent;
        float shadowCoef;
    }gs_in[];

    out GS_OUT{
        vec4 color;
        vec3 pos;
        float shadowCoef;
        vec3 tangent;
        float widthCoef;
    }gs_out;

    out vec4 fColor;
    out vec3 fPos;

    void main() {
        float halfWidth = 0.01f;
        gs_out.color = gs_in[1].color;

        for (int i = 0; i < 2; i++) {
            gs_out.tangent = gs_in[i].tangent;
            gs_out.shadowCoef = gs_in[i].shadowCoef;
            vec3 widthVec = vec3(1, 0, 0);
            for (int j = 0; j < 2; j++) {

                float deltaWidth = (1 - 2 * j) * halfWidth;
                gs_out.pos = gs_in[i].pos;
                gs_out.widthCoef = 1.0 - 2.0 * j;
                gl_Position = (gl_in[i].gl_Position + vec4(widthVec * deltaWidth, 0.0));
                EmitVertex();
            }
        }

        EndPrimitive();
    }
);

const char* hairGeometryShader2 = STRINGIFY(
    #version 410 compatibility \n
    layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

in TES_OUT{
    vec4 color;
    vec3 pos;
    vec3 tangent;
    float shadowCoef;
}gs_in[];

out GS_OUT{
    vec4 color;
    vec3 pos;
    float shadowCoef;
    vec3 tangent;
    float widthCoef;
}gs_out;

out vec4 fColor;
out vec3 fPos;

void main() {
    float halfWidth = 0.02f;
    gs_out.color = gs_in[1].color;

    for (int i = 0; i < 2; i++) {
        gs_out.tangent = gs_in[i].tangent;
        gs_out.shadowCoef = gs_in[i].shadowCoef;
        vec3 widthVec = vec3(1, 0, 0);
        for (int j = 0; j < 2; j++) {

            float deltaWidth = (1 - 2 * j) * halfWidth;
            gs_out.pos = gs_in[i].pos;
            gs_out.widthCoef = 1.0 - 2.0 * j;
            gl_Position = (gl_in[i].gl_Position + vec4(widthVec * deltaWidth, 0.0));
            EmitVertex();
        }
    }

    EndPrimitive();
}
);

const char* hairFragmentShaderMarschner = STRINGIFY(
    #version 410 compatibility \n

    in GS_OUT{
        vec4 color;
        vec3 pos;
        float shadowCoef;
        vec3 tangent;
        float widthCoef;
    }fs_in;

    out vec4 color;

    uniform vec3 lightPos;
    uniform vec3 viewPos;

    float sinOfVec(vec3 a1, vec3 a2) {
        return sqrt(1 - pow(dot(a1, a2), 2));
    }

    float g(float b, float x) {
        return exp(-pow(x, 2) / 2 / pow(b, 2)) / b / sqrt(2 * 3.1415926);
    }

    float g_2016(float b, float thetaI, float thetaR, float a) {
        return exp(-pow(sin(thetaI) + sin(thetaR) - a, 2) / 2 / pow(b, 2)) / b / sqrt(2 * 3.1415926);
    }

    float fresnel(float eta, float x) {
        float f0 = pow((eta - 1) / (eta + 1), 2);
        return f0 + (1 - f0) * pow(1 - x, 5);
    }

    vec3 proj2normPlane(vec3 v, vec3 t) {
        vec3 proj2tangent = dot(v, t) * t;
        return v - proj2tangent;
    }

    float interp(float a, float b, float r) {
        return a * (1 - r) + b * r;
    }
    void main()
    {
        float PI = 3.1415926;
        float eta = 1.55;
        vec3 objectColor = fs_in.color.xyz;
        vec3 lightColor = vec3(1.0, 1.0, 1.0);
        vec3 result = vec3(0, 0, 0);

        vec3 tangent = normalize(fs_in.tangent);
        vec3 viewDir = normalize(viewPos - fs_in.pos);
        vec3 lightDir = normalize(lightPos - fs_in.pos);
        lightDir = vec3(0, 0, 1);

        float phi = acos(dot(normalize(proj2normPlane(viewDir, tangent)), normalize(proj2normPlane(lightDir, tangent))));

        float _thetaV = acos(dot(viewDir, tangent));
        float _thetaL = acos(dot(lightDir, tangent));

        float thetaV = PI / 2 - _thetaV;
        float thetaL = PI / 2 - _thetaL;

        float thetaH = (thetaV + thetaL) / 2;
        float thetaD = (thetaV - thetaL) / 2;
        float Alpha[3];
        Alpha[0] = -PI / 180 * 7;
        Alpha[1] = -Alpha[0] / 2;
        Alpha[2] = -Alpha[0] / 2 * 3;

        float Beta[3];
        Beta[0]= 0.4*0.4;//roughness*roughness
        Beta[1] = Beta[0] / 2;
        Beta[2] = Beta[0] * 2;

        float M[3];

        for (int i = 0; i < 3; i++) {
            M[i] = g_2016(Beta[i], thetaL, thetaV, Alpha[i]);
        }

        float cosHalfAngle = sqrt(0.5 + 0.5 * dot(viewDir, lightDir));
        float Nr = cos(phi / 2) / 4 * fresnel(eta, cosHalfAngle);
        Nr = cos(phi / 2) / 4;
        vec3 componentR = lightColor * M[0] * Nr;

        float eta2 = 1.19 / cos(thetaD) + 0.36 * cos(thetaD);
        float htt = (1 + (0.6 - 0.8 * cos(phi)) / eta2) * cos(phi / 2);
        float C = 0.6;
        float Att = pow(1 - fresnel(eta2, cosHalfAngle), 2) * exp(log(C) * sqrt(1 - pow(htt / eta2, 2)) / 2 / cos(thetaD));
        float Ntt = exp(-3.65 * cos(phi) - 3.98) * Att;

        vec3 componentTT = objectColor * lightColor * M[1] * Ntt;

        //float Atrt = pow(1 - fresnel(eta, cosHalfAngle), 2) * fresnel(1.0f / eta, cos(asin(sqrt(3)/2/eta))) * pow(exp(log(C) * 0.8 / cos(thetaD)), 2);
        float Atrt = pow(1 - fresnel(eta, cosHalfAngle), 2)  * pow(exp(log(C) * 0.8 / cos(thetaD)), 2);
        float Ntrt = exp(17 * cos(phi) - 16.78) * Atrt;
        vec3 componentTRT = objectColor * lightColor * M[2] * Ntrt;

        // ambient
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * objectColor;

        // diffuse 
        float diffuseStrength = 0.3;
        float diff = max(sinOfVec(tangent, lightDir), 0.0);
        vec3 diffuse = diffuseStrength * diff * objectColor;

        float lightRatio = 1.0 - abs(fs_in.widthCoef);
        result = ((componentR + componentTT + componentTRT) * 0.6 + diffuse)*fs_in.shadowCoef + ambient;
        //result = (componentR + componentTT + componentTRT) * fs_in.shadowCoef;
        color = vec4(result, 0.9);
    }
);

const char* hairFragmentShader2 = STRINGIFY(
    #version 410 compatibility \n

    in GS_OUT{
        vec4 color;
        vec3 pos;
        float shadowCoef;
        vec3 tangent;
        float widthCoef;
    }fs_in;

out vec4 color;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
    color = vec4(1.0, 0.82, 0.0, 1.0);
}
);
