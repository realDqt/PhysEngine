#define STRINGIFY(A) #A

// TODO:UNIMPLEMENTED

const char *meshVertexShader = STRINGIFY(
    // #version 150\n
    in vec3 position;
    in vec3 normal;
    varying vec3 position_eye;
    varying vec3 normal_eye;
    in vec4 Ka;
    in vec4 Kd;
    in vec4 Ks;
    in vec2 texcoord;
    varying vec2 texcoordi;
    varying vec4 Kai;
    varying vec4 Kdi;
    varying vec4 Ksi;
    uniform vec4 fixed_color=vec4(1,0,0,0);
    uniform vec4 position_offset=vec4(0,0,0,0);
    uniform vec3 scale=vec3(1,1,1);

    void main()\n
    {
        position_eye = vec3(gl_ModelViewMatrix * (position_offset+vec4(scale*position, 1.0))); \n
        normal_eye = gl_NormalMatrix * normal; \n
        normal_eye = normalize(normal_eye); \n
        gl_Position = gl_ModelViewProjectionMatrix * (position_offset+vec4(scale*position, 1.0)); \n 
        Kai = Ka; \n
        Kdi = Kd; \n
        Ksi = Ks; \n
        texcoordi = texcoord; \n
        gl_FrontColor=fixed_color;
    }
);

// fragment shader
const char *meshFragmentShader = STRINGIFY(
    // #version 150
    
    in vec3 position_eye;
    in vec3 normal_eye;
    uniform vec3 light_position_eye;
    vec3 Ls = vec3 (1, 1, 1);
    vec3 Ld = vec3 (1, 1, 1);
    vec3 La = vec3 (1, 1, 1);
    in vec4 Ksi;
    in vec4 Kdi;
    in vec4 Kai;
    in vec2 texcoordi;
    uniform sampler2D tex;
    uniform float specular_exponent;
    uniform float lighting_factor;
    uniform float texture_factor;
    uniform float matcap_factor;
    uniform float double_sided;
    varying vec3 outColor;
    void main()
    {
        vec4 s = vec4(normalize(light_position_eye - position_eye), 0.0);\n
        vec4 r = vec4(reflect(-s.xyz,normal_eye),0.0); \n
        vec4 v = vec4(normalize(-position_eye), 0.0); \n
        float spec = max(dot(v,r), 0.0); \n
        float diff = max(dot(normal_eye, s.xyz),0.0); \n

        vec3 diffColor = diff * gl_Color; \n
        vec3 specColor = pow(spec,3) * vec3(1.0,1.0,1.0); \n
        vec3 ambientColor = vec3(0.1,0.1,0.1); \n
        
        gl_FragColor = vec4(diffColor + 0.5 * specColor + ambientColor, 1.0);
    }
);

// vertex shader
// const char *meshVertexShader = STRINGIFY(
//     #version 150
//     uniform mat4 view;
//     uniform mat4 proj;
//     uniform mat4 normal_matrix;
//     in vec3 position;
//     in vec3 normal;
//     out vec3 position_eye;
//     out vec3 normal_eye;
//     in vec4 Ka;
//     in vec4 Kd;
//     in vec4 Ks;
//     in vec2 texcoord;
//     out vec2 texcoordi;
//     out vec4 Kai;
//     out vec4 Kdi;
//     out vec4 Ksi;

//     void main()
//     {
//         position_eye = vec3 (view * vec4 (position, 1.0));
//         normal_eye = vec3 (normal_matrix * vec4 (normal, 0.0));
//         normal_eye = normalize(normal_eye);
//         gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * vec4(position, 1.0);"
//         Kai = Ka;
//         Kdi = Kd;
//         Ksi = Ks;
//         texcoordi = texcoord;
//     }
// );

// // fragment shader
// const char *meshFragmentShader = STRINGIFY(
//     #version 150
//     uniform mat4 view;
//     uniform mat4 proj;
//     uniform vec4 fixed_color;
//     in vec3 position_eye;
//     in vec3 normal_eye;
//     uniform vec3 light_position_eye;
//     vec3 Ls = vec3 (1, 1, 1);
//     vec3 Ld = vec3 (1, 1, 1);
//     vec3 La = vec3 (1, 1, 1);
//     in vec4 Ksi;
//     in vec4 Kdi;
//     in vec4 Kai;
//     in vec2 texcoordi;
//     uniform sampler2D tex;
//     uniform float specular_exponent;
//     uniform float lighting_factor;
//     uniform float texture_factor;
//     uniform float matcap_factor;
//     uniform float double_sided;
//     out vec4 outColor;
//     void main()
//     {
//         if(matcap_factor == 1.0f)
//         {
//             vec2 uv = normalize(normal_eye).xy * 0.5 + 0.5;
//             outColor = texture(tex, uv);
//         }else
//         {
//             vec3 Ia = La * vec3(Kai);    // ambient intensity

//             vec3 vector_to_light_eye = light_position_eye - position_eye;
//             vec3 direction_to_light_eye = normalize (vector_to_light_eye);
//             float dot_prod = dot (direction_to_light_eye, normalize(normal_eye));
//             float clamped_dot_prod = abs(max (dot_prod, -double_sided));
//             vec3 Id = Ld * vec3(Kdi) * clamped_dot_prod;    // Diffuse intensity

//             vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normal_eye));
//             vec3 surface_to_viewer_eye = normalize (-position_eye);
//             float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
//             dot_prod_specular = float(abs(dot_prod)==dot_prod) * abs(max (dot_prod_specular, -double_sided));
//             float specular_factor = pow (dot_prod_specular, specular_exponent);
//             vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity
//             vec4 color = vec4(lighting_factor * (Is + Id) + Ia + (1.0-lighting_factor) * vec3(Kdi),(Kai.a+Ksi.a+Kdi.a)/3);
//             outColor = mix(vec4(1,1,1,1), texture(tex, texcoordi), texture_factor) * color;
//             if (fixed_color != vec4(0.0)) outColor = fixed_color;
//         }
//     }
// );