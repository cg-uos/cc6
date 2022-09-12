#version 330 core

layout(location = 0) in vec4 aPosition;

out vec3 vTexCoord;
uniform mat4 MVP;
uniform vec3    scale;

void main() {
    vTexCoord = aPosition.xyz;
    gl_Position = MVP*((2.0*aPosition-vec4(1,1,1,0))*vec4(scale,1));
//    gl_Position = MVP*aPosition;
}
