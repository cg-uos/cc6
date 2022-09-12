#version 330 core

in vec3 vTexCoord;

out vec4 fColor;
uniform float zTexCoord;

void main() {
    fColor = vec4(vTexCoord, 1);
}

