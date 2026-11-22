#version 450

layout(location = 0) in vec2 guest_position;

void main()
{
    gl_Position = vec4(guest_position, 0.0, 1.0);
}
