#version 330 core

uniform sampler2D u_texture;
uniform sampler2D u_mask;
uniform sampler2D u_feedback;
uniform vec2 u_resolution;
uniform float u_time;
uniform int u_frame_idx;
uniform int u_seed;
uniform float u_intensity;

in vec2 v_uv;

out vec4 fragColor;

void main() {
    // Mandatory mask protection: discard or passthrough where subject is protected
    float mask_val = texture(u_mask, v_uv).r;
    if (mask_val > 0.5) {
        fragColor = texture(u_texture, v_uv);
        return;
    }

    // Shader logic here — only affects glitch zone (mask_val <= 0.5)
    vec4 tex = texture(u_texture, v_uv);
    fragColor = tex;
}
