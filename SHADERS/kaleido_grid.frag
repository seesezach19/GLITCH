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

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x * 34.0) + 10.0) * x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m; m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

vec2 triangle_wave(vec2 uv) {
    return abs(mod(uv, 2.0) - 1.0);
}

void main() {
    float mask_val = texture(u_mask, v_uv).r;
    vec4 orig = texture(u_texture, v_uv);
    float mask_effect = 1.0 - smoothstep(0.3, 0.7, mask_val);
    if (mask_effect < 0.001) { fragColor = orig; return; }

    float t = u_time + float(u_frame_idx) * 0.12;
    float pulse = 0.7 + 0.3 * sin(t * 0.8);
    float eff = (u_intensity * pulse) + 0.15;

    float tiles = 2.0 + eff * 3.0;
    vec2 uv = v_uv;
    vec2 centered = uv - 0.5;

    float angle = snoise(vec2(t * 0.1, 0.0)) * eff * 0.15;
    float ca = cos(angle), sa = sin(angle);
    centered = vec2(ca * centered.x - sa * centered.y,
                    sa * centered.x + ca * centered.y);
    uv = centered + 0.5;

    vec2 tiled_uv = uv * tiles;
    vec2 cell = floor(tiled_uv);
    float cell_noise = snoise(cell * 3.7 + t * 0.05) * 0.5 + 0.5;
    vec2 mirrored = triangle_wave(tiled_uv + cell_noise * eff * 0.3);

    float cell_angle = snoise(cell * 2.3 + t * 0.08) * eff * 0.4;
    float cca = cos(cell_angle), csa = sin(cell_angle);
    vec2 cell_centered = mirrored - 0.5;
    mirrored = vec2(cca * cell_centered.x - csa * cell_centered.y,
                    csa * cell_centered.x + cca * cell_centered.y) + 0.5;

    mirrored = clamp(mirrored, 0.002, 0.998);

    vec4 tex = texture(u_texture, mirrored);
    vec4 feedback = texture(u_feedback, v_uv);
    tex = mix(tex, feedback, eff * 0.2);

    vec2 edge = abs(fract(tiled_uv) - 0.5) * 2.0;
    float grid_line = smoothstep(0.92, 1.0, max(edge.x, edge.y));
    vec3 grid_color = vec3(0.0, 0.8, 0.9) * grid_line * eff * 0.5;

    float portal_glow = smoothstep(0.6, 0.3, length(fract(tiled_uv) - 0.5));
    vec3 glow_tint = mix(vec3(0.1, 0.0, 0.2), vec3(0.0, 0.3, 0.4), cell_noise);
    tex.rgb += glow_tint * portal_glow * eff * 0.15;
    tex.rgb += grid_color;

    fragColor = mix(orig, tex, mask_effect);
}
