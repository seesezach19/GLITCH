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

void main() {
    float mask_val = texture(u_mask, v_uv).r;
    vec4 orig = texture(u_texture, v_uv);
    float mask_effect = 1.0 - smoothstep(0.3, 0.7, mask_val);
    if (mask_effect < 0.001) { fragColor = orig; return; }

    float t = u_time + float(u_frame_idx) * 0.18;
    float eff = u_intensity + 0.15;

    float dot_density = mix(3.0, 6.0, eff);
    float cell_px = u_resolution.y / (dot_density * 40.0);
    vec2 cell_size = vec2(cell_px) / u_resolution;

    vec2 cell_id = floor(v_uv / cell_size);
    vec2 cell_center = (cell_id + 0.5) * cell_size;
    vec2 local = (v_uv - cell_center) / cell_size;

    vec3 sample_col = texture(u_texture, cell_center).rgb;

    float sat_boost = 1.0 + eff * 0.6;
    float luma = dot(sample_col, vec3(0.299, 0.587, 0.114));
    sample_col = mix(vec3(luma), sample_col, sat_boost);
    sample_col = clamp(sample_col, 0.0, 1.0);

    float contrast = 1.0 + eff * 0.4;
    sample_col = clamp((sample_col - 0.5) * contrast + 0.5, 0.0, 1.0);

    float cell_noise = snoise(cell_id * 3.7 + t * 0.3) * 0.5 + 0.5;
    float pulse = 0.85 + 0.15 * sin(t * 2.5 + cell_noise * 6.28);

    float base_radius = 0.35 + luma * 0.15;
    float radius = base_radius * pulse;

    float dist = length(local);

    float dot_edge = smoothstep(radius, radius - 0.08, dist);

    float glow_radius = radius + 0.12 * eff;
    float glow = smoothstep(glow_radius, radius, dist) * 0.3 * eff;

    vec3 bg = vec3(0.02, 0.02, 0.03);
    vec3 dot_col = sample_col * dot_edge;
    vec3 glow_col = sample_col * glow;

    vec3 result = bg + dot_col + glow_col;

    vec4 feedback = texture(u_feedback, v_uv);
    result = mix(result, feedback.rgb, eff * 0.08);

    fragColor = mix(orig, vec4(result, 1.0), mask_effect);
}
