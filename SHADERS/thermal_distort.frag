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

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm3(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 3; i++) { v += a * noise(p); p *= 2.2; a *= 0.5; }
    return v;
}

void main() {
    float mask_val = texture(u_mask, v_uv).r;
    vec4 orig = texture(u_texture, v_uv);

    float mask_effect = 1.0 - smoothstep(0.3, 0.7, mask_val);
    if (mask_effect < 0.001) { fragColor = orig; return; }

    float t = u_time + float(u_frame_idx) * 0.15;
    float pulse = 0.7 + 0.3 * sin(t * 2.0);
    float inten = u_intensity * pulse;

    float wave1 = sin(v_uv.y * 14.0 + t * 3.5) * sin(v_uv.y * 9.0 + t * 2.5);
    float wave2 = sin(v_uv.y * 22.0 + t * 4.0) * 0.4;
    float turb = fbm3(v_uv * 8.0 + t * 0.6);
    float disp = (wave1 + wave2 + turb * 0.8) * inten * 0.08 * mask_effect;

    float vert_disp = noise(v_uv * 5.0 + t * 0.8) * inten * 0.03 * mask_effect;

    vec2 uv_r = clamp(v_uv + vec2(disp * 1.4, vert_disp), 0.002, 0.998);
    vec2 uv_g = clamp(v_uv + vec2(disp, vert_disp * 0.5), 0.002, 0.998);
    vec2 uv_b = clamp(v_uv + vec2(disp * 0.6, -vert_disp), 0.002, 0.998);

    float r = texture(u_texture, uv_r).r;
    float g = texture(u_texture, uv_g).g;
    float b = texture(u_texture, uv_b).b;
    vec4 tex = vec4(r, g, b, 1.0);

    vec4 feedback = texture(u_feedback, v_uv);
    tex = mix(tex, feedback, inten * 0.3);

    float lum = dot(tex.rgb, vec3(0.299, 0.587, 0.114));

    vec3 cool_heat = vec3(1.0, 0.7, 0.2);
    vec3 hot_heat = vec3(1.0, 0.25, 0.0);
    vec3 white_heat = vec3(1.0, 0.95, 0.8);
    vec3 heat_col = mix(cool_heat, hot_heat, lum);
    heat_col = mix(heat_col, white_heat, pow(lum, 3.0) * 0.4);
    tex.rgb = mix(tex.rgb, heat_col, inten * 0.55 * mask_effect);

    float shimmer = pow(noise(v_uv * 5.0 + t * 0.9), 1.5);
    float flicker = 0.8 + 0.2 * sin(t * 8.0 + v_uv.x * 20.0);
    tex.rgb += vec3(1.0, 0.65, 0.25) * shimmer * flicker * inten * 0.4 * mask_effect;

    float mirage = noise(vec2(v_uv.x * 3.0, t * 1.5)) * noise(vec2(v_uv.x * 7.0, t * 2.0));
    tex.rgb += vec3(1.0, 0.9, 0.7) * mirage * inten * 0.2 * mask_effect;

    fragColor = mix(orig, tex, mask_effect);
}
