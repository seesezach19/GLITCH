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

    float t = u_time + float(u_frame_idx) * 0.15;
    float pulse = 0.7 + 0.3 * sin(t * 1.1);
    float eff = (u_intensity * pulse) + 0.15;

    float n_bands = 8.0 + eff * 16.0;
    float band_height = 1.0 / n_bands;
    float band_idx = floor(v_uv.y / band_height);
    float band_frac = fract(v_uv.y / band_height);

    float band_seed = band_idx * 13.37 + float(u_seed) * 0.01;
    float band_shift = snoise(vec2(band_seed, t * 0.3)) * eff * 0.12;

    float wave_drift = sin(v_uv.y * 40.0 + t * 2.0) * eff * 0.008;
    float jitter = snoise(vec2(band_idx * 7.1, t * 0.5 + 100.0));
    float extra_shift = step(0.7, abs(jitter)) * jitter * eff * 0.06;

    float total_shift = band_shift + wave_drift + extra_shift;

    float edge_fade = smoothstep(0.0, 0.08, band_frac) * smoothstep(1.0, 0.92, band_frac);
    total_shift *= edge_fade;

    vec2 shifted_uv = vec2(v_uv.x + total_shift * mask_effect, v_uv.y);
    shifted_uv.x = clamp(shifted_uv.x, 0.002, 0.998);

    vec4 tex = texture(u_texture, shifted_uv);

    float split = eff * 0.4;
    float r_shift = total_shift * 1.2;
    float b_shift = total_shift * 0.8;
    tex.r = texture(u_texture, vec2(clamp(v_uv.x + r_shift * mask_effect, 0.002, 0.998), v_uv.y)).r;
    tex.b = texture(u_texture, vec2(clamp(v_uv.x + b_shift * mask_effect, 0.002, 0.998), v_uv.y)).b;

    float scan_line = smoothstep(0.48, 0.5, band_frac) * smoothstep(0.52, 0.5, band_frac);
    tex.rgb = mix(tex.rgb, tex.rgb * 0.7, scan_line * eff * 0.6);

    float glitch_flash = step(0.92, snoise(vec2(band_idx * 3.1, t * 1.5)));
    tex.rgb = mix(tex.rgb, vec3(1.0) - tex.rgb, glitch_flash * eff * 0.3 * mask_effect);

    vec4 feedback = texture(u_feedback, v_uv);
    tex = mix(tex, feedback, eff * 0.15);

    fragColor = mix(orig, tex, mask_effect);
}
