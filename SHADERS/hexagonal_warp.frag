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

vec4 hex_grid(vec2 uv) {
    vec2 s = vec2(1.0, 1.732);
    vec2 h = s * 0.5;
    vec2 a = mod(uv, s) - h;
    vec2 b = mod(uv - h, s) - h;
    vec2 gv = dot(a, a) < dot(b, b) ? a : b;
    vec2 id = uv - gv;
    return vec4(gv, id);
}

void main() {
    float mask_val = texture(u_mask, v_uv).r;
    vec4 orig = texture(u_texture, v_uv);

    float mask_effect = 1.0 - smoothstep(0.3, 0.7, mask_val);
    if (mask_effect < 0.001) { fragColor = orig; return; }

    float t = u_time + float(u_frame_idx) * 0.15;
    float pulse = 0.7 + 0.3 * sin(t * 1.8);
    float inten = u_intensity * pulse;

    float scale = 5.0 + inten * 3.0;
    vec4 hx = hex_grid(v_uv * scale);
    vec2 cell = hx.xy;
    vec2 cell_id = hx.zw;

    float cell_phase = hash(cell_id * 1.3) * 6.28;
    float cell_speed = 0.8 + hash(cell_id * 2.7) * 0.8;

    vec2 warp = vec2(
        sin(t * cell_speed + cell_phase) * noise(cell_id * 0.7 + t * 0.25),
        cos(t * cell_speed * 0.9 + cell_phase) * noise(cell_id * 0.7 + t * 0.3 + 30.0)
    ) * inten * 0.18 * mask_effect;

    vec2 suv = clamp(v_uv + warp, 0.002, 0.998);
    vec4 tex = texture(u_texture, suv);

    vec4 feedback = texture(u_feedback, v_uv);
    tex = mix(tex, feedback, inten * 0.3);

    float ch = hash(cell_id + 7.0);
    float ct = fract(t * 0.08 + ch);
    vec3 cc;
    if (ct < 0.33)      cc = mix(vec3(0.0, 0.4, 0.6), vec3(0.1, 0.7, 0.5), ct * 3.0);
    else if (ct < 0.66) cc = mix(vec3(0.1, 0.7, 0.5), vec3(0.5, 0.2, 0.8), (ct - 0.33) * 3.0);
    else                cc = mix(vec3(0.5, 0.2, 0.8), vec3(0.0, 0.4, 0.6), (ct - 0.66) * 3.0);
    tex.rgb = mix(tex.rgb, cc, inten * 0.5 * mask_effect);

    float dist = length(cell);
    float hex_radius = 0.5;
    float edge = smoothstep(hex_radius - 0.08, hex_radius, dist);

    float edge_pulse = 0.7 + 0.3 * sin(t * 2.5 + hash(cell_id) * 6.28);
    vec3 ec = mix(vec3(0.0, 0.9, 1.0), vec3(0.9, 0.2, 1.0), hash(cell_id + 3.0));
    float glow = edge * inten * edge_pulse * mask_effect;
    tex.rgb = mix(tex.rgb, ec, glow * 0.9);
    tex.rgb += ec * pow(max(edge, 0.0), 3.0) * inten * 0.4 * mask_effect;

    float inner_dark = max(0.0, 1.0 - dist * 2.2);
    tex.rgb *= 1.0 - inner_dark * inten * 0.12 * mask_effect;

    fragColor = mix(orig, tex, mask_effect);
}
