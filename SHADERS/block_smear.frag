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

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    float mask_val = texture(u_mask, v_uv).r;
    vec4 orig = texture(u_texture, v_uv);
    float mask_effect = 1.0 - smoothstep(0.3, 0.7, mask_val);
    if (mask_effect < 0.001) { fragColor = orig; return; }

    float t = u_time + float(u_frame_idx) * 0.12;
    float pulse = 0.7 + 0.3 * sin(t * 0.9);
    float eff = (u_intensity * pulse) + 0.15;

    float block_px = mix(48.0, 10.0, eff);
    vec2 block_size = vec2(block_px) / u_resolution;
    vec2 block_id = floor(v_uv / block_size);
    vec2 block_uv = fract(v_uv / block_size);

    float n1 = snoise(block_id * 1.7 + t * 0.15);
    float n2 = hash(block_id + floor(t * 2.0));
    float is_glitched = step(0.55 - eff * 0.2, abs(n1));

    vec2 sample_uv = v_uv;

    if (is_glitched > 0.5) {
        float mode = hash(block_id * 3.1 + floor(t * 1.5));

        if (mode < 0.35) {
            vec2 swap_offset = vec2(
                floor(snoise(block_id * 5.0 + t * 0.2) * 4.0),
                floor(snoise(block_id * 5.0 + t * 0.2 + 50.0) * 3.0)
            );
            vec2 src_block = block_id + swap_offset;
            sample_uv = (src_block + block_uv) * block_size;
        }
        else if (mode < 0.65) {
            float smear_dir = sign(n2 - 0.5);
            float smear_amt = eff * 0.04;
            sample_uv.x = v_uv.x + smear_dir * smear_amt;
            sample_uv.y = v_uv.y;
        }
        else {
            vec2 neighbor = block_id + vec2(sign(n1), 0.0);
            float blend = smoothstep(0.0, 0.3, block_uv.x) * step(n2, 0.6);
            vec2 neighbor_uv = (neighbor + block_uv) * block_size;
            neighbor_uv = clamp(neighbor_uv, 0.002, 0.998);
            vec4 neighbor_col = texture(u_texture, neighbor_uv);
            sample_uv = v_uv;
            vec4 tex_here = texture(u_texture, sample_uv);
            vec4 tex_blend = mix(tex_here, neighbor_col, blend * eff);

            vec4 feedback = texture(u_feedback, v_uv);
            tex_blend = mix(tex_blend, feedback, eff * 0.15);

            float edge_x = smoothstep(0.0, 0.05, block_uv.x) * smoothstep(1.0, 0.95, block_uv.x);
            float edge_y = smoothstep(0.0, 0.05, block_uv.y) * smoothstep(1.0, 0.95, block_uv.y);
            float block_edge = 1.0 - edge_x * edge_y;
            tex_blend.rgb += vec3(0.15, 0.3, 0.2) * block_edge * eff * 0.3 * is_glitched;

            fragColor = mix(orig, tex_blend, mask_effect * is_glitched);
            return;
        }
    }

    sample_uv = clamp(sample_uv, 0.002, 0.998);
    vec4 tex = texture(u_texture, sample_uv);

    vec4 feedback = texture(u_feedback, v_uv);
    tex = mix(tex, feedback, eff * 0.15);

    float quant_shift = is_glitched * eff * 0.15;
    tex.rgb = mix(tex.rgb, floor(tex.rgb * 8.0) / 8.0, quant_shift);

    float edge_x = smoothstep(0.0, 0.05, block_uv.x) * smoothstep(1.0, 0.95, block_uv.x);
    float edge_y = smoothstep(0.0, 0.05, block_uv.y) * smoothstep(1.0, 0.95, block_uv.y);
    float block_edge = 1.0 - edge_x * edge_y;
    tex.rgb += vec3(0.15, 0.3, 0.2) * block_edge * eff * 0.3 * is_glitched;

    fragColor = mix(orig, tex, mask_effect * mix(1.0, is_glitched, 0.6));
}
