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

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * noise(p); p *= 2.0; a *= 0.5; }
    return v;
}

float caustic_pattern(vec2 uv, float t) {
    vec2 i = floor(uv);
    vec2 f = fract(uv);
    float d1 = 10.0, d2 = 10.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 n = vec2(float(x), float(y));
            vec2 p = vec2(hash(i + n), hash(i + n + 100.0));
            p = 0.5 + 0.5 * sin(t * 0.9 + 6.28 * p);
            float d = length(f - n - p);
            if (d < d1) { d2 = d1; d1 = d; }
            else if (d < d2) { d2 = d; }
        }
    }
    return d2 - d1;
}

void main() {
    float mask_val = texture(u_mask, v_uv).r;
    vec4 orig = texture(u_texture, v_uv);

    float mask_effect = 1.0 - smoothstep(0.3, 0.7, mask_val);
    if (mask_effect < 0.001) { fragColor = orig; return; }

    float t = u_time + float(u_frame_idx) * 0.15;
    float pulse = 0.7 + 0.3 * sin(t * 1.5);
    float inten = u_intensity * pulse;

    vec2 uv = v_uv;
    vec2 warp;
    warp.x = sin(uv.y * 8.0 + t * 2.0) * cos(uv.x * 6.0 + t * 1.4);
    warp.y = cos(uv.x * 8.0 + t * 1.7) * sin(uv.y * 6.0 + t * 1.2);
    warp += vec2(fbm(uv * 5.0 + t * 0.6), fbm(uv * 5.0 + t * 0.6 + 30.0)) - 0.5;
    uv += warp * inten * 0.12 * mask_effect;
    uv = clamp(uv, 0.002, 0.998);

    vec4 tex = texture(u_texture, uv);

    vec4 feedback = texture(u_feedback, v_uv);
    tex = mix(tex, feedback, inten * 0.35);

    float c1 = caustic_pattern(v_uv * 6.0, t);
    float c2 = caustic_pattern(v_uv * 4.0 + 10.0, t * 0.7);
    float c3 = caustic_pattern(v_uv * 8.0 + 25.0, t * 1.3);
    float caustic = pow(c1, 1.2) * 0.5 + pow(c2, 1.2) * 0.3 + pow(c3, 1.5) * 0.2;
    caustic = pow(caustic, 0.7);

    vec3 deep = vec3(0.05, 0.2, 0.35);
    vec3 bright = vec3(0.4, 0.95, 1.0);
    vec3 highlight = vec3(1.0, 1.0, 0.95);
    vec3 cc = mix(deep, bright, caustic);
    cc = mix(cc, highlight, pow(caustic, 3.0));

    tex.rgb += cc * caustic * inten * 1.2 * mask_effect;

    tex.rgb = mix(tex.rgb, tex.rgb * vec3(0.8, 1.0, 1.15), inten * 0.3 * mask_effect);

    float sparkle = pow(max(caustic - 0.6, 0.0) / 0.4, 4.0);
    tex.rgb += vec3(0.8, 1.0, 1.0) * sparkle * inten * 0.5 * mask_effect;

    fragColor = mix(orig, tex, mask_effect);
}
