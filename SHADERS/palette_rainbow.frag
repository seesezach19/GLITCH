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

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

void main() {
    float mask_val = texture(u_mask, v_uv).r;
    vec4 orig = texture(u_texture, v_uv);
    float mask_effect = 1.0 - smoothstep(0.3, 0.7, mask_val);
    if (mask_effect < 0.001) { fragColor = orig; return; }

    float t = u_time + float(u_frame_idx) * 0.1;
    float pulse = 0.7 + 0.3 * sin(t * 0.7);
    float eff = (u_intensity * pulse) + 0.15;

    vec3 col = orig.rgb;
    float luma = dot(col, vec3(0.299, 0.587, 0.114));

    float levels = max(3.0, 16.0 - eff * 10.0);
    vec3 crushed = floor(col * levels + 0.5) / levels;

    float noise_offset = snoise(v_uv * 8.0 + t * 0.1) * 0.05 * eff;
    float hue = fract(luma * (1.5 + eff * 2.0) + t * 0.05 + noise_offset);
    float sat = 0.6 + eff * 0.3;
    float val = luma * 0.8 + 0.2;
    vec3 rainbow = hsv2rgb(vec3(hue, sat, val));

    float rainbow_amt = eff * 0.55;
    vec3 tinted = mix(crushed, rainbow, rainbow_amt);

    float band_noise = snoise(vec2(luma * 20.0, t * 0.3));
    float banding = smoothstep(0.3, 0.5, abs(band_noise)) * eff * 0.15;
    tinted += vec3(banding * 0.5, -banding * 0.3, banding * 0.4);

    float block_size = 4.0 + (1.0 - eff) * 12.0;
    vec2 block_uv = floor(v_uv * u_resolution / block_size) * block_size / u_resolution;
    float block_luma = dot(texture(u_texture, block_uv).rgb, vec3(0.299, 0.587, 0.114));
    float artifact = step(0.8, abs(snoise(block_uv * 30.0 + t * 0.5)));
    vec3 artifact_color = hsv2rgb(vec3(fract(block_luma * 3.0 + t * 0.1), 0.9, 0.8));
    tinted = mix(tinted, artifact_color, artifact * eff * 0.2);

    vec3 hsv = rgb2hsv(tinted);
    hsv.x = fract(hsv.x + sin(t * 0.15) * eff * 0.05);
    tinted = hsv2rgb(hsv);

    vec4 feedback = texture(u_feedback, v_uv);
    vec4 result = vec4(tinted, 1.0);
    result = mix(result, feedback, eff * 0.12);

    fragColor = mix(orig, result, mask_effect);
}
