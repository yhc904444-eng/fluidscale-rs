// [Opt 1 & 3] Fused Spatial Upscaler (Bicubic+CAS / Lanczos / SharpBilinear / Nearest)
// + Zero-D2D1 Shader-Inline 5x7 Bitmap Font FPS Counter ("captured / output")
// Eliminates D2D1/DWrite SwapChain interop so DirectFlip / MPO / VRR stay intact!

struct UpscaleOsdParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mode: u32,          // 0=off/bilinear, 1=bicubic_cas, 2=lanczos, 3=sharp_bilinear, 4=nearest
    sharpness: f32,
    show_fps: u32,
    osd_scale: u32,
    captured_fps: u32,
    output_fps: u32,
    active_tile_pct: u32,
    flow_scale_pct: u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: UpscaleOsdParams;

fn sample_src(px: vec2<i32>) -> vec3<f32> {
    let clamped = clamp(px, vec2<i32>(0), vec2<i32>(i32(params.src_width) - 1, i32(params.src_height) - 1));
    return textureLoad(src_tex, clamped, 0).rgb;
}

fn cubic_weight(x: f32) -> f32 {
    let ax = abs(x);
    if (ax <= 1.0) {
        return (1.5 * ax - 2.5) * ax * ax + 1.0;
    } else if (ax < 2.0) {
        return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0;
    }
    return 0.0;
}

fn sample_bicubic(uv: vec2<f32>) -> vec3<f32> {
    let coord = uv * vec2<f32>(f32(params.src_width), f32(params.src_height)) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(coord));
    let f = fract(coord);

    var acc = vec3<f32>(0.0);
    var wsum: f32 = 0.0;
    for (var j = -1; j <= 2; j = j + 1) {
        let wy = cubic_weight(f32(j) - f.y);
        for (var i = -1; i <= 2; i = i + 1) {
            let wx = cubic_weight(f32(i) - f.x);
            let w = wx * wy;
            acc = acc + sample_src(base + vec2<i32>(i, j)) * w;
            wsum = wsum + w;
        }
    }
    return clamp(acc / max(wsum, 1e-5), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sinc(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1e-4) {
        return 1.0;
    }
    let pix = 3.14159265 * x;
    return sin(pix) / pix;
}

fn lanczos2_weight(x: f32) -> f32 {
    if (abs(x) >= 2.0) {
        return 0.0;
    }
    return sinc(x) * sinc(x * 0.5);
}

fn sample_lanczos2(uv: vec2<f32>) -> vec3<f32> {
    let coord = uv * vec2<f32>(f32(params.src_width), f32(params.src_height)) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(coord));
    let f = fract(coord);

    var acc = vec3<f32>(0.0);
    var wsum: f32 = 0.0;
    for (var j = -1; j <= 2; j = j + 1) {
        let wy = lanczos2_weight(f32(j) - f.y);
        for (var i = -1; i <= 2; i = i + 1) {
            let wx = lanczos2_weight(f32(i) - f.x);
            let w = wx * wy;
            acc = acc + sample_src(base + vec2<i32>(i, j)) * w;
            wsum = wsum + w;
        }
    }
    return clamp(acc / max(wsum, 1e-5), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn apply_cas(center: vec3<f32>, uv: vec2<f32>, sharpness: f32) -> vec3<f32> {
    if (sharpness <= 0.001) {
        return center;
    }
    let texel = vec2<f32>(1.0 / f32(params.src_width), 1.0 / f32(params.src_height));
    let n = textureSampleLevel(src_tex, linear_sampler, uv + vec2<f32>(0.0, -texel.y), 0.0).rgb;
    let s = textureSampleLevel(src_tex, linear_sampler, uv + vec2<f32>(0.0,  texel.y), 0.0).rgb;
    let w = textureSampleLevel(src_tex, linear_sampler, uv + vec2<f32>(-texel.x, 0.0), 0.0).rgb;
    let e = textureSampleLevel(src_tex, linear_sampler, uv + vec2<f32>( texel.x, 0.0), 0.0).rgb;

    let mn = min(center, min(min(n, s), min(w, e)));
    let mx = max(center, max(max(n, s), max(w, e)));
    let amp = clamp(min(mn, vec3<f32>(1.0) - mx) / max(mx, vec3<f32>(1e-4)), vec3<f32>(0.0), vec3<f32>(1.0));
    let peak = -1.0 / mix(8.0, 5.0, clamp(sharpness, 0.0, 1.0));
    let weight = sqrt(amp) * peak;

    let num = center + (n + s + w + e) * weight;
    let den = vec3<f32>(1.0) + 4.0 * weight;
    return clamp(num / max(den, vec3<f32>(1e-4)), vec3<f32>(0.0), vec3<f32>(1.0));
}

// 3x5 compact bitmap font for digits '0'-'9' and '/' (15 bits per glyph)
fn glyph_bits(ch: u32) -> u32 {
    switch ch {
        case 0u: { return 31599u; } // 0
        case 1u: { return 9362u; }  // 1
        case 2u: { return 29671u; } // 2
        case 3u: { return 29391u; } // 3
        case 4u: { return 23497u; } // 4
        case 5u: { return 31183u; } // 5
        case 6u: { return 31215u; } // 6
        case 7u: { return 29257u; } // 7
        case 8u: { return 31727u; } // 8
        case 9u: { return 31695u; } // 9
        case 10u: { return 4681u; } // '/'
        default: { return 0u; }     // space
    }
}

fn glyph_pixel(ch: u32, gx: u32, gy: u32) -> bool {
    if (gx >= 3u || gy >= 5u) {
        return false;
    }
    let bit_idx = (4u - gy) * 3u + (2u - gx);
    return ((glyph_bits(ch) >> bit_idx) & 1u) != 0u;
}

fn render_osd(px: u32, py: u32, bg: vec3<f32>) -> vec3<f32> {
    let sc = max(1u, params.osd_scale);
    let ox = 16u;
    let oy = 14u;
    let cell_w = 4u * sc;
    let cell_h = 6u * sc;
    // Format: "CCC / OOO" (9 characters)
    let box_w = 9u * cell_w + 12u;
    let box_h = cell_h + 10u;

    if (px < ox || py < oy || px >= ox + box_w || py >= oy + box_h) {
        return bg;
    }

    var color = bg * 0.35; // translucent dark pill backdrop
    let lx = px - ox;
    let ly = py - oy;
    if (lx < 6u || ly < 5u) {
        return color;
    }

    let char_idx = (lx - 6u) / cell_w;
    let gx = ((lx - 6u) % cell_w) / sc;
    let gy = (ly - 5u) / sc;

    let cap = min(params.captured_fps, 999u);
    let out_f = min(params.output_fps, 999u);

    var ch: u32 = 11u;
    switch char_idx {
        case 0u: { ch = select(11u, (cap / 100u) % 10u, cap >= 100u); }
        case 1u: { ch = select(11u, (cap / 10u) % 10u, cap >= 10u); }
        case 2u: { ch = cap % 10u; }
        case 3u: { ch = 11u; }
        case 4u: { ch = 10u; } // '/'
        case 5u: { ch = 11u; }
        case 6u: { ch = select(11u, (out_f / 100u) % 10u, out_f >= 100u); }
        case 7u: { ch = select(11u, (out_f / 10u) % 10u, out_f >= 10u); }
        case 8u: { ch = out_f % 10u; }
        default: { ch = 11u; }
    }

    if (glyph_pixel(ch, gx, gy)) {
        color = select(vec3<f32>(0.90, 0.94, 0.98), vec3<f32>(0.40, 0.95, 0.65), char_idx >= 6u);
    }
    return color;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
        / vec2<f32>(f32(params.dst_width), f32(params.dst_height));

    var rgb: vec3<f32>;
    if (params.mode == 4u) {
        // Nearest / Integer
        let sp = vec2<i32>(i32(uv.x * f32(params.src_width)), i32(uv.y * f32(params.src_height)));
        rgb = sample_src(sp);
    } else if (params.mode == 1u) {
        // Bicubic + Contrast Adaptive Sharpening
        let base = sample_bicubic(uv);
        rgb = apply_cas(base, uv, params.sharpness);
    } else if (params.mode == 2u) {
        // Lanczos-2 + Contrast Adaptive Sharpening ("电影锐利")
        let base = sample_lanczos2(uv);
        rgb = apply_cas(base, uv, params.sharpness);
    } else if (params.mode == 3u) {
        // Sharp bilinear (anti-aliased pixel-edge curve)
        let coord = uv * vec2<f32>(f32(params.src_width), f32(params.src_height));
        let f = fract(coord);
        let sharpened = clamp((f - vec2<f32>(0.5)) * 2.0 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
        let suv = (floor(coord) + sharpened) / vec2<f32>(f32(params.src_width), f32(params.src_height));
        rgb = textureSampleLevel(src_tex, linear_sampler, suv, 0.0).rgb;
    } else {
        rgb = textureSampleLevel(src_tex, linear_sampler, uv, 0.0).rgb;
    }

    if (params.show_fps != 0u) {
        rgb = render_osd(gid.x, gid.y, rgb);
    }

    textureStore(dst_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb, 1.0));
}
