// [Opt 4] Motion-Gated Temporal Hysteresis UI Detector + Edge Prior + Explicit HUD Hints
//
// Key anti-jello insight:
//   Only update UI confidence when surrounding background motion > motion_gate_threshold!
//   - If camera/scene is moving AND this pixel stays static -> ramp UP ui_confidence quickly (attack_rate)
//   - If camera/scene is moving AND this pixel changes -> ramp DOWN slowly (decay_rate, hysteresis)
//   - If camera/scene is still -> FREEZE ui_confidence (never misclassify static background as UI)
//   - High Sobel edge gradient boosts UI lock (sharp text / thin crosshairs)
//   - User-defined HUD hint rects force strong static UI lock at zero heuristic cost
//
// Packed RGBA8 output: R = ui_confidence, G = 0 (filled by ui_dilate), B = edge_mag, A = hint_flag

struct UiDetectParams {
    width: u32,
    height: u32,
    motion_gate_threshold: f32,
    static_diff_threshold: f32,
    attack_rate: f32,
    decay_rate: f32,
    edge_prior_weight: f32,
    hint_count: u32,
    hud_hints: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var prev_tex: texture_2d<f32>;
@group(0) @binding(1) var curr_tex: texture_2d<f32>;
@group(0) @binding(2) var flow_tex: texture_2d<f32>;
@group(0) @binding(3) var prev_ui_tex: texture_2d<f32>;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var ui_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<uniform> params: UiDetectParams;
@group(0) @binding(7) var<storage, read> tile_flags: array<u32>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn in_hint_rect(uv: vec2<f32>, r: vec4<f32>) -> bool {
    return uv.x >= r.x && uv.x <= (r.x + r.z) && uv.y >= r.y && uv.y <= (r.y + r.w);
}

// Tile-gated flow read: ignores stale vectors left in flow_tex on static skipped tiles!
fn sample_active_flow_mag(uv: vec2<f32>, tiles_x: u32, tiles_y: u32) -> f32 {
    let cuv = clamp(uv, vec2<f32>(0.0), vec2<f32>(0.9999));
    let tx = min(u32(cuv.x * f32(params.width)) / 16u, tiles_x - 1u);
    let ty = min(u32(cuv.y * f32(params.height)) / 16u, tiles_y - 1u);
    if (tile_flags[ty * tiles_x + tx] == 0u) {
        return 0.0;
    }
    return textureSampleLevel(flow_tex, linear_sampler, cuv, 0.0).a;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
        / vec2<f32>(f32(params.width), f32(params.height));

    let c0 = textureLoad(prev_tex, p, 0).rgb;
    let c1 = textureLoad(curr_tex, p, 0).rgb;
    let pix_diff = max(abs(luma(c1) - luma(c0)), length(c1 - c0) * 0.5);

    // 3x3 Sobel-like gradient for sharp UI text / crosshair prior
    let max_p = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    let l_l = luma(textureLoad(curr_tex, clamp(p + vec2<i32>(-1, 0), vec2<i32>(0), max_p), 0).rgb);
    let l_r = luma(textureLoad(curr_tex, clamp(p + vec2<i32>( 1, 0), vec2<i32>(0), max_p), 0).rgb);
    let l_u = luma(textureLoad(curr_tex, clamp(p + vec2<i32>( 0,-1), vec2<i32>(0), max_p), 0).rgb);
    let l_d = luma(textureLoad(curr_tex, clamp(p + vec2<i32>( 0, 1), vec2<i32>(0), max_p), 0).rgb);
    let edge_mag = clamp(sqrt((l_r - l_l) * (l_r - l_l) + (l_d - l_u) * (l_d - l_u)) * 2.5, 0.0, 1.0);

    // Sample neighborhood background flow magnitude gated by live tile_flags (zero stale flow!)
    let tiles_x = max(1u, (params.width + 15u) / 16u);
    let tiles_y = max(1u, (params.height + 15u) / 16u);
    let off = vec2<f32>(16.0 / f32(params.width), 16.0 / f32(params.height));
    let m0 = sample_active_flow_mag(uv, tiles_x, tiles_y);
    let m1 = sample_active_flow_mag(uv + vec2<f32>( off.x, 0.0), tiles_x, tiles_y);
    let m2 = sample_active_flow_mag(uv + vec2<f32>(-off.x, 0.0), tiles_x, tiles_y);
    let m3 = sample_active_flow_mag(uv + vec2<f32>(0.0,  off.y), tiles_x, tiles_y);
    let m4 = sample_active_flow_mag(uv + vec2<f32>(0.0, -off.y), tiles_x, tiles_y);
    let bg_motion = max(m0, max(max(m1, m2), max(m3, m4)));

    var prev_conf = textureLoad(prev_ui_tex, p, 0).r;

    // Check explicit Profile HUD static region hints
    var is_hint = false;
    let count = min(params.hint_count, 4u);
    for (var i = 0u; i < count; i = i + 1u) {
        if (in_hint_rect(uv, params.hud_hints[i])) {
            is_hint = true;
        }
    }

    var next_conf = prev_conf;
    if (is_hint) {
        // Explicit hint: lock static HUD pixels while still allowing dynamic moving background behind transparent hints
        if (pix_diff < params.static_diff_threshold * 2.5) {
            next_conf = 1.0;
        } else {
            next_conf = max(0.0, prev_conf - params.decay_rate * 2.0);
        }
    } else if (bg_motion > params.motion_gate_threshold) {
        // Motion-gated update: camera/background is moving here
        let edge_boost = 1.0 + params.edge_prior_weight * edge_mag;
        if (pix_diff < params.static_diff_threshold) {
            // Static pixel while background moves -> strong UI evidence!
            next_conf = clamp(prev_conf + params.attack_rate * edge_boost, 0.0, 1.0);
        } else {
            // Pixel moved/changed -> slow hysteresis decay
            next_conf = clamp(prev_conf - params.decay_rate, 0.0, 1.0);
        }
    }
    // Else: bg_motion <= gate -> keep prev_conf unchanged (freeze)

    textureStore(ui_out, p, vec4<f32>(next_conf, 0.0, edge_mag, select(0.0, 1.0, is_hint)));
}
