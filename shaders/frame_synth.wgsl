// [Opt 1 & 4] Occlusion-Aware One-Sided Background Warping + Top-Layer Crisp UI Composite
// + Tile-Skip Fast Path + Scene-Cut Fallback + Interactive Split-Curtain A/B Compare

struct SynthParams {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
    alpha: f32,               // interpolation timestamp t in [0, 1]
    scene_cut_threshold: f32, // fraction of drastically changed tiles triggering passthrough
    top_layer_composite: u32,
    debug_view: u32,          // 0 = normal, 1 = full-frame debug mask/tile overlay
    split_x: f32,             // normalized [0, 1] A/B curtain divider position
    compare_mode: u32,        // 0 = off, 1 = debug mask/tiles, 2 = raw prev frame, 3 = naive warp (no UI shield)
    _pad0: u32,
    _pad1: u32,
};

struct FrameStats {
    active_tiles: u32,
    scene_cut_tiles: u32,
    diff_sum_fixed: u32,
    total_tiles: u32,
};

@group(0) @binding(0) var prev_tex: texture_2d<f32>;
@group(0) @binding(1) var curr_tex: texture_2d<f32>;
@group(0) @binding(2) var flow_tex: texture_2d<f32>;
@group(0) @binding(3) var ui_packed_tex: texture_2d<f32>;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var synth_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<uniform> params: SynthParams;
@group(0) @binding(7) var<storage, read> tile_flags: array<u32>;
@group(0) @binding(8) var<storage, read> stats: FrameStats;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let inv_size = vec2<f32>(1.0 / f32(params.width), 1.0 / f32(params.height));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) * inv_size;

    let tile_idx = (gid.y / 16u) * params.tiles_x + (gid.x / 16u);
    let tile_active = tile_flags[tile_idx];
    let curr_rgb = textureLoad(curr_tex, p, 0).rgb;
    let prev_rgb = textureLoad(prev_tex, p, 0).rgb;

    // 1. Scene-cut protection: pass through crisp real frame on sudden cuts / menus
    let cut_ratio = f32(stats.scene_cut_tiles) / max(1.0, f32(stats.total_tiles));
    if (cut_ratio > params.scene_cut_threshold) {
        textureStore(synth_out, p, vec4<f32>(curr_rgb, 1.0));
        return;
    }

    let ui_info = textureLoad(ui_packed_tex, p, 0);
    let ui_conf = ui_info.r;
    let flow_sample = textureSampleLevel(flow_tex, linear_sampler, uv, 0.0);
    let flow_uv = flow_sample.xy * inv_size;
    let flow_conf = flow_sample.z;
    let flow_mag = flow_sample.a;

    let on_left_curtain = params.compare_mode != 0u && uv.x < params.split_x;

    // 2. Endpoint continuity & tile-skip fast path:
    //    alpha <= 0.001 is timestamp t=0 (prev_rgb), alpha >= 0.999 is t=1 (curr_rgb).
    //    Using curr_rgb at alpha=0 would cause a 1-frame forward-then-backward stutter!
    var result_rgb = select(prev_rgb, curr_rgb, params.alpha >= 0.5 || tile_active == 0u);
    if (on_left_curtain && params.compare_mode == 2u) {
        // Compare Mode 2: Left side shows raw stepped native frame without interpolation
        result_rgb = prev_rgb;
    } else if (tile_active != 0u && params.alpha > 0.001 && params.alpha < 0.999) {
        let t = params.alpha;
        let uv0 = clamp(uv - flow_uv * t, vec2<f32>(0.0), vec2<f32>(1.0));
        let uv1 = clamp(uv + flow_uv * (1.0 - t), vec2<f32>(0.0), vec2<f32>(1.0));

        let c0 = textureSampleLevel(prev_tex, linear_sampler, uv0, 0.0).rgb;
        let c1 = textureSampleLevel(curr_tex, linear_sampler, uv1, 0.0).rgb;

        if (on_left_curtain && params.compare_mode == 3u) {
            // Compare Mode 3: Naive unprotected warp (shows jello bleeding around HUD!)
            result_rgb = mix(c0, c1, t);
        } else {
            // 3. Anti-bleed UI exclusion: reject background samples that land inside dilated UI mask!
            let ui_excl_0 = textureSampleLevel(ui_packed_tex, linear_sampler, uv0, 0.0).g;
            let ui_excl_1 = textureSampleLevel(ui_packed_tex, linear_sampler, uv1, 0.0).g;

            var w0 = (1.0 - t) * (1.0 - smoothstep(0.35, 0.70, ui_excl_0));
            var w1 = t * (1.0 - smoothstep(0.35, 0.70, ui_excl_1));

            let color_err = length(c1 - c0);
            if (color_err > 0.45 && flow_conf < 0.5) {
                if (t < 0.5) {
                    w1 = w1 * 0.25;
                } else {
                    w0 = w0 * 0.25;
                }
            }

            let w_sum = w0 + w1;
            if (w_sum > 0.001) {
                result_rgb = (c0 * w0 + c1 * w1) / w_sum;
            } else {
                result_rgb = mix(c0, c1, t);
            }

            // 4. Top-layer crisp UI compositing
            if (params.top_layer_composite != 0u) {
                let ui_alpha = smoothstep(0.45, 0.75, ui_conf);
                result_rgb = mix(result_rgb, curr_rgb, ui_alpha);
            }
        }
    }

    // Clean understated analysis heatmap (full frame OR left side of A/B compare mode 1)
    let show_debug = params.debug_view != 0u || (on_left_curtain && params.compare_mode == 1u);
    if (show_debug) {
        if (ui_conf > 0.45) {
            // Clean glacier-blue highlight for locked static HUD/crosshair
            result_rgb = mix(result_rgb, vec3<f32>(0.31, 0.61, 0.98), 0.58);
        } else if (tile_active != 0u) {
            // Active moving tile: subtle hairline grid + gentle warm motion highlight
            let energy = clamp(flow_mag * 0.04, 0.0, 0.25);
            result_rgb = mix(result_rgb, vec3<f32>(0.85, 0.90, 0.98), energy);
            if ((gid.x % 16u) == 0u || (gid.y % 16u) == 0u) {
                result_rgb = mix(result_rgb, vec3<f32>(0.55, 0.72, 0.95), 0.30);
            }
        } else {
            // Static skipped tile: calm darkened matte tint (visually proves 0 optical-flow cost)
            result_rgb = result_rgb * vec3<f32>(0.38, 0.42, 0.48);
        }
    }

    textureStore(synth_out, p, vec4<f32>(result_rgb, 1.0));
}
