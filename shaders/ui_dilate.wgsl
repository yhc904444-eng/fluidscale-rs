// [Opt 4] Morphological Dilation Pass for Anti-Bleed UI Exclusion Zone
// Dilates R (ui_confidence) by dilation_radius (1-2px) into G (ui_exclusion_zone),
// so background warping never samples across UI boundaries (eliminates HUD halo/jello).

struct DilateParams {
    width: u32,
    height: u32,
    radius: i32,
    _pad: u32,
};

@group(0) @binding(0) var ui_in: texture_2d<f32>;
@group(0) @binding(1) var ui_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: DilateParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let max_p = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    let center = textureLoad(ui_in, p, 0);

    var dilated: f32 = center.r;
    let rad = clamp(params.radius, 0, 3);
    for (var dy = -rad; dy <= rad; dy = dy + 1) {
        for (var dx = -rad; dx <= rad; dx = dx + 1) {
            let sp = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), max_p);
            let s = textureLoad(ui_in, sp, 0).r;
            dilated = max(dilated, s);
        }
    }

    textureStore(ui_out, p, vec4<f32>(center.r, dilated, center.b, center.a));
}
