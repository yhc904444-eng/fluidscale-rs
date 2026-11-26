// Fullscreen triangle presenter for DirectFlip / SwapChain output
// Supports exact aspect-ratio preservation (clean black letterbox/pillarbox) and stretch-to-fill.

struct PresentParams {
    scale_x: f32,
    scale_y: f32,
    offset_x: f32,
    offset_y: f32,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 0.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@group(0) @binding(0) var final_tex: texture_2d<f32>;
@group(0) @binding(1) var final_sampler: sampler;
@group(0) @binding(2) var<uniform> p_params: PresentParams;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = (in.uv - vec2<f32>(p_params.offset_x, p_params.offset_y))
        / vec2<f32>(max(p_params.scale_x, 1e-4), max(p_params.scale_y, 1e-4));
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return textureSample(final_tex, final_sampler, uv);
}
