// [Opt 1 & 2] LDS-Assisted Feature Pyramid Builder (Luma + Spatial Gradients)
// Packs (luma, grad_x, grad_y, edge_mag) into a single compact RGBA16F texture.

struct PyramidParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: PyramidParams;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    let inv_dst = vec2<f32>(1.0 / f32(params.dst_width), 1.0 / f32(params.dst_height));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) * inv_dst;

    // 4-tap tent filter for anti-aliased downsample
    let half_texel = 0.5 * inv_dst;
    let c00 = textureSampleLevel(src_tex, linear_sampler, uv + vec2<f32>(-half_texel.x, -half_texel.y), 0.0).rgb;
    let c10 = textureSampleLevel(src_tex, linear_sampler, uv + vec2<f32>( half_texel.x, -half_texel.y), 0.0).rgb;
    let c01 = textureSampleLevel(src_tex, linear_sampler, uv + vec2<f32>(-half_texel.x,  half_texel.y), 0.0).rgb;
    let c11 = textureSampleLevel(src_tex, linear_sampler, uv + vec2<f32>( half_texel.x,  half_texel.y), 0.0).rgb;

    let l00 = luma(c00);
    let l10 = luma(c10);
    let l01 = luma(c01);
    let l11 = luma(c11);

    let center_luma = 0.25 * (l00 + l10 + l01 + l11);
    let gx = 0.5 * ((l10 - l00) + (l11 - l01));
    let gy = 0.5 * ((l01 - l00) + (l11 - l10));
    let edge_mag = sqrt(gx * gx + gy * gy);

    textureStore(dst_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(center_luma, gx, gy, edge_mag));
}
