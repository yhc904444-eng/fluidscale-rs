// [Opt 1 & 3] 16x16 Tile Static/Active Detector + Indirect Dispatch Generator
// Also computes scene-cut tile ratio and content-aware dedup luma delta in a single pass.

struct TileParams {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
    tile_skip_threshold: f32,
    scene_cut_tile_threshold: f32,
    _pad0: u32,
    _pad1: u32,
};

struct IndirectArgs {
    x: atomic<u32>,
    y: u32,
    z: u32,
};

struct FrameStats {
    active_tiles: atomic<u32>,
    scene_cut_tiles: atomic<u32>,
    diff_sum_fixed: atomic<u32>,
    total_tiles: u32,
};

@group(0) @binding(0) var prev_tex: texture_2d<f32>;
@group(0) @binding(1) var curr_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: TileParams;
@group(0) @binding(3) var<storage, read_write> tile_flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> active_tile_coords: array<u32>;
@group(0) @binding(5) var<storage, read_write> indirect_args: IndirectArgs;
@group(0) @binding(6) var<storage, read_write> stats: FrameStats;

var<workgroup> shared_diff: array<f32, 256>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let px = wg_id.x * 16u + lid.x;
    let py = wg_id.y * 16u + lid.y;

    var d: f32 = 0.0;
    if (px < params.width && py < params.height) {
        let c0 = textureLoad(prev_tex, vec2<i32>(i32(px), i32(py)), 0).rgb;
        let c1 = textureLoad(curr_tex, vec2<i32>(i32(px), i32(py)), 0).rgb;
        let diff_rgb = abs(c1 - c0);
        d = max(abs(luma(c1) - luma(c0)), max(diff_rgb.r, max(diff_rgb.g, diff_rgb.b)) * 0.5);
    }

    shared_diff[lidx] = d;
    workgroupBarrier();

    // Parallel workgroup reduction (256 -> 1)
    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lidx < stride) {
            shared_diff[lidx] = shared_diff[lidx] + shared_diff[lidx + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (lidx == 0u && wg_id.x < params.tiles_x && wg_id.y < params.tiles_y) {
        let mean_diff = shared_diff[0] / 256.0;
        let tile_idx = wg_id.y * params.tiles_x + wg_id.x;

        // Accumulate fixed-point frame delta for content-aware deduplication
        let diff_fixed = u32(clamp(mean_diff * 65535.0, 0.0, 65535.0));
        atomicAdd(&stats.diff_sum_fixed, diff_fixed);

        if (mean_diff > params.tile_skip_threshold) {
            tile_flags[tile_idx] = 1u;
            let slot = atomicAdd(&indirect_args.x, 1u);
            // Pack (tile_x, tile_y) into one u32
            active_tile_coords[slot] = (wg_id.y << 16u) | (wg_id.x & 0xFFFFu);
            atomicAdd(&stats.active_tiles, 1u);
        } else {
            tile_flags[tile_idx] = 0u;
        }

        if (mean_diff > params.scene_cut_tile_threshold) {
            atomicAdd(&stats.scene_cut_tiles, 1u);
        }
    }
}
