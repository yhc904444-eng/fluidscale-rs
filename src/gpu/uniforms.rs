use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TileParamsUniform {
    pub width: u32,
    pub height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub tile_skip_threshold: f32,
    pub scene_cut_tile_threshold: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PyramidParamsUniform {
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FlowParamsUniform {
    pub flow_width: u32,
    pub flow_height: u32,
    pub src_width: u32,
    pub src_height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub search_radius: i32,
    pub flow_scale_inv: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct UiDetectParamsUniform {
    pub width: u32,
    pub height: u32,
    pub motion_gate_threshold: f32,
    pub static_diff_threshold: f32,
    pub attack_rate: f32,
    pub decay_rate: f32,
    pub edge_prior_weight: f32,
    pub hint_count: u32,
    pub hud_hints: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DilateParamsUniform {
    pub width: u32,
    pub height: u32,
    pub radius: i32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SynthParamsUniform {
    pub width: u32,
    pub height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub alpha: f32,
    pub scene_cut_threshold: f32,
    pub top_layer_composite: u32,
    pub debug_view: u32,
    pub split_x: f32,
    pub compare_mode: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct UpscaleOsdParamsUniform {
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub mode: u32,
    pub sharpness: f32,
    pub show_fps: u32,
    pub osd_scale: u32,
    pub captured_fps: u32,
    pub output_fps: u32,
    pub active_tile_pct: u32,
    pub flow_scale_pct: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PresentParamsUniform {
    pub scale_x: f32,
    pub scale_y: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct FrameStatsReadback {
    pub active_tiles: u32,
    pub scene_cut_tiles: u32,
    pub diff_sum_fixed: u32,
    pub total_tiles: u32,
}

#[allow(dead_code)]
impl FrameStatsReadback {
    pub fn active_ratio(&self) -> f32 {
        if self.total_tiles == 0 {
            1.0
        } else {
            self.active_tiles as f32 / self.total_tiles as f32
        }
    }

    pub fn mean_abs_diff(&self) -> f32 {
        if self.total_tiles == 0 {
            0.0
        } else {
            (self.diff_sum_fixed as f32 / 65535.0) / self.total_tiles as f32
        }
    }
}
