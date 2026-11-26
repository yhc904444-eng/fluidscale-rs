use crate::config::{AppConfig, GameProfile};
use crate::gpu::render_graph::{RenderGraphBuilder, TextureSpec, TransientResourcePool};
use crate::gpu::uniforms::*;
use anyhow::Result;
use std::time::Instant;
use tracing::info;
use wgpu::util::DeviceExt;

const TILE_ACTIVITY_WGSL: &str = include_str!("../../shaders/tile_activity.wgsl");
const PYRAMID_DOWN_WGSL: &str = include_str!("../../shaders/pyramid_down.wgsl");
const OPTICAL_FLOW_WGSL: &str = include_str!("../../shaders/optical_flow.wgsl");
const UI_DETECT_WGSL: &str = include_str!("../../shaders/ui_detect.wgsl");
const UI_DILATE_WGSL: &str = include_str!("../../shaders/ui_dilate.wgsl");
const FRAME_SYNTH_WGSL: &str = include_str!("../../shaders/frame_synth.wgsl");
const UPSCALE_CAS_WGSL: &str = include_str!("../../shaders/upscale_cas.wgsl");
const PRESENT_WGSL: &str = include_str!("../../shaders/present.wgsl");

#[allow(dead_code)]
pub struct FrameTelemetry {
    pub active_tiles: u32,
    pub total_tiles: u32,
    pub scene_cut_tiles: u32,
    pub mean_abs_diff: f32,
    pub gpu_submit_ms: f32,
}

/// Complete GPU compute & presentation pipeline:
///   1. `tile_activity.wgsl`  -> 16x16 static/active tile detector + IndirectDispatch args + dedup/scene-cut stats
///   2. `pyramid_down.wgsl`   -> Flow-scale luma + gradient pyramids (prev & curr)
///   3. `optical_flow.wgsl`   -> **Indirect Dispatch** ONLY over active tiles (static tiles = 0 GPU work)
///   4. `ui_detect.wgsl`      -> Motion-gated temporal hysteresis UI mask + Sobel edge prior + HUD hints
///   5. `ui_dilate.wgsl`      -> Morphological dilation for anti-bleed UI exclusion zone
///   6. `frame_synth.wgsl`    -> Occlusion-aware one-sided background warp + top-layer crisp UI composite
///   7. `upscale_cas.wgsl`    -> Fused Spatial Upscaler + CAS + Zero-D2D1 Shader-Inline FPS Counter
///   8. `present.wgsl`        -> Fullscreen triangle blit to SwapChain surface
pub struct FluidPipeline {
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub flow_width: u32,
    pub flow_height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,

    pub pool: TransientResourcePool,
    ui_ping_pong: bool,
    frame_ping_pong: bool,

    linear_sampler: wgpu::Sampler,

    // Compute pipelines
    tile_pipeline: wgpu::ComputePipeline,
    pyramid_pipeline: wgpu::ComputePipeline,
    flow_pipeline: wgpu::ComputePipeline,
    ui_detect_pipeline: wgpu::ComputePipeline,
    ui_dilate_pipeline: wgpu::ComputePipeline,
    synth_pipeline: wgpu::ComputePipeline,
    upscale_pipeline: wgpu::ComputePipeline,
    present_pipeline: Option<wgpu::RenderPipeline>,

    // Storage & indirect buffers
    tile_flags_buf: wgpu::Buffer,
    active_tile_coords_buf: wgpu::Buffer,
    indirect_args_buf: wgpu::Buffer,
    stats_buf: wgpu::Buffer,
    stats_readback_buf: wgpu::Buffer,

    // Uniform buffers
    tile_ubo: wgpu::Buffer,
    pyr_ubo: wgpu::Buffer,
    flow_ubo: wgpu::Buffer,
    ui_detect_ubo: wgpu::Buffer,
    dilate_ubo: wgpu::Buffer,
    synth_ubo: wgpu::Buffer,
    upscale_ubo: wgpu::Buffer,
    present_ubo: wgpu::Buffer,
}

impl FluidPipeline {
    pub fn new(
        device: &wgpu::Device,
        cfg: &AppConfig,
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        flow_scale: f32,
        surface_format: Option<wgpu::TextureFormat>,
    ) -> Result<Self> {
        let src_width = src_width.max(16);
        let src_height = src_height.max(16);
        let dst_width = dst_width.max(16);
        let dst_height = dst_height.max(16);
        let flow_scale = flow_scale.clamp(0.20, 1.0);
        let flow_width = ((src_width as f32 * flow_scale).round() as u32).max(16);
        let flow_height = ((src_height as f32 * flow_scale).round() as u32).max(16);

        let tiles_x = src_width.div_ceil(16);
        let tiles_y = src_height.div_ceil(16);
        let total_tiles = tiles_x * tiles_y;

        // Build RenderGraph & compile transient aliasing plan
        let mut rg = RenderGraphBuilder::new();
        let src_rw_usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        let transient_usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST;

        let src_rgba8 = TextureSpec {
            width: src_width,
            height: src_height,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: src_rw_usage,
        };
        let src_transient_rgba8 = TextureSpec {
            width: src_width,
            height: src_height,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: transient_usage,
        };
        let dst_rgba8 = TextureSpec {
            width: dst_width,
            height: dst_height,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: transient_usage | wgpu::TextureUsages::COPY_SRC,
        };
        let flow_rgba16f = TextureSpec {
            width: flow_width,
            height: flow_height,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: transient_usage,
        };

        // Persistent history textures (ping-pong frames + temporal UI hysteresis history)
        rg.add_persistent("frame_a", src_rgba8);
        rg.add_persistent("frame_b", src_rgba8);
        rg.add_persistent("ui_hist_a", src_transient_rgba8);
        rg.add_persistent("ui_hist_b", src_transient_rgba8);

        // Transient pass textures with exact [first_pass, last_pass] lifetimes:
        // Pass 1-2: pyramid_down -> prev_pyr & curr_pyr, consumed at Pass 3 (optical_flow)
        rg.add_transient("prev_pyr", flow_rgba16f, 1, 3);
        rg.add_transient("curr_pyr", flow_rgba16f, 2, 3);
        // Pass 3: optical_flow -> flow_tex, consumed at Pass 4 (ui_detect) and Pass 6 (frame_synth)
        rg.add_transient("flow_tex", flow_rgba16f, 3, 6);
        // Pass 4: ui_detect -> ui_raw, consumed at Pass 5 (ui_dilate) -> DEAD after Pass 5!
        rg.add_transient("ui_raw", src_transient_rgba8, 4, 5);
        // Pass 6: frame_synth -> synth_tex (ALIASES dead ui_raw slot!), consumed at Pass 7
        rg.add_transient("synth_tex", src_transient_rgba8, 6, 7);
        // Pass 7: upscale_cas -> output_tex, consumed at Pass 8 (present)
        rg.add_transient("output_tex", dst_rgba8, 7, 8);

        let plan = rg.compile(cfg.gpu.enable_vram_aliasing);
        info!(
            src = format!("{src_width}x{src_height}"),
            flow = format!("{flow_width}x{flow_height}"),
            dst = format!("{dst_width}x{dst_height}"),
            physical_slots = plan.physical_specs.len(),
            naive_mb = format!("{:.2}", plan.naive_vram_bytes as f64 / (1024.0 * 1024.0)),
            aliased_mb = format!("{:.2}", plan.aliased_vram_bytes as f64 / (1024.0 * 1024.0)),
            saved_pct = format!("{:.1}%", plan.saved_percent()),
            "RenderGraph compiled transient VRAM pool"
        );

        let pool = TransientResourcePool::realize(device, plan);

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fluidscale-linear-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let tile_pipeline = create_compute(device, "tile-activity", TILE_ACTIVITY_WGSL);
        let pyramid_pipeline = create_compute(device, "pyramid-down", PYRAMID_DOWN_WGSL);
        let flow_pipeline = create_compute(device, "optical-flow", OPTICAL_FLOW_WGSL);
        let ui_detect_pipeline = create_compute(device, "ui-detect", UI_DETECT_WGSL);
        let ui_dilate_pipeline = create_compute(device, "ui-dilate", UI_DILATE_WGSL);
        let synth_pipeline = create_compute(device, "frame-synth", FRAME_SYNTH_WGSL);
        let upscale_pipeline = create_compute(device, "upscale-cas", UPSCALE_CAS_WGSL);

        let present_pipeline = surface_format.map(|fmt| create_present_pipeline(device, fmt));

        let tile_buf_bytes = (total_tiles.max(1) as u64) * 4;
        let tile_flags_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile-flags"),
            size: tile_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let active_tile_coords_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("active-tile-coords"),
            size: tile_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indirect_args_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indirect-dispatch-args"),
            contents: bytemuck::cast_slice(&[0u32, 1u32, 1u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
        });
        let stats_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frame-stats"),
            contents: bytemuck::bytes_of(&FrameStatsReadback {
                total_tiles,
                ..Default::default()
            }),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let stats_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-stats-readback"),
            size: std::mem::size_of::<FrameStatsReadback>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_ubo = make_ubo(
            device,
            "tile-ubo",
            &TileParamsUniform {
                width: src_width,
                height: src_height,
                tiles_x,
                tiles_y,
                tile_skip_threshold: cfg.gpu.tile_skip_threshold,
                scene_cut_tile_threshold: 0.25,
                _pad0: 0,
                _pad1: 0,
            },
        );
        let pyr_ubo = make_ubo(
            device,
            "pyr-ubo",
            &PyramidParamsUniform {
                src_width,
                src_height,
                dst_width: flow_width,
                dst_height: flow_height,
            },
        );
        let flow_ubo = make_ubo(
            device,
            "flow-ubo",
            &FlowParamsUniform {
                flow_width,
                flow_height,
                src_width,
                src_height,
                tiles_x,
                tiles_y,
                search_radius: 16,
                flow_scale_inv: 1.0 / flow_scale,
            },
        );
        let ui_detect_ubo = make_ubo(
            device,
            "ui-detect-ubo",
            &UiDetectParamsUniform {
                width: src_width,
                height: src_height,
                motion_gate_threshold: cfg.ui_protection.motion_gate_threshold,
                static_diff_threshold: cfg.ui_protection.static_diff_threshold,
                attack_rate: cfg.ui_protection.attack_rate,
                decay_rate: cfg.ui_protection.decay_rate,
                edge_prior_weight: cfg.ui_protection.edge_prior_weight,
                hint_count: 0,
                hud_hints: [[0.0; 4]; 4],
            },
        );
        let dilate_ubo = make_ubo(
            device,
            "dilate-ubo",
            &DilateParamsUniform {
                width: src_width,
                height: src_height,
                radius: cfg.ui_protection.dilation_radius as i32,
                _pad: 0,
            },
        );
        let synth_ubo = make_ubo(
            device,
            "synth-ubo",
            &SynthParamsUniform {
                width: src_width,
                height: src_height,
                tiles_x,
                tiles_y,
                alpha: 0.5,
                scene_cut_threshold: cfg.frame_gen.scene_cut_threshold,
                top_layer_composite: u32::from(cfg.ui_protection.top_layer_composite),
                debug_view: 0,
                split_x: 0.5,
                compare_mode: 0,
                _pad0: 0,
                _pad1: 0,
            },
        );
        let upscale_ubo = make_ubo(
            device,
            "upscale-ubo",
            &UpscaleOsdParamsUniform {
                src_width,
                src_height,
                dst_width,
                dst_height,
                mode: cfg.upscaling_mode_id(),
                sharpness: cfg.upscaling.sharpness,
                show_fps: u32::from(cfg.osd.show_fps),
                osd_scale: cfg.osd.scale,
                captured_fps: 0,
                output_fps: 0,
                active_tile_pct: 100,
                flow_scale_pct: (flow_scale * 100.0) as u32,
            },
        );

        let present_ubo = make_ubo(
            device,
            "present-ubo",
            &PresentParamsUniform {
                scale_x: 1.0,
                scale_y: 1.0,
                offset_x: 0.0,
                offset_y: 0.0,
            },
        );

        Ok(Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            flow_width,
            flow_height,
            tiles_x,
            tiles_y,
            pool,
            ui_ping_pong: false,
            frame_ping_pong: false,
            linear_sampler,
            tile_pipeline,
            pyramid_pipeline,
            flow_pipeline,
            ui_detect_pipeline,
            ui_dilate_pipeline,
            synth_pipeline,
            upscale_pipeline,
            present_pipeline,
            tile_flags_buf,
            active_tile_coords_buf,
            indirect_args_buf,
            stats_buf,
            stats_readback_buf,
            tile_ubo,
            pyr_ubo,
            flow_ubo,
            ui_detect_ubo,
            dilate_ubo,
            synth_ubo,
            upscale_ubo,
            present_ubo,
        })
    }

    /// Upload a new RGBA8 source frame into the ping-pong history ring.
    pub fn upload_source_frame(&mut self, queue: &wgpu::Queue, rgba_bytes: &[u8]) {
        self.frame_ping_pong = !self.frame_ping_pong;
        let curr_name = if self.frame_ping_pong {
            "frame_b"
        } else {
            "frame_a"
        };
        let tex = self.pool.texture(curr_name);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.src_width * 4),
                rows_per_image: Some(self.src_height),
            },
            wgpu::Extent3d {
                width: self.src_width,
                height: self.src_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Update presentation aspect-ratio mapping onto the target swapchain surface (`surf_w × surf_h`).
    pub fn set_present_aspect(
        &self,
        queue: &wgpu::Queue,
        surf_w: u32,
        surf_h: u32,
        preserve_aspect: bool,
    ) {
        let params = compute_letterbox_params(
            self.dst_width,
            self.dst_height,
            surf_w,
            surf_h,
            preserve_aspect,
        );
        queue.write_buffer(&self.present_ubo, 0, bytemuck::bytes_of(&params));
    }

    /// Clear temporal UI hysteresis masks, optical flow field, and tile activity flags.
    /// Call on scene switches or capture target changes so stale HUD masks/flow never leak across scenes.
    pub fn reset_history(&mut self, queue: &wgpu::Queue) {
        let src_zeros = vec![0u8; (self.src_width * self.src_height * 4) as usize];
        for name in ["ui_hist_a", "ui_hist_b"] {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: self.pool.texture(name),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &src_zeros,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.src_width * 4),
                    rows_per_image: Some(self.src_height),
                },
                wgpu::Extent3d {
                    width: self.src_width,
                    height: self.src_height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let flow_zeros = vec![0u8; (self.flow_width * self.flow_height * 8) as usize];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.pool.texture("flow_tex"),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &flow_zeros,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.flow_width * 8),
                rows_per_image: Some(self.flow_height),
            },
            wgpu::Extent3d {
                width: self.flow_width,
                height: self.flow_height,
                depth_or_array_layers: 1,
            },
        );
        let tile_zeros = vec![0u8; (self.tiles_x * self.tiles_y * 4) as usize];
        queue.write_buffer(&self.tile_flags_buf, 0, &tile_zeros);
        self.ui_ping_pong = false;
    }

    /// Advance the input frame ring and return the storage TextureView for zero-copy GPU rendering
    /// directly into the latest source frame (used by the multi-scene GPU Demo Lab).
    pub fn advance_and_get_write_frame_view(&mut self) -> &wgpu::TextureView {
        self.frame_ping_pong = !self.frame_ping_pong;
        let curr_name = if self.frame_ping_pong {
            "frame_b"
        } else {
            "frame_a"
        };
        self.pool.view(curr_name)
    }

    fn prev_and_curr_frame_names(&self) -> (&'static str, &'static str) {
        if self.frame_ping_pong {
            ("frame_a", "frame_b")
        } else {
            ("frame_b", "frame_a")
        }
    }

    fn prev_and_curr_ui_names(&self) -> (&'static str, &'static str) {
        if self.ui_ping_pong {
            ("ui_hist_b", "ui_hist_a")
        } else {
            ("ui_hist_a", "ui_hist_b")
        }
    }

    /// Execute the motion-analysis + UI-protection passes whenever a new real frame arrives.
    /// Returns GPU telemetry (active tiles skipped, scene-cut tiles, mean frame diff).
    pub fn analyze_motion_and_ui(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cfg: &AppConfig,
        profile: Option<&GameProfile>,
        readback_stats: bool,
    ) -> FrameTelemetry {
        let t0 = Instant::now();
        let total_tiles = self.tiles_x * self.tiles_y;

        // Reset indirect dispatch args (x=0, y=1, z=1) and frame stats
        queue.write_buffer(
            &self.indirect_args_buf,
            0,
            bytemuck::cast_slice(&[0u32, 1u32, 1u32]),
        );
        queue.write_buffer(
            &self.stats_buf,
            0,
            bytemuck::bytes_of(&FrameStatsReadback {
                active_tiles: 0,
                scene_cut_tiles: 0,
                diff_sum_fixed: 0,
                total_tiles,
            }),
        );

        // Keep live-tunable tile threshold and UI dilation radius synchronized every pair
        queue.write_buffer(
            &self.tile_ubo,
            0,
            bytemuck::bytes_of(&TileParamsUniform {
                width: self.src_width,
                height: self.src_height,
                tiles_x: self.tiles_x,
                tiles_y: self.tiles_y,
                tile_skip_threshold: cfg.gpu.tile_skip_threshold,
                scene_cut_tile_threshold: 0.25,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        queue.write_buffer(
            &self.dilate_ubo,
            0,
            bytemuck::bytes_of(&DilateParamsUniform {
                width: self.src_width,
                height: self.src_height,
                radius: cfg.ui_protection.dilation_radius as i32,
                _pad: 0,
            }),
        );

        // Update HUD static region hints from active profile
        let mut hints = [[0.0f32; 4]; 4];
        let mut hint_count = 0u32;
        if let Some(p) = profile {
            for (i, r) in p.hud_hints.iter().take(4).enumerate() {
                hints[i] = *r;
                hint_count += 1;
            }
        }
        queue.write_buffer(
            &self.ui_detect_ubo,
            0,
            bytemuck::bytes_of(&UiDetectParamsUniform {
                width: self.src_width,
                height: self.src_height,
                motion_gate_threshold: cfg.ui_protection.motion_gate_threshold,
                static_diff_threshold: cfg.ui_protection.static_diff_threshold,
                attack_rate: cfg.ui_protection.attack_rate,
                decay_rate: cfg.ui_protection.decay_rate,
                edge_prior_weight: cfg.ui_protection.edge_prior_weight,
                hint_count,
                hud_hints: hints,
            }),
        );

        let (prev_frame, curr_frame) = self.prev_and_curr_frame_names();
        let (prev_ui, next_ui) = self.prev_and_curr_ui_names();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fluidscale-analyze-encoder"),
        });

        // Pass 0: Tile activity + IndirectDispatch args + dedup/scene-cut stats
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile-bg"),
                layout: &self.tile_pipeline.get_bind_group_layout(0),
                entries: &[
                    tex_entry(0, self.pool.view(prev_frame)),
                    tex_entry(1, self.pool.view(curr_frame)),
                    buf_entry(2, &self.tile_ubo),
                    buf_entry(3, &self.tile_flags_buf),
                    buf_entry(4, &self.active_tile_coords_buf),
                    buf_entry(5, &self.indirect_args_buf),
                    buf_entry(6, &self.stats_buf),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-tile-activity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tile_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.tiles_x, self.tiles_y, 1);
        }

        // Pass 1 & 2: Pyramid downsample (prev & curr)
        for (src_n, dst_n) in [(prev_frame, "prev_pyr"), (curr_frame, "curr_pyr")] {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pyr-bg"),
                layout: &self.pyramid_pipeline.get_bind_group_layout(0),
                entries: &[
                    tex_entry(0, self.pool.view(src_n)),
                    samp_entry(1, &self.linear_sampler),
                    tex_entry(2, self.pool.view(dst_n)),
                    buf_entry(3, &self.pyr_ubo),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-pyramid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pyramid_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.flow_width.div_ceil(16), self.flow_height.div_ceil(16), 1);
        }

        // Pass 3: Sparse Optical Flow via GPU Indirect Dispatch (ONLY active tiles!)
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flow-bg"),
                layout: &self.flow_pipeline.get_bind_group_layout(0),
                entries: &[
                    tex_entry(0, self.pool.view("prev_pyr")),
                    tex_entry(1, self.pool.view("curr_pyr")),
                    tex_entry(2, self.pool.view("flow_tex")),
                    buf_entry(3, &self.flow_ubo),
                    buf_entry(4, &self.active_tile_coords_buf),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-optical-flow-indirect"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.flow_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups_indirect(&self.indirect_args_buf, 0);
        }

        // Pass 4: Motion-Gated Temporal Hysteresis UI Detector -> ui_raw
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ui-detect-bg"),
                layout: &self.ui_detect_pipeline.get_bind_group_layout(0),
                entries: &[
                    tex_entry(0, self.pool.view(prev_frame)),
                    tex_entry(1, self.pool.view(curr_frame)),
                    tex_entry(2, self.pool.view("flow_tex")),
                    tex_entry(3, self.pool.view(prev_ui)),
                    samp_entry(4, &self.linear_sampler),
                    tex_entry(5, self.pool.view("ui_raw")),
                    buf_entry(6, &self.ui_detect_ubo),
                    buf_entry(7, &self.tile_flags_buf),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-ui-detect"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ui_detect_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.src_width.div_ceil(16), self.src_height.div_ceil(16), 1);
        }

        // Pass 5: Morphological Dilation (ui_raw -> next_ui)
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ui-dilate-bg"),
                layout: &self.ui_dilate_pipeline.get_bind_group_layout(0),
                entries: &[
                    tex_entry(0, self.pool.view("ui_raw")),
                    tex_entry(1, self.pool.view(next_ui)),
                    buf_entry(2, &self.dilate_ubo),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-ui-dilate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ui_dilate_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.src_width.div_ceil(16), self.src_height.div_ceil(16), 1);
        }

        if readback_stats {
            encoder.copy_buffer_to_buffer(
                &self.stats_buf,
                0,
                &self.stats_readback_buf,
                0,
                std::mem::size_of::<FrameStatsReadback>() as u64,
            );
        }

        queue.submit(Some(encoder.finish()));
        self.ui_ping_pong = !self.ui_ping_pong;

        let mut stats = FrameStatsReadback {
            active_tiles: total_tiles,
            total_tiles,
            ..Default::default()
        };
        if readback_stats {
            let slice = self.stats_readback_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            device.poll(wgpu::Maintain::Wait);
            if matches!(rx.recv(), Ok(Ok(()))) {
                let data = slice.get_mapped_range();
                stats = *bytemuck::from_bytes::<FrameStatsReadback>(&data);
                drop(data);
                self.stats_readback_buf.unmap();
            }
        }

        FrameTelemetry {
            active_tiles: stats.active_tiles,
            total_tiles: stats.total_tiles,
            scene_cut_tiles: stats.scene_cut_tiles,
            mean_abs_diff: stats.mean_abs_diff(),
            gpu_submit_ms: t0.elapsed().as_secs_f32() * 1000.0,
        }
    }

    /// Synthesize interpolated frame at `alpha` in `[0, 1]`, run fused Upscale+CAS+InlineOSD,
    /// and optionally present to the SwapChain target view.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_and_present(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cfg: &AppConfig,
        alpha: f32,
        captured_fps: u32,
        output_fps: u32,
        active_tile_pct: u32,
        flow_scale_pct: u32,
        debug_view: bool,
        target_surface_view: Option<&wgpu::TextureView>,
    ) {
        self.synthesize_and_present_with_compare(
            device,
            queue,
            cfg,
            alpha,
            captured_fps,
            output_fps,
            active_tile_pct,
            flow_scale_pct,
            debug_view,
            0.5,
            0,
            target_surface_view,
        );
    }

    /// Extended synthesis entry supporting live A/B split-curtain comparison mode.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_and_present_with_compare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cfg: &AppConfig,
        alpha: f32,
        captured_fps: u32,
        output_fps: u32,
        active_tile_pct: u32,
        flow_scale_pct: u32,
        debug_view: bool,
        split_x: f32,
        compare_mode: u32,
        target_surface_view: Option<&wgpu::TextureView>,
    ) {
        let (prev_frame, curr_frame) = self.prev_and_curr_frame_names();
        // After analyze_motion_and_ui toggled ui_ping_pong, current packed UI is prev_ui_name
        let (curr_ui_packed, _) = self.prev_and_curr_ui_names();

        queue.write_buffer(
            &self.synth_ubo,
            0,
            bytemuck::bytes_of(&SynthParamsUniform {
                width: self.src_width,
                height: self.src_height,
                tiles_x: self.tiles_x,
                tiles_y: self.tiles_y,
                alpha: alpha.clamp(0.0, 1.0),
                scene_cut_threshold: cfg.frame_gen.scene_cut_threshold,
                top_layer_composite: u32::from(cfg.ui_protection.top_layer_composite),
                debug_view: u32::from(debug_view),
                split_x: split_x.clamp(0.0, 1.0),
                compare_mode,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        queue.write_buffer(
            &self.upscale_ubo,
            0,
            bytemuck::bytes_of(&UpscaleOsdParamsUniform {
                src_width: self.src_width,
                src_height: self.src_height,
                dst_width: self.dst_width,
                dst_height: self.dst_height,
                mode: cfg.upscaling_mode_id(),
                sharpness: cfg.upscaling.sharpness,
                show_fps: u32::from(cfg.osd.show_fps),
                osd_scale: cfg.osd.scale,
                captured_fps,
                output_fps,
                active_tile_pct,
                flow_scale_pct,
            }),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fluidscale-synth-present-encoder"),
        });

        // Pass 6: Occlusion-Aware Frame Synthesis -> synth_tex (aliases dead ui_raw slot!)
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("synth-bg"),
                layout: &self.synth_pipeline.get_bind_group_layout(0),
                entries: &[
                    tex_entry(0, self.pool.view(prev_frame)),
                    tex_entry(1, self.pool.view(curr_frame)),
                    tex_entry(2, self.pool.view("flow_tex")),
                    tex_entry(3, self.pool.view(curr_ui_packed)),
                    samp_entry(4, &self.linear_sampler),
                    tex_entry(5, self.pool.view("synth_tex")),
                    buf_entry(6, &self.synth_ubo),
                    buf_entry(7, &self.tile_flags_buf),
                    buf_entry(8, &self.stats_buf),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-frame-synth"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.synth_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.src_width.div_ceil(16), self.src_height.div_ceil(16), 1);
        }

        // Pass 7: Fused Spatial Upscale + CAS + Zero-D2D1 Inline Bitmap FPS OSD -> output_tex
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("upscale-bg"),
                layout: &self.upscale_pipeline.get_bind_group_layout(0),
                entries: &[
                    tex_entry(0, self.pool.view("synth_tex")),
                    samp_entry(1, &self.linear_sampler),
                    tex_entry(2, self.pool.view("output_tex")),
                    buf_entry(3, &self.upscale_ubo),
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-upscale-cas-osd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.upscale_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.dst_width.div_ceil(16), self.dst_height.div_ceil(16), 1);
        }

        // Pass 8: Present to SwapChain surface if provided
        if let (Some(target), Some(present_pipe)) = (target_surface_view, &self.present_pipeline) {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("present-bg"),
                layout: &present_pipe.get_bind_group_layout(0),
                entries: &[
                    tex_entry(0, self.pool.view("output_tex")),
                    samp_entry(1, &self.linear_sampler),
                    buf_entry(2, &self.present_ubo),
                ],
            });
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass-present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(present_pipe);
            rpass.set_bind_group(0, &bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

fn create_compute(device: &wgpu::Device, label: &str, wgsl: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_present_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("present-shader"),
        source: wgpu::ShaderSource::Wgsl(PRESENT_WGSL.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("present-pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn make_ubo<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, data: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn tex_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn samp_entry(binding: u32, sampler: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    }
}

/// Compute exact letterbox / pillarbox UV transform mapping a `content_w × content_h` image
/// onto a `surf_w × surf_h` SwapChain surface (or 1:1 stretch if `preserve_aspect == false`).
pub fn compute_letterbox_params(
    content_w: u32,
    content_h: u32,
    surf_w: u32,
    surf_h: u32,
    preserve_aspect: bool,
) -> PresentParamsUniform {
    if !preserve_aspect || content_w == 0 || content_h == 0 || surf_w == 0 || surf_h == 0 {
        return PresentParamsUniform {
            scale_x: 1.0,
            scale_y: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        };
    }
    let src_ar = content_w as f32 / content_h as f32;
    let dst_ar = surf_w as f32 / surf_h as f32;
    if (src_ar - dst_ar).abs() < 0.002 {
        PresentParamsUniform {
            scale_x: 1.0,
            scale_y: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    } else if src_ar > dst_ar {
        let scale_y = dst_ar / src_ar;
        PresentParamsUniform {
            scale_x: 1.0,
            scale_y,
            offset_x: 0.0,
            offset_y: (1.0 - scale_y) * 0.5,
        }
    } else {
        let scale_x = src_ar / dst_ar;
        PresentParamsUniform {
            scale_x,
            scale_y: 1.0,
            offset_x: (1.0 - scale_x) * 0.5,
            offset_y: 0.0,
        }
    }
}

/// Compute largest pixel dimensions fitting `src_w × src_h` inside `max_w × max_h` while keeping aspect ratio.
pub fn fit_dimensions_preserve_aspect(src_w: u32, src_h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if src_w == 0 || src_h == 0 || max_w == 0 || max_h == 0 {
        return (max_w.max(16), max_h.max(16));
    }
    let sx = max_w as f64 / src_w as f64;
    let sy = max_h as f64 / src_h as f64;
    let scale = sx.min(sy);
    let w = ((src_w as f64 * scale).round() as u32).clamp(16, max_w);
    let h = ((src_h as f64 * scale).round() as u32).clamp(16, max_h);
    (w, h)
}

fn buf_entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::gpu::context::GpuContext;

    fn readback_output_rgba(
        pipe: &FluidPipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<u8> {
        let w = pipe.dst_width;
        let h = pipe.dst_height;
        let unpadded_bpr = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-output-readback"),
            size: (padded_bpr * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test-readback-enc"),
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: pipe.pool.texture("output_tex"),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let mapped = slice.get_mapped_range();
        let mut out = vec![0u8; (w * h * 4) as usize];
        for y in 0..h as usize {
            let src_start = y * padded_bpr as usize;
            let dst_start = y * unpadded_bpr as usize;
            out[dst_start..dst_start + unpadded_bpr as usize]
                .copy_from_slice(&mapped[src_start..src_start + unpadded_bpr as usize]);
        }
        drop(mapped);
        readback.unmap();
        out
    }

    fn make_shifted_pattern(w: u32, h: u32, shift_x: i32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let sx = (x as i32 - shift_x).rem_euclid(w as i32) as f32;
                let sy = y as f32;
                // Multi-frequency rich texture (avoids periodic aliasing over 16px shift)
                let v1 = ((sx * 0.09).sin() * 0.5 + 0.5) * 140.0;
                let v2 = ((sx * 0.037 + sy * 0.06).cos() * 0.5 + 0.5) * 90.0;
                let bar = if ((sx as i32) % 97) < 20 { 25.0 } else { 0.0 };
                let r = (v1 + bar).clamp(0.0, 255.0) as u8;
                let g = (v2 + v1 * 0.4).clamp(0.0, 255.0) as u8;
                let b = (255.0 - v1 * 0.6).clamp(0.0, 255.0) as u8;
                let i = ((y * w + x) * 4) as usize;
                buf[i] = r;
                buf[i + 1] = g;
                buf[i + 2] = b;
                buf[i + 3] = 255;
            }
        }
        buf
    }

    #[test]
    fn gpu_optical_flow_tracks_fast_16px_shift_accurately() {
        let mut cfg = AppConfig::default();
        cfg.osd.show_fps = false;
        cfg.upscaling.algorithm = "off".into();
        cfg.frame_gen.scene_cut_threshold = 0.99; // don't trigger scene-cut on synthetic full-frame shift
        let Ok(gpu) = GpuContext::new(&cfg.gpu, None) else {
            return; // skip cleanly on headless CI without GPU
        };
        let (w, h) = (256u32, 128u32);
        let mut pipe = FluidPipeline::new(&gpu.device, &cfg, w, h, w, h, 0.5, None).unwrap();

        let f0 = make_shifted_pattern(w, h, 0);
        let f1 = make_shifted_pattern(w, h, 16);
        let gt_mid = make_shifted_pattern(w, h, 8);

        pipe.upload_source_frame(&gpu.queue, &f0);
        pipe.upload_source_frame(&gpu.queue, &f1);
        let _ = pipe.analyze_motion_and_ui(&gpu.device, &gpu.queue, &cfg, None, true);
        pipe.synthesize_and_present(&gpu.device, &gpu.queue, &cfg, 0.5, 30, 60, 100, 50, false, None);
        let synth = readback_output_rgba(&pipe, &gpu.device, &gpu.queue);

        // Compare interior [32..224, 16..112] against ground-truth +8px frame vs naive 50/50 blend
        let mut flow_err = 0.0f64;
        let mut naive_err = 0.0f64;
        let mut count = 0u64;
        for y in 16..112 {
            for x in 32..224 {
                let i = ((y * w + x) * 4) as usize;
                for c in 0..3 {
                    let s = synth[i + c] as f64;
                    let gt = gt_mid[i + c] as f64;
                    let naive = 0.5 * (f0[i + c] as f64 + f1[i + c] as f64);
                    flow_err += (s - gt).abs();
                    naive_err += (naive - gt).abs();
                }
                count += 3;
            }
        }
        let flow_mae = flow_err / count as f64;
        let naive_mae = naive_err / count as f64;
        println!("GPU Optical Flow +16px shift test: flow_MAE={flow_mae:.2}, naive_blend_MAE={naive_mae:.2}");
        assert!(
            flow_mae < 12.0 && flow_mae < naive_mae * 0.45,
            "Expected accurate +16px hierarchical flow tracking (flow_mae={flow_mae:.2} vs naive_mae={naive_mae:.2})"
        );
    }

    #[test]
    fn gpu_tile_skip_zeroes_static_regions_and_clears_stale_tiles() {
        let mut cfg = AppConfig::default();
        cfg.osd.show_fps = false;
        let Ok(gpu) = GpuContext::new(&cfg.gpu, None) else {
            return;
        };
        let (w, h) = (256u32, 128u32);
        let mut pipe = FluidPipeline::new(&gpu.device, &cfg, w, h, w, h, 0.5, None).unwrap();

        // Top half completely static, bottom half shifts by +12px
        let mut f0 = make_shifted_pattern(w, h, 0);
        let mut f1 = make_shifted_pattern(w, h, 12);
        for y in 0..64 {
            let row = (y * w * 4) as usize..(y * w * 4 + w * 4) as usize;
            f0[row.clone()].fill(40);
            f1[row].fill(40);
        }

        pipe.upload_source_frame(&gpu.queue, &f0);
        pipe.upload_source_frame(&gpu.queue, &f1);
        let t1 = pipe.analyze_motion_and_ui(&gpu.device, &gpu.queue, &cfg, None, true);
        let skipped_pct = (1.0 - t1.active_tiles as f64 / t1.total_tiles.max(1) as f64) * 100.0;
        println!("GPU Tile-Skip test: active={}/{}, skipped={skipped_pct:.1}%", t1.active_tiles, t1.total_tiles);
        assert!(skipped_pct >= 45.0, "Expected >=45% tiles skipped on half-static frame, got {skipped_pct:.1}%");

        // Now send an identical static frame: active_tiles must drop to 0
        pipe.upload_source_frame(&gpu.queue, &f1);
        let t2 = pipe.analyze_motion_and_ui(&gpu.device, &gpu.queue, &cfg, None, true);
        assert_eq!(t2.active_tiles, 0, "Expected 0 active tiles when scene stops moving");
    }

    #[test]
    fn gpu_ui_anti_jello_locks_static_hud_over_moving_background() {
        let mut cfg = AppConfig::default();
        cfg.osd.show_fps = false;
        cfg.upscaling.algorithm = "off".into();
        cfg.ui_protection.top_layer_composite = true;
        cfg.frame_gen.scene_cut_threshold = 0.99;
        let Ok(gpu) = GpuContext::new(&cfg.gpu, None) else {
            return;
        };
        let (w, h) = (256u32, 128u32);
        let mut pipe = FluidPipeline::new(&gpu.device, &cfg, w, h, w, h, 0.5, None).unwrap();

        // Render 6 frames of fast-moving background with a crisp bright-white static HUD square [40..88, 40..88]
        for f in 0..6i32 {
            let mut img = make_shifted_pattern(w, h, f * 10);
            for y in 40..88 {
                for x in 40..88 {
                    let i = ((y * w + x) * 4) as usize;
                    img[i] = 255;
                    img[i + 1] = 255;
                    img[i + 2] = 255;
                    img[i + 3] = 255;
                }
            }
            pipe.upload_source_frame(&gpu.queue, &img);
            if f >= 1 {
                let _ = pipe.analyze_motion_and_ui(&gpu.device, &gpu.queue, &cfg, None, false);
            }
        }

        // Synthesize midpoint alpha=0.5 and verify HUD interior remains solid crisp white (zero jello smear)
        pipe.synthesize_and_present(&gpu.device, &gpu.queue, &cfg, 0.5, 30, 60, 100, 50, false, None);
        let synth = readback_output_rgba(&pipe, &gpu.device, &gpu.queue);

        let mut min_channel = 255u8;
        for y in 46..82 {
            for x in 46..82 {
                let i = ((y * w + x) * 4) as usize;
                min_channel = min_channel.min(synth[i]).min(synth[i + 1]).min(synth[i + 2]);
            }
        }
        println!("GPU UI Anti-Jello test: min HUD interior pixel value at alpha=0.5 is {min_channel}");
        assert!(
            min_channel >= 245,
            "Expected static HUD over moving background to stay crisp white (>=245), got {min_channel}"
        );
    }

    #[test]
    fn gpu_alpha_endpoints_and_scene_cut_protection_work_correctly() {
        let mut cfg = AppConfig::default();
        cfg.osd.show_fps = false;
        cfg.upscaling.algorithm = "off".into();
        cfg.frame_gen.scene_cut_threshold = 0.65;
        let Ok(gpu) = GpuContext::new(&cfg.gpu, None) else {
            return;
        };
        let (w, h) = (128u32, 64u32);
        let mut pipe = FluidPipeline::new(&gpu.device, &cfg, w, h, w, h, 0.5, None).unwrap();

        // 1. Normal non-cut frame transition (80 -> 120, delta=0.157 < 0.25 scene-cut tile threshold):
        //    alpha=0.0 MUST output prev_frame (80), NOT curr_frame (120)!
        //    alpha=1.0 MUST output curr_frame (120)!
        let f0 = vec![80u8; (w * h * 4) as usize];
        let f1 = vec![120u8; (w * h * 4) as usize];
        pipe.upload_source_frame(&gpu.queue, &f0);
        pipe.upload_source_frame(&gpu.queue, &f1);
        let _ = pipe.analyze_motion_and_ui(&gpu.device, &gpu.queue, &cfg, None, false);

        pipe.synthesize_and_present(&gpu.device, &gpu.queue, &cfg, 0.0, 30, 60, 100, 50, false, None);
        let out0 = readback_output_rgba(&pipe, &gpu.device, &gpu.queue);
        pipe.synthesize_and_present(&gpu.device, &gpu.queue, &cfg, 1.0, 30, 60, 100, 50, false, None);
        let out1 = readback_output_rgba(&pipe, &gpu.device, &gpu.queue);

        let mid_idx = ((32 * w + 64) * 4) as usize;
        assert!(
            out0[mid_idx].abs_diff(80) <= 2,
            "alpha=0.0 must match prev_frame (80), got {}",
            out0[mid_idx]
        );
        assert!(
            out1[mid_idx].abs_diff(120) <= 2,
            "alpha=1.0 must match curr_frame (120), got {}",
            out1[mid_idx]
        );

        // 2. Sudden full-screen scene cut (40 -> 220, delta=0.71 > 0.25):
        //    even at alpha=0.5, scene-cut protection MUST pass through crisp curr_frame (220)!
        let cut_prev = vec![40u8; (w * h * 4) as usize];
        let cut_curr = vec![220u8; (w * h * 4) as usize];
        pipe.upload_source_frame(&gpu.queue, &cut_prev);
        pipe.upload_source_frame(&gpu.queue, &cut_curr);
        let telem = pipe.analyze_motion_and_ui(&gpu.device, &gpu.queue, &cfg, None, true);
        assert_eq!(telem.scene_cut_tiles, telem.total_tiles);
        pipe.synthesize_and_present(&gpu.device, &gpu.queue, &cfg, 0.5, 30, 60, 100, 50, false, None);
        let out_cut = readback_output_rgba(&pipe, &gpu.device, &gpu.queue);
        assert!(
            out_cut[mid_idx].abs_diff(220) <= 2,
            "scene cut at alpha=0.5 must pass through crisp curr_frame (220), got {}",
            out_cut[mid_idx]
        );
    }

    #[test]
    fn aspect_ratio_letterbox_preserves_16_9_on_16_10_laptop_screen() {
        // 1920x1080 (16:9) on 2560x1600 (16:10) should fit to 2560x1440 with 5% top/bottom black bars
        let (fw, fh) = fit_dimensions_preserve_aspect(1920, 1080, 2560, 1600);
        assert_eq!((fw, fh), (2560, 1440));
        let p = compute_letterbox_params(fw, fh, 2560, 1600, true);
        assert!((p.scale_x - 1.0).abs() < 1e-4);
        assert!((p.scale_y - 0.9).abs() < 1e-3);
        assert!((p.offset_y - 0.05).abs() < 1e-3);

        // Stretch mode maps 1:1 with zero offset
        let s = compute_letterbox_params(fw, fh, 2560, 1600, false);
        assert_eq!((s.scale_x, s.scale_y, s.offset_x, s.offset_y), (1.0, 1.0, 0.0, 0.0));
    }
}
