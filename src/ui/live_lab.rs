//! Embedded Zero-Copy GPU Multi-Scene Live Lab running directly inside eframe's wgpu device!
//!
//! Renders 4 high-contrast GPU showcase scenes (`shaders/demo_scenes.wgsl`) at selectable
//! low input framerates (15 / 20 / 30 FPS) and runs the full 7-stage FluidPipeline + A/B split curtain
//! directly on the GPU so smoothness, upscaling sharpness, Tile-Skip, and UI Anti-Jello differences
//! are immediately unmistakable.

use crate::config::{AppConfig, GameProfile};
use crate::gpu::context::DynamicFlowScale;
use crate::gpu::pipeline::FluidPipeline;
use crate::pacing::FramePacer;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const DEMO_SCENES_WGSL: &str = include_str!("../../shaders/demo_scenes.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DemoSceneUniform {
    width: u32,
    height: u32,
    scene_id: u32,
    scene_cut: u32,
    time_sec: f32,
    speed: f32,
    _pad0: u32,
    _pad1: u32,
}

pub struct LiveGpuLab {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: FluidPipeline,
    scene_pipeline: wgpu::ComputePipeline,
    scene_ubo: wgpu::Buffer,

    pub texture_id: egui::TextureId,
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,

    frame_idx: u32,
    sim_time_sec: f32,
    last_input_tick: Instant,
    pacer: FramePacer,
    flow_ctrl: DynamicFlowScale,

    built_flow_scale: f32,
    built_aliasing: bool,
    last_cfg_base_scale: f32,
    last_cfg_dyn_flow: bool,
    pending_dyn_scale: Option<f32>,
    last_flow_rebuild: Instant,
    last_scene_id: u32,

    // Interactive lab controls
    pub paused: bool,
    pub fg_enabled: bool,
    pub trigger_scene_cut: bool,
    /// 0=Highway Racing, 1=FPS Pan+Crosshair, 2=RPG Half-Static, 3=Sharpness Target
    pub scene_id: u32,
    /// Simulated native input FPS (15, 20, 30) — lower FPS makes FrameGen A/B difference crystal clear!
    pub simulated_input_fps: u32,
    /// Motion speed multiplier (0.6, 1.0, 1.6)
    pub motion_speed: f32,
    /// Interactive A/B split curtain [0.0..1.0] & compare mode (0=full, 1=heatmap, 2=raw low-fps, 3=no-ui-shield)
    pub split_x: f32,
    pub compare_mode: u32,

    // Telemetry history
    pub tile_skip_history: VecDeque<f32>,
    pub gpu_ms_history: VecDeque<f32>,
    pub latest_skipped_pct: f32,
    pub latest_gpu_ms: f32,
    pub latest_scene_cuts: u32,
    pub latest_naive_mb: f32,
    pub latest_aliased_mb: f32,
    pub latest_vram_saved_pct: f32,
    pub latest_physical_slots: usize,
}

impl LiveGpuLab {
    pub fn new(
        render_state: &eframe::egui_wgpu::RenderState,
        cfg: &AppConfig,
    ) -> Result<Self> {
        // Higher resolution zero-copy GPU showcase (768x432 -> 1152x648)
        let src_width = 768u32;
        let src_height = 432u32;
        let dst_width = 1152u32;
        let dst_height = 648u32;

        let device = render_state.device.clone();
        let queue = render_state.queue.clone();
        let flow_ctrl = DynamicFlowScale::new(&cfg.gpu);
        let built_flow_scale = flow_ctrl.current();
        let built_aliasing = cfg.gpu.enable_vram_aliasing;

        let pipeline = FluidPipeline::new(
            &device,
            cfg,
            src_width,
            src_height,
            dst_width,
            dst_height,
            built_flow_scale,
            None,
        )?;

        let scene_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("demo-scenes-wgsl"),
            source: wgpu::ShaderSource::Wgsl(DEMO_SCENES_WGSL.into()),
        });
        let scene_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("demo-scenes-pipeline"),
            layout: None,
            module: &scene_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let scene_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("demo-scene-ubo"),
            contents: bytemuck::bytes_of(&DemoSceneUniform {
                width: src_width,
                height: src_height,
                scene_id: 0,
                scene_cut: 0,
                time_sec: 0.0,
                speed: 1.0,
                _pad0: 0,
                _pad1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let texture_id = render_state.renderer.write().register_native_texture(
            &device,
            pipeline.pool.view("output_tex"),
            wgpu::FilterMode::Linear,
        );

        let latest_naive_mb = pipeline.pool.plan.naive_vram_bytes as f32 / (1024.0 * 1024.0);
        let latest_aliased_mb = pipeline.pool.plan.aliased_vram_bytes as f32 / (1024.0 * 1024.0);
        let latest_vram_saved_pct = pipeline.pool.plan.saved_percent() as f32;
        let latest_physical_slots = pipeline.pool.plan.physical_specs.len();

        let mut lab = Self {
            device,
            queue,
            pipeline,
            scene_pipeline,
            scene_ubo,
            texture_id,
            src_width,
            src_height,
            dst_width,
            dst_height,
            frame_idx: 0,
            sim_time_sec: 0.0,
            last_input_tick: Instant::now(),
            pacer: FramePacer::new(&cfg.pacing),
            flow_ctrl,
            built_flow_scale,
            built_aliasing,
            last_cfg_base_scale: cfg.gpu.base_flow_scale,
            last_cfg_dyn_flow: cfg.gpu.dynamic_flow_scale,
            pending_dyn_scale: None,
            last_flow_rebuild: Instant::now(),
            last_scene_id: 0,
            paused: false,
            fg_enabled: true,
            trigger_scene_cut: false,
            scene_id: 0,
            simulated_input_fps: 20, // 20 FPS -> 120 FPS makes smoothness contrast unmistakable!
            motion_speed: 1.0,
            split_x: 0.50,
            compare_mode: 2, // Default to Raw Low-FPS vs Interpolated so difference jumps out immediately!
            tile_skip_history: VecDeque::with_capacity(96),
            gpu_ms_history: VecDeque::with_capacity(96),
            latest_skipped_pct: 0.0,
            latest_gpu_ms: 0.0,
            latest_scene_cuts: 0,
            latest_naive_mb,
            latest_aliased_mb,
            latest_vram_saved_pct,
            latest_physical_slots,
        };

        // Prime both ping-pong frame buffers on GPU
        lab.render_scene_frame_gpu(0.0, false);
        lab.render_scene_frame_gpu(0.05, false);
        Ok(lab)
    }

    fn render_scene_frame_gpu(&mut self, time_sec: f32, scene_cut: bool) {
        self.queue.write_buffer(
            &self.scene_ubo,
            0,
            bytemuck::bytes_of(&DemoSceneUniform {
                width: self.src_width,
                height: self.src_height,
                scene_id: self.scene_id,
                scene_cut: u32::from(scene_cut),
                time_sec,
                speed: self.motion_speed,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        let dst_view = self.pipeline.advance_and_get_write_frame_view();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("demo-scene-bg"),
            layout: &self.scene_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.scene_ubo.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("demo-scene-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-demo-scene"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scene_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                self.src_width.div_ceil(16),
                self.src_height.div_ceil(16),
                1,
            );
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn sync_pipeline_if_needed(
        &mut self,
        render_state: &eframe::egui_wgpu::RenderState,
        cfg: &AppConfig,
    ) {
        // Re-seed dynamic controller when user manually changes base scale or dynamic toggle
        if (cfg.gpu.base_flow_scale - self.last_cfg_base_scale).abs() >= 0.01
            || cfg.gpu.dynamic_flow_scale != self.last_cfg_dyn_flow
        {
            self.flow_ctrl = DynamicFlowScale::new(&cfg.gpu);
            self.last_cfg_base_scale = cfg.gpu.base_flow_scale;
            self.last_cfg_dyn_flow = cfg.gpu.dynamic_flow_scale;
            self.pending_dyn_scale = Some(self.flow_ctrl.current());
        }

        let target_scale = self
            .pending_dyn_scale
            .take()
            .unwrap_or(self.built_flow_scale)
            .clamp(0.25, 1.0);
        if (target_scale - self.built_flow_scale).abs() >= 0.04
            || cfg.gpu.enable_vram_aliasing != self.built_aliasing
        {
            if let Ok(new_pipe) = FluidPipeline::new(
                &self.device,
                cfg,
                self.src_width,
                self.src_height,
                self.dst_width,
                self.dst_height,
                target_scale,
                None,
            ) {
                render_state
                    .renderer
                    .write()
                    .update_egui_texture_from_wgpu_texture(
                        &self.device,
                        new_pipe.pool.view("output_tex"),
                        wgpu::FilterMode::Linear,
                        self.texture_id,
                    );
                self.latest_naive_mb =
                    new_pipe.pool.plan.naive_vram_bytes as f32 / (1024.0 * 1024.0);
                self.latest_aliased_mb =
                    new_pipe.pool.plan.aliased_vram_bytes as f32 / (1024.0 * 1024.0);
                self.latest_vram_saved_pct = new_pipe.pool.plan.saved_percent() as f32;
                self.latest_physical_slots = new_pipe.pool.plan.physical_specs.len();
                self.pipeline = new_pipe;
                self.built_flow_scale = target_scale;
                self.built_aliasing = cfg.gpu.enable_vram_aliasing;
                self.last_flow_rebuild = Instant::now();
                self.render_scene_frame_gpu(self.sim_time_sec, false);
                self.render_scene_frame_gpu(self.sim_time_sec + 0.03, false);
            }
        }
    }

    pub fn step(&mut self, cfg: &AppConfig, profile: Option<&GameProfile>) {
        // Clean reset on scene switch: flush old HUD hysteresis mask, flow field, and sparklines immediately
        if self.scene_id != self.last_scene_id {
            self.last_scene_id = self.scene_id;
            self.pipeline.reset_history(&self.queue);
            self.tile_skip_history.clear();
            self.gpu_ms_history.clear();
            let in_fps = self.simulated_input_fps.clamp(10, 60) as f32;
            let step_dt = 1.0 / in_fps;
            self.render_scene_frame_gpu(self.sim_time_sec, false);
            self.sim_time_sec += step_dt;
            self.render_scene_frame_gpu(self.sim_time_sec, false);
            let telem = self
                .pipeline
                .analyze_motion_and_ui(&self.device, &self.queue, cfg, profile, true);
            let active_ratio = telem.active_tiles as f32 / telem.total_tiles.max(1) as f32;
            self.latest_skipped_pct = (1.0 - active_ratio).clamp(0.0, 1.0) * 100.0;
            self.last_input_tick = Instant::now();
        }

        if self.paused {
            return;
        }

        let now = Instant::now();
        let in_fps = self.simulated_input_fps.clamp(10, 60) as f64;
        let input_period = Duration::from_secs_f64(1.0 / in_fps);

        if now.duration_since(self.last_input_tick) >= input_period {
            let dt = now.duration_since(self.last_input_tick).as_secs_f32().min(0.12);
            self.last_input_tick = now;
            self.frame_idx += 1;
            self.sim_time_sec += dt;

            let cut = std::mem::take(&mut self.trigger_scene_cut);
            self.render_scene_frame_gpu(self.sim_time_sec, cut);

            let do_readback = self.frame_idx % 3 == 0 || cut;
            let t0 = Instant::now();
            let telem = self.pipeline.analyze_motion_and_ui(
                &self.device,
                &self.queue,
                cfg,
                profile,
                do_readback,
            );
            let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;

            if do_readback {
                let active_ratio = telem.active_tiles as f32 / telem.total_tiles.max(1) as f32;
                let skipped_pct = (1.0 - active_ratio).clamp(0.0, 1.0) * 100.0;
                self.latest_skipped_pct = skipped_pct;
                self.latest_gpu_ms = elapsed_ms;
                if telem.scene_cut_tiles as f32 / telem.total_tiles.max(1) as f32
                    > cfg.frame_gen.scene_cut_threshold
                {
                    self.latest_scene_cuts += 1;
                }
                if let Some(new_scale) = self.flow_ctrl.update(elapsed_ms, 8.33) {
                    if self.last_flow_rebuild.elapsed() >= Duration::from_millis(1500) {
                        self.pending_dyn_scale = Some(new_scale);
                    }
                }
                push_bounded(&mut self.tile_skip_history, skipped_pct, 90);
                push_bounded(&mut self.gpu_ms_history, elapsed_ms, 90);
            }
            self.pacer
                .on_captured_frame(telem.mean_abs_diff.max(0.01), Some(1.0 / in_fps));
        }

        let effective_fg = self.fg_enabled && !cfg.frame_gen.mode.eq_ignore_ascii_case("off");
        let alpha = if effective_fg {
            (self.last_input_tick.elapsed().as_secs_f64() * in_fps).clamp(0.0, 1.0) as f32
        } else {
            0.0 // Hold stepped input frame when frame-gen is turned off
        };

        let active_pct = (100.0 - self.latest_skipped_pct).clamp(0.0, 100.0).round() as u32;
        self.pipeline.synthesize_and_present_with_compare(
            &self.device,
            &self.queue,
            cfg,
            alpha,
            self.simulated_input_fps,
            self.pacer.output_fps.round() as u32,
            active_pct,
            (self.built_flow_scale * 100.0) as u32,
            false,
            self.split_x,
            self.compare_mode,
            None,
        );
        self.pacer.on_output_presented();
    }

    pub fn captured_fps(&self) -> f64 {
        self.simulated_input_fps as f64
    }

    pub fn output_fps(&self) -> f64 {
        self.pacer.output_fps
    }

    pub fn flow_dims(&self) -> (u32, u32) {
        (self.pipeline.flow_width, self.pipeline.flow_height)
    }

    pub fn current_flow_scale(&self) -> f32 {
        self.built_flow_scale
    }
}

fn push_bounded(buf: &mut VecDeque<f32>, val: f32, max_len: usize) {
    if buf.len() >= max_len {
        buf.pop_front();
    }
    buf.push_back(val);
}
