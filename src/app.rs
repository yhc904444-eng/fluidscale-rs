use crate::capture::dxgi::DxgiDesktopCapturer;
use crate::capture::window_gdi::WindowGdiCapturer;
use crate::capture::FrameCapturer;
use crate::config::AppConfig;
use crate::gpu::context::{DynamicFlowScale, GpuContext};
use crate::gpu::pipeline::{fit_dimensions_preserve_aspect, FluidPipeline};
use crate::pacing::etw::PresentTimestampTracker;
use crate::pacing::timer::HybridPacerTimer;
use crate::pacing::FramePacer;
use crate::platform::win32::{
    configure_clickthrough_overlay, exclude_window_from_capture, get_monitor_rect_for_window,
    get_window_rect, is_window_alive, list_visible_windows, GpuPriorityGuard,
};
use anyhow::{anyhow, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowAttributes, WindowId};

/// Generate a synthetic moving game-like frame with:
/// - Upper 28% static twilight sky (tests [Opt 1] Tile-Skipping Indirect Dispatch!)
/// - Lower scrolling terrain + fast moving spheres/pillars (tests Optical Flow + Frame Synthesis!)
/// - Bottom-left translucent HUD panel + crisp center crosshair (tests [Opt 4] Motion-Gated UI Anti-Jello!)
pub fn render_synthetic_frame(buf: &mut [u8], w: u32, h: u32, frame_idx: u32, scene_cut: bool) {
    let t = frame_idx as f32 * 0.08;
    let scroll_x = (frame_idx * 9) as i32;
    let sky_end = (h as f32 * 0.28) as u32;

    for y in 0..h {
        let row_off = (y * w * 4) as usize;
        let is_sky = y < sky_end;
        for x in 0..w {
            let i = row_off + (x * 4) as usize;
            let (mut r, mut g, mut b): (u8, u8, u8);

            if scene_cut {
                // High-contrast orange/teal flash to trigger scene-cut detector
                r = 220;
                g = ((x * 255) / w) as u8;
                b = 40;
            } else if is_sky {
                // Completely static upper sky band -> 16x16 tiles here will be skipped by tile_activity.wgsl!
                let grad = (y * 255 / sky_end.max(1)) as u8;
                r = 14 + grad / 8;
                g = 22 + grad / 5;
                b = 48 + grad / 3;
            } else {
                // Scrolling parallax checker/mountain waves
                let sx = (x as i32 + scroll_x).rem_euclid(64);
                let sy = (y as i32).rem_euclid(64);
                let checker = ((sx / 32) ^ (sy / 32)) == 0;
                let wave = (((x as f32 * 0.025 + t * 2.0).sin() * 0.5 + 0.5) * 60.0) as u8;
                if checker {
                    r = 35 + wave;
                    g = 75 + wave / 2;
                    b = 90 + wave / 3;
                } else {
                    r = 24 + wave / 2;
                    g = 42 + wave / 3;
                    b = 58 + wave / 2;
                }

                // Fast-moving bright orb
                let orb_x = ((t * 140.0) % (w as f32)) as i32;
                let orb_y = (h as f32 * 0.56 + (t * 2.5).sin() * (h as f32 * 0.14)) as i32;
                let dx = x as i32 - orb_x;
                let dy = y as i32 - orb_y;
                if dx * dx + dy * dy < 38 * 38 {
                    r = 245;
                    g = 170;
                    b = 45;
                }
            }

            // Static bottom-left HUD panel (health bar + static label bars)
            let hud_x0 = (w as f32 * 0.025) as u32;
            let hud_x1 = (w as f32 * 0.25) as u32;
            let hud_y0 = (h as f32 * 0.83) as u32;
            let hud_y1 = (h as f32 * 0.95) as u32;
            if x >= hud_x0 && x < hud_x1 && y >= hud_y0 && y < hud_y1 {
                let border = x < hud_x0 + 2 || x + 2 >= hud_x1 || y < hud_y0 + 2 || y + 2 >= hud_y1;
                if border {
                    r = 210;
                    g = 230;
                    b = 250;
                } else {
                    // Translucent dark backing + bright HP bar
                    r = (r / 4).saturating_add(16);
                    g = (g / 4).saturating_add(22);
                    b = (b / 4).saturating_add(30);
                    let bar_y0 = hud_y0 + (hud_y1 - hud_y0) / 3;
                    let bar_y1 = hud_y0 + (hud_y1 - hud_y0) * 2 / 3;
                    let bar_x1 = hud_x0 + (hud_x1 - hud_x0) * 3 / 4;
                    if x >= hud_x0 + 10 && x < bar_x1 && y >= bar_y0 && y < bar_y1 {
                        r = 55;
                        g = 220;
                        b = 115;
                    }
                }
            }

            // Crisp static center crosshair (1px sharp lines - classic jello stress test!)
            let cx = w / 2;
            let cy = h / 2;
            let on_h = y.abs_diff(cy) <= 1 && x.abs_diff(cx) <= 14 && x.abs_diff(cx) >= 3;
            let on_v = x.abs_diff(cx) <= 1 && y.abs_diff(cy) <= 14 && y.abs_diff(cy) >= 3;
            if on_h || on_v {
                r = 255;
                g = 255;
                b = 255;
            }

            buf[i] = r;
            buf[i + 1] = g;
            buf[i + 2] = b;
            buf[i + 3] = 255;
        }
    }
}

/// Structured result returned by the real GPU benchmark (used by both CLI and Desktop UI).
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub gpu_name: String,
    pub backend: String,
    pub src_w: u32,
    pub src_h: u32,
    pub dst_w: u32,
    pub dst_h: u32,
    pub flow_w: u32,
    pub flow_h: u32,
    pub frames: u32,
    pub tile_skipped_pct: f64,
    pub naive_vram_mb: f64,
    pub aliased_vram_mb: f64,
    pub vram_saved_pct: f64,
    pub physical_slots: usize,
    pub avg_gpu_ms_per_pair: f64,
    pub avg_gpu_ms_per_out_frame: f64,
    pub scene_cuts_caught: u32,
    pub pll_slew: f64,
}

/// Headless GPU benchmark returning structured metrics verifying ALL 4 optimization pillars.
pub fn run_benchmark_report(
    cfg: &AppConfig,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    frames: u32,
) -> Result<BenchmarkReport> {
    let gpu = GpuContext::new(&cfg.gpu, None)?;
    let mut flow_ctrl = DynamicFlowScale::new(&cfg.gpu);
    let mut pipe = FluidPipeline::new(
        &gpu.device,
        cfg,
        src_w,
        src_h,
        dst_w,
        dst_h,
        flow_ctrl.current(),
        None,
    )?;

    let mut frame_buf = vec![0u8; (src_w * src_h * 4) as usize];
    let profile = cfg.match_profile("Default");
    let mut pacer = FramePacer::new(&cfg.pacing);

    render_synthetic_frame(&mut frame_buf, src_w, src_h, 0, false);
    pipe.upload_source_frame(&gpu.queue, &frame_buf);

    let mut total_active_tiles: u64 = 0;
    let mut total_tiles_acc: u64 = 0;
    let mut scene_cuts_detected: u32 = 0;
    let mut total_gpu_ms: f64 = 0.0;

    for f in 1..=frames {
        let is_cut = f == frames / 2;
        render_synthetic_frame(&mut frame_buf, src_w, src_h, f, is_cut);
        pipe.upload_source_frame(&gpu.queue, &frame_buf);

        let t0 = Instant::now();
        let telem = pipe.analyze_motion_and_ui(&gpu.device, &gpu.queue, cfg, profile, true);
        let active_pct = ((telem.active_tiles as f32 / telem.total_tiles.max(1) as f32) * 100.0)
            .round() as u32;

        for sub in 0..2u32 {
            let alpha = if sub == 0 { 0.5 } else { 1.0 };
            pipe.synthesize_and_present(
                &gpu.device,
                &gpu.queue,
                cfg,
                alpha,
                60,
                120,
                active_pct,
                (flow_ctrl.current() * 100.0) as u32,
                false,
                None,
            );
        }
        gpu.device.poll(wgpu::Maintain::Wait);
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;

        total_gpu_ms += step_ms;
        total_active_tiles += telem.active_tiles as u64;
        total_tiles_acc += telem.total_tiles as u64;
        if telem.scene_cut_tiles as f32 / telem.total_tiles.max(1) as f32
            > cfg.frame_gen.scene_cut_threshold
        {
            scene_cuts_detected += 1;
        }
        pacer.on_captured_frame(telem.mean_abs_diff, Some(1.0 / 60.0));
        let _ = flow_ctrl.update(step_ms as f32, 16.67);
    }

    let skipped_pct =
        (1.0 - (total_active_tiles as f64 / total_tiles_acc.max(1) as f64)) * 100.0;
    let avg_ms = total_gpu_ms / frames.max(1) as f64;

    Ok(BenchmarkReport {
        gpu_name: gpu.adapter_info.name,
        backend: format!("{:?}", gpu.adapter_info.backend),
        src_w,
        src_h,
        dst_w,
        dst_h,
        flow_w: pipe.flow_width,
        flow_h: pipe.flow_height,
        frames,
        tile_skipped_pct: skipped_pct,
        naive_vram_mb: pipe.pool.plan.naive_vram_bytes as f64 / (1024.0 * 1024.0),
        aliased_vram_mb: pipe.pool.plan.aliased_vram_bytes as f64 / (1024.0 * 1024.0),
        vram_saved_pct: pipe.pool.plan.saved_percent(),
        physical_slots: pipe.pool.plan.physical_specs.len(),
        avg_gpu_ms_per_pair: avg_ms,
        avg_gpu_ms_per_out_frame: avg_ms * 0.5,
        scene_cuts_caught: scene_cuts_detected,
        pll_slew: pacer.last_slew_factor(),
    })
}

/// CLI entrypoint printing the structured GPU benchmark report.
pub fn run_benchmark(
    cfg: &AppConfig,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    frames: u32,
) -> Result<()> {
    let r = run_benchmark_report(cfg, src_w, src_h, dst_w, dst_h, frames)?;
    println!("\n================ FluidScale-RS GPU Benchmark ================");
    println!("GPU Backend      : {} ({})", r.gpu_name, r.backend);
    println!(
        "Resolution       : {}x{} -> {}x{} (Flow {}x{})",
        r.src_w, r.src_h, r.dst_w, r.dst_h, r.flow_w, r.flow_h
    );
    println!(
        "[Opt 1] Tile Skip: {:.1}% static 16x16 tiles skipped via IndirectDispatch",
        r.tile_skipped_pct
    );
    println!(
        "[Opt 2] VRAM Pool: {:.2} MB naive -> {:.2} MB aliased (saved {:.1}%, {} physical slots)",
        r.naive_vram_mb, r.aliased_vram_mb, r.vram_saved_pct, r.physical_slots
    );
    println!(
        "[Opt 3] Pacing   : avg GPU {:.2} ms/pair ({:.2} ms/out frame, ~{:.0} FPS GPU cap) | PLL slew {:.4}",
        r.avg_gpu_ms_per_pair,
        r.avg_gpu_ms_per_out_frame,
        1000.0 / r.avg_gpu_ms_per_out_frame.max(0.01),
        r.pll_slew
    );
    println!(
        "[Opt 4] UI/Cut   : motion-gated hysteresis active | scene cuts caught: {}",
        r.scene_cuts_caught
    );
    println!("=============================================================\n");
    Ok(())
}

/// Interactive visual mode: either live synthetic Demo or real Windows Capture Overlay.
pub fn run_windowed(cfg: AppConfig, capture_target: Option<String>, is_demo: bool) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut runner = WindowedRunner::new(cfg, capture_target, is_demo)?;
    event_loop.run_app(&mut runner)?;
    Ok(())
}

struct WindowedRunner {
    cfg: AppConfig,
    is_demo: bool,
    capture_target: Option<String>,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    surface_cfg: Option<wgpu::SurfaceConfiguration>,
    gpu: Option<GpuContext>,
    pipeline: Option<FluidPipeline>,
    capturer: Option<Box<dyn FrameCapturer>>,
    _priority_guard: Option<GpuPriorityGuard>,
    etw_tracker: Option<PresentTimestampTracker>,
    pacer: FramePacer,
    timer: HybridPacerTimer,
    flow_ctrl: DynamicFlowScale,

    // Live runtime state
    synthetic_buf: Vec<u8>,
    src_frame_idx: u32,
    last_input_tick: Instant,
    next_present_deadline: Instant,
    active_tile_pct: u32,
    debug_view: bool,
    fg_enabled: bool,
    ui_protect_enabled: bool,
    trigger_scene_cut: bool,
    target_hwnd: Option<isize>,
    target_title: String,
    has_first_frame: bool,
    last_sample_luma: Vec<u8>,
    surface_format: Option<wgpu::TextureFormat>,
    last_flow_rebuild: Instant,
}

impl WindowedRunner {
    fn new(cfg: AppConfig, capture_target: Option<String>, is_demo: bool) -> Result<Self> {
        let pacer = FramePacer::new(&cfg.pacing);
        let timer = HybridPacerTimer::new(cfg.pacing.spin_wait_us);
        let flow_ctrl = DynamicFlowScale::new(&cfg.gpu);
        let fg_enabled = !cfg.frame_gen.mode.eq_ignore_ascii_case("off");
        let now = Instant::now();
        Ok(Self {
            cfg,
            is_demo,
            capture_target,
            window: None,
            surface: None,
            surface_cfg: None,
            gpu: None,
            pipeline: None,
            capturer: None,
            _priority_guard: None,
            etw_tracker: None,
            pacer,
            timer,
            flow_ctrl,
            synthetic_buf: Vec::new(),
            src_frame_idx: 0,
            last_input_tick: now,
            next_present_deadline: now,
            active_tile_pct: 100,
            debug_view: false,
            fg_enabled,
            ui_protect_enabled: true,
            trigger_scene_cut: false,
            target_hwnd: None,
            target_title: "Default".to_string(),
            has_first_frame: false,
            last_sample_luma: Vec::new(),
            surface_format: None,
            last_flow_rebuild: now,
        })
    }

    fn init_graphics(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let (src_w, src_h, dst_w, dst_h, win_title, overlay_pos, target_pid) = if self.is_demo {
            (
                960u32,
                540u32,
                1280u32,
                720u32,
                "FluidScale-RS Demo | [D] Debug Mask/Tiles | [F] FrameGen | [U] UI Anti-Jello | [Space] SceneCut"
                    .to_string(),
                None,
                0u32,
            )
        } else {
            // Real Windows capture mode
            let windows = list_visible_windows();
            let chosen = if let Some(ref needle) = self.capture_target {
                let n = needle.to_lowercase();
                windows
                    .iter()
                    .find(|w| w.title.to_lowercase().contains(&n))
                    .cloned()
                    .ok_or_else(|| anyhow!("No visible window matching '{needle}'"))?
            } else {
                windows
                    .first()
                    .cloned()
                    .ok_or_else(|| anyhow!("No visible desktop window found to capture"))?
            };

            self.target_hwnd = Some(chosen.hwnd);
            self.target_title = chosen.title.clone();

            info!(
                title = %chosen.title,
                pid = chosen.pid,
                size = format!("{}x{}", chosen.width, chosen.height),
                "Attaching capture target"
            );

            let win_rect = get_window_rect(chosen.hwnd).unwrap_or((
                chosen.x,
                chosen.y,
                chosen.width.max(16) as u32,
                chosen.height.max(16) as u32,
            ));

            let cap: Box<dyn FrameCapturer> =
                if self.cfg.capture.api.eq_ignore_ascii_case("window_gdi") {
                    Box::new(WindowGdiCapturer::new(chosen.hwnd)?)
                } else {
                    match DxgiDesktopCapturer::new_for_rect(Some(win_rect)) {
                        Ok(dxgi) => Box::new(dxgi),
                        Err(e) => {
                            warn!(
                                error = %e,
                                "DXGI Desktop Duplication unavailable; falling back to Window GDI capture"
                            );
                            Box::new(WindowGdiCapturer::new(chosen.hwnd)?)
                        }
                    }
                };
            let (cw, ch) = cap.dimensions();
            self.capturer = Some(cap);

            // When upscaling is enabled, fill the target window's monitor in physical pixels;
            // when upscaling is "off" (FrameGen-only), match the target window's physical rect 1:1.
            let (out_w, out_h, out_pos) = if !self.cfg.upscaling.algorithm.eq_ignore_ascii_case("off")
            {
                if let Some(mon) = get_monitor_rect_for_window(chosen.hwnd) {
                    (mon.width, mon.height, Some((mon.x, mon.y)))
                } else {
                    (cw, ch, Some((win_rect.0, win_rect.1)))
                }
            } else {
                (cw, ch, Some((win_rect.0, win_rect.1)))
            };

            (
                cw,
                ch,
                out_w,
                out_h,
                format!("FluidScale-RS Overlay - {}", chosen.title),
                out_pos,
                chosen.pid,
            )
        };

        let mut attrs = WindowAttributes::default()
            .with_title(win_title)
            .with_inner_size(PhysicalSize::new(dst_w, dst_h));
        if !self.is_demo {
            attrs = attrs.with_decorations(false).with_transparent(false);
        }
        let window = Arc::new(event_loop.create_window(attrs)?);

        if !self.is_demo {
            if let Some((ox, oy)) = overlay_pos {
                window.set_outer_position(PhysicalPosition::new(ox, oy));
            }
            if let Ok(wh) = window.window_handle() {
                if let RawWindowHandle::Win32(h) = wh.as_raw() {
                    let hwnd_raw = h.hwnd.get();
                    let _ = exclude_window_from_capture(hwnd_raw);
                    let _ = configure_clickthrough_overlay(hwnd_raw);
                }
            }
            self._priority_guard = Some(GpuPriorityGuard::elevate());
            self.etw_tracker = Some(PresentTimestampTracker::start(
                target_pid,
                self.cfg.pacing.enable_etw,
            ));
        }

        let backends = crate::gpu::context::parse_backends(&self.cfg.gpu.backend);
        let temp_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let surface = temp_instance.create_surface(window.clone())?;

        // Pick adapter compatible with surface, honoring Dual-GPU `preferred_adapter` if configured
        let preferred = self.cfg.gpu.preferred_adapter.trim().to_lowercase();
        let adapter = if !preferred.is_empty() {
            temp_instance
                .enumerate_adapters(backends)
                .into_iter()
                .find(|a| {
                    a.get_info().name.to_lowercase().contains(&preferred)
                        && a.is_surface_supported(&surface)
                })
                .or_else(|| {
                    warn!(
                        preferred = %self.cfg.gpu.preferred_adapter,
                        "Preferred GPU adapter unavailable for surface; falling back to default"
                    );
                    pollster::block_on(temp_instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: Some(&surface),
                        force_fallback_adapter: false,
                    }))
                })
        } else {
            pollster::block_on(temp_instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }))
        }
        .ok_or_else(|| anyhow!("No compatible GPU adapter for window surface"))?;
        let adapter_info = adapter.get_info();
        info!(
            adapter = %adapter_info.name,
            backend = ?adapter_info.backend,
            "Overlay surface initialized on GPU"
        );
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fluidscale-window-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Bgra8Unorm || *f == wgpu::TextureFormat::Rgba8Unorm)
            .unwrap_or(caps.formats[0]);

        let present_mode = match self.cfg.capture.present_mode.to_lowercase().as_str() {
            "immediate" if caps.present_modes.contains(&wgpu::PresentMode::Immediate) => {
                wgpu::PresentMode::Immediate
            }
            "fifo" => wgpu::PresentMode::Fifo,
            _ if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) => {
                wgpu::PresentMode::Mailbox
            }
            _ => wgpu::PresentMode::Fifo,
        };

        let size = window.inner_size();
        let surface_cfg = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(16),
            height: size.height.max(16),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &surface_cfg);

        let (pipe_dst_w, pipe_dst_h) = if self.cfg.preserve_aspect_ratio() {
            fit_dimensions_preserve_aspect(src_w, src_h, surface_cfg.width, surface_cfg.height)
        } else {
            (surface_cfg.width, surface_cfg.height)
        };

        let mut pipeline = FluidPipeline::new(
            &device,
            &self.cfg,
            src_w,
            src_h,
            pipe_dst_w,
            pipe_dst_h,
            self.flow_ctrl.current(),
            Some(format),
        )?;
        pipeline.set_present_aspect(
            &queue,
            surface_cfg.width,
            surface_cfg.height,
            self.cfg.preserve_aspect_ratio(),
        );
        self.surface_format = Some(format);

        if self.is_demo {
            self.synthetic_buf = vec![0u8; (src_w * src_h * 4) as usize];
            render_synthetic_frame(&mut self.synthetic_buf, src_w, src_h, 0, false);
            pipeline.upload_source_frame(&queue, &self.synthetic_buf);
            pipeline.upload_source_frame(&queue, &self.synthetic_buf);
            self.has_first_frame = true;
        }

        self.gpu = Some(GpuContext {
            instance: temp_instance,
            adapter,
            device,
            queue,
            adapter_info,
        });
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);
        self.surface_cfg = Some(surface_cfg);
        self.window = Some(window);
        Ok(())
    }

    fn tick_frame(&mut self) {
        let (Some(gpu), Some(pipe), Some(surface)) =
            (self.gpu.as_ref(), self.pipeline.as_mut(), self.surface.as_ref())
        else {
            return;
        };

        let now = Instant::now();
        // Produce new real input frame (30 FPS synthetic in demo mode, or real DXGI/GDI capture)
        if self.is_demo {
            let demo_input_period = Duration::from_secs_f64(1.0 / 30.0);
            if now.duration_since(self.last_input_tick) >= demo_input_period {
                self.last_input_tick = now;
                self.src_frame_idx += 1;
                let cut = std::mem::take(&mut self.trigger_scene_cut);
                render_synthetic_frame(
                    &mut self.synthetic_buf,
                    pipe.src_width,
                    pipe.src_height,
                    self.src_frame_idx,
                    cut,
                );
                pipe.upload_source_frame(&gpu.queue, &self.synthetic_buf);
                let mut local_cfg = self.cfg.clone();
                local_cfg.ui_protection.top_layer_composite = self.ui_protect_enabled;
                let profile = self.cfg.match_profile("Default");
                let telem = pipe.analyze_motion_and_ui(
                    &gpu.device,
                    &gpu.queue,
                    &local_cfg,
                    profile,
                    self.src_frame_idx % 8 == 0,
                );
                if self.src_frame_idx % 8 == 0 {
                    self.active_tile_pct = ((telem.active_tiles as f32
                        / telem.total_tiles.max(1) as f32)
                        * 100.0)
                        .round() as u32;
                }
                self.pacer
                    .on_captured_frame(telem.mean_abs_diff.max(0.01), Some(1.0 / 30.0));
            }
        } else if let Some(cap) = self.capturer.as_mut() {
            if let Ok(Some(frame)) = cap.capture_next(0) {
                if frame.width == pipe.src_width && frame.height == pipe.src_height {
                    // Fast 512-tap pre-upload dedup so duplicate GDI/locked-FPS frames never clobber (prev, curr)
                    let (sample_diff, sample_luma) =
                        sampled_luma_diff(&frame.rgba, &self.last_sample_luma);
                    let is_dup = self.has_first_frame
                        && self.cfg.pacing.enable_dedup
                        && sample_diff < self.cfg.pacing.dedup_threshold;

                    if !is_dup {
                        self.last_sample_luma = sample_luma;
                        let etw_dt = self
                            .etw_tracker
                            .as_ref()
                            .and_then(|t| t.record_qpc_sample(frame.qpc));
                        if !self.has_first_frame {
                            // Seed both ping-pong buffers on first frame so frame 0 isn't black
                            pipe.upload_source_frame(&gpu.queue, &frame.rgba);
                            pipe.upload_source_frame(&gpu.queue, &frame.rgba);
                            self.has_first_frame = true;
                        } else {
                            pipe.upload_source_frame(&gpu.queue, &frame.rgba);
                        }
                        let profile = self.cfg.match_profile(&self.target_title);
                        let telem = pipe.analyze_motion_and_ui(
                            &gpu.device,
                            &gpu.queue,
                            &self.cfg,
                            profile,
                            true,
                        );
                        if self.pacer.on_captured_frame(telem.mean_abs_diff.max(sample_diff), etw_dt) {
                            self.last_input_tick = now;
                            self.active_tile_pct = ((telem.active_tiles as f32
                                / telem.total_tiles.max(1) as f32)
                                * 100.0)
                                .round() as u32;
                        }
                    }
                }
            }
        }

        // Wait until PI-PLL paced output deadline
        self.timer.wait_until(self.next_present_deadline);

        let alpha = if self.fg_enabled {
            self.pacer
                .interpolation_alpha(self.last_input_tick.elapsed().as_secs_f64())
        } else {
            1.0
        };

        let frame_tex = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                if let Some(scfg) = &self.surface_cfg {
                    surface.configure(&gpu.device, scfg);
                }
                return;
            }
            Err(_) => return,
        };
        let view = frame_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut local_cfg = self.cfg.clone();
        local_cfg.ui_protection.top_layer_composite = self.ui_protect_enabled;

        let synth_t0 = Instant::now();
        pipe.synthesize_and_present(
            &gpu.device,
            &gpu.queue,
            &local_cfg,
            alpha,
            self.pacer.captured_fps.round() as u32,
            self.pacer.output_fps.round() as u32,
            self.active_tile_pct,
            (self.flow_ctrl.current() * 100.0) as u32,
            self.debug_view,
            Some(&view),
        );
        frame_tex.present();
        self.pacer.on_output_presented();

        let mode = if self.fg_enabled {
            self.cfg.frame_gen.mode.as_str()
        } else {
            "off"
        };
        let interval = self.pacer.next_output_interval(
            self.cfg.pacing.queue_target,
            mode,
            self.cfg.frame_gen.multiplier,
            self.cfg.frame_gen.target_fps,
        );
        self.next_present_deadline = Instant::now() + interval;

        // [Opt 1] Live Dynamic FlowScale feedback loop (rebuilds flow pyramid smoothly under load spikes)
        let step_ms = synth_t0.elapsed().as_secs_f32() * 1000.0;
        let budget_ms = (interval.as_secs_f32() * 1000.0).clamp(4.0, 33.0);
        if let Some(new_scale) = self.flow_ctrl.update(step_ms, budget_ms) {
            if self.last_flow_rebuild.elapsed() >= Duration::from_secs(2) {
                if let (Some(gpu), Some(old_pipe), Some(scfg)) =
                    (self.gpu.as_ref(), self.pipeline.as_ref(), self.surface_cfg.as_ref())
                {
                    if let Ok(new_pipe) = FluidPipeline::new(
                        &gpu.device,
                        &self.cfg,
                        old_pipe.src_width,
                        old_pipe.src_height,
                        old_pipe.dst_width,
                        old_pipe.dst_height,
                        new_scale,
                        self.surface_format,
                    ) {
                        new_pipe.set_present_aspect(
                            &gpu.queue,
                            scfg.width,
                            scfg.height,
                            self.cfg.preserve_aspect_ratio(),
                        );
                        info!(new_flow_scale = format!("{:.2}", new_scale), "DynamicFlowScale rebuilt pipeline");
                        self.pipeline = Some(new_pipe);
                        self.has_first_frame = false;
                        self.last_flow_rebuild = Instant::now();
                    }
                }
            }
        }
    }
}

/// Fast O(512) evenly-spaced luma fingerprint for pre-upload duplicate frame rejection.
fn sampled_luma_diff(rgba: &[u8], prev_samples: &[u8]) -> (f32, Vec<u8>) {
    let pixel_count = rgba.len() / 4;
    if pixel_count == 0 {
        return (1.0, Vec::new());
    }
    let samples = 512usize.min(pixel_count);
    let stride = (pixel_count / samples).max(1);
    let mut curr = Vec::with_capacity(samples);
    for i in 0..samples {
        let base = (i * stride) * 4;
        let r = rgba[base] as u32;
        let g = rgba[base + 1] as u32;
        let b = rgba[base + 2] as u32;
        let y = ((r * 54 + g * 183 + b * 19) >> 8) as u8;
        curr.push(y);
    }
    if prev_samples.len() != curr.len() {
        return (1.0, curr);
    }
    let sum: u32 = curr
        .iter()
        .zip(prev_samples.iter())
        .map(|(&a, &b)| a.abs_diff(b) as u32)
        .sum();
    let mean = (sum as f32) / (curr.len() as f32 * 255.0);
    (mean, curr)
}

impl ApplicationHandler for WindowedRunner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            if let Err(e) = self.init_graphics(event_loop) {
                eprintln!("Initialization failed: {e:?}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if new_size.width >= 16 && new_size.height >= 16 {
                    if let (Some(gpu), Some(surface), Some(scfg)) =
                        (&self.gpu, &self.surface, &mut self.surface_cfg)
                    {
                        scfg.width = new_size.width;
                        scfg.height = new_size.height;
                        surface.configure(&gpu.device, scfg);
                        if let Some(pipe) = &self.pipeline {
                            pipe.set_present_aspect(
                                &gpu.queue,
                                scfg.width,
                                scfg.height,
                                self.cfg.preserve_aspect_ratio(),
                            );
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Escape => event_loop.exit(),
                        KeyCode::KeyD => {
                            self.debug_view = !self.debug_view;
                            info!(debug_view = self.debug_view, "Toggled UI Mask & Active-Tile Debug View");
                        }
                        KeyCode::KeyF => {
                            self.fg_enabled = !self.fg_enabled;
                            info!(frame_gen = self.fg_enabled, "Toggled Frame Generation");
                        }
                        KeyCode::KeyU => {
                            self.ui_protect_enabled = !self.ui_protect_enabled;
                            info!(ui_anti_jello = self.ui_protect_enabled, "Toggled UI Anti-Jello Protection");
                        }
                        KeyCode::Space => {
                            self.trigger_scene_cut = true;
                            info!("Triggered synthetic Scene Cut");
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.tick_frame();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if !self.is_demo {
            if let Some(hwnd) = self.target_hwnd {
                if !is_window_alive(hwnd) {
                    info!("Target capture window closed; exiting overlay cleanly");
                    event_loop.exit();
                    return;
                }
            }
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}
