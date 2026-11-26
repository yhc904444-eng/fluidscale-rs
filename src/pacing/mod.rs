pub mod dedup;
pub mod etw;
pub mod filter;
pub mod pll;
pub mod timer;

use crate::config::PacingConfig;
use dedup::ContentDedup;
use filter::RobustFrametimeEstimator;
use pll::PllQueueController;
use std::time::{Duration, Instant};

/// Unified frame-pacing engine combining:
/// 1. Outlier-resistant Median/Huber + 1D Kalman frametime estimator
/// 2. PI closed-loop PLL queue watermark controller
/// 3. Content-aware duplicate-frame detector
/// 4. Adaptive / Fixed fractional interpolation timestamp synthesis
pub struct FramePacer {
    estimator: RobustFrametimeEstimator,
    pll: PllQueueController,
    dedup: ContentDedup,
    last_real_present: Option<Instant>,
    pub captured_fps: f64,
    pub output_fps: f64,
    output_frame_counter: u32,
    output_fps_window_start: Instant,
    captured_frame_counter: u32,
    captured_fps_window_start: Instant,
}

impl FramePacer {
    pub fn new(cfg: &PacingConfig) -> Self {
        let now = Instant::now();
        Self {
            estimator: RobustFrametimeEstimator::new(cfg.filter_window.max(5)),
            pll: PllQueueController::new(
                cfg.queue_target,
                cfg.pll_kp,
                cfg.pll_ki,
                cfg.pll_max_slew,
            ),
            dedup: ContentDedup::new(cfg.enable_dedup, cfg.dedup_threshold),
            last_real_present: None,
            captured_fps: 0.0,
            output_fps: 0.0,
            output_frame_counter: 0,
            output_fps_window_start: now,
            captured_frame_counter: 0,
            captured_fps_window_start: now,
        }
    }

    /// Record an incoming captured frame with its measured content delta (e.g. mean abs luma diff).
    /// Returns `true` if this is a genuine new frame, `false` if deduplicated.
    pub fn on_captured_frame(&mut self, content_delta: f32, etw_present_sec: Option<f64>) -> bool {
        if !self.dedup.accept(content_delta) {
            return false;
        }

        let now = Instant::now();
        let dt_sec = if let Some(etw_dt) = etw_present_sec {
            etw_dt
        } else if let Some(prev) = self.last_real_present {
            now.duration_since(prev).as_secs_f64()
        } else {
            1.0 / 60.0
        };
        self.last_real_present = Some(now);

        if dt_sec > 0.0005 && dt_sec < 0.5 {
            self.estimator.push(dt_sec);
        }

        self.captured_frame_counter += 1;
        let elapsed = now.duration_since(self.captured_fps_window_start).as_secs_f64();
        if elapsed >= 0.5 {
            self.captured_fps = self.captured_frame_counter as f64 / elapsed;
            self.captured_frame_counter = 0;
            self.captured_fps_window_start = now;
        }
        true
    }

    /// Compute the next output frame interval (in seconds) regulated by the PI-PLL controller.
    pub fn next_output_interval(
        &mut self,
        queue_depth: f64,
        mode: &str,
        fixed_multiplier: u32,
        adaptive_target_fps: f32,
    ) -> Duration {
        let base_input_dt = self.estimator.estimate_sec().clamp(1.0 / 360.0, 1.0 / 10.0);
        let nominal_out_dt = match mode {
            "fixed" => base_input_dt / (fixed_multiplier.max(1) as f64),
            "adaptive" => 1.0 / (adaptive_target_fps.clamp(24.0, 480.0) as f64),
            _ => base_input_dt,
        };

        let slew_factor = self.pll.update(queue_depth, nominal_out_dt);
        let paced_dt = (nominal_out_dt * slew_factor).clamp(1.0 / 500.0, 0.1);
        Duration::from_secs_f64(paced_dt)
    }

    /// Compute interpolation timestamp `t` in `[0.0, 1.0]` for the current output slot
    /// between `frame[N-1]` and `frame[N]`.
    pub fn interpolation_alpha(&self, elapsed_since_pair_start: f64) -> f32 {
        let input_dt = self.estimator.estimate_sec().max(1e-4);
        (elapsed_since_pair_start / input_dt).clamp(0.0, 1.0) as f32
    }

    pub fn on_output_presented(&mut self) {
        self.output_frame_counter += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.output_fps_window_start).as_secs_f64();
        if elapsed >= 0.5 {
            self.output_fps = self.output_frame_counter as f64 / elapsed;
            self.output_frame_counter = 0;
            self.output_fps_window_start = now;
        }
    }

    #[allow(dead_code)]
    pub fn estimated_input_fps(&self) -> f64 {
        let dt = self.estimator.estimate_sec();
        if dt > 0.0 { 1.0 / dt } else { 0.0 }
    }

    pub fn last_slew_factor(&self) -> f64 {
        self.pll.last_slew()
    }
}
