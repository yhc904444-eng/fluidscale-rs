use std::collections::VecDeque;

/// Outlier-resistant Median/MAD + 1D Kalman frametime estimator.
///
/// Unlike a plain moving average (which gets dragged by a single shader-compilation stutter spike
/// and causes rubber-banding in Adaptive Frame Generation), this filter:
/// 1. Computes rolling median and MAD (Median Absolute Deviation) over a sliding window.
/// 2. Applies a Huber-style clamp to incoming observations beyond `2.5 * MAD` so single-frame
///    stutter spikes are suppressed without blocking genuine framerate shifts.
/// 3. Feeds the robust observation into a 1D Kalman filter for smooth, low-lag frametime tracking.
pub struct RobustFrametimeEstimator {
    window_size: usize,
    samples: VecDeque<f64>,
    /// Kalman state estimate (seconds per frame)
    x: f64,
    /// Kalman error covariance
    p: f64,
    /// Process noise variance (allows tracking real FPS ramps)
    q: f64,
    /// Measurement noise variance (suppresses DWM capture jitter)
    r: f64,
    initialized: bool,
}

impl RobustFrametimeEstimator {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(5),
            samples: VecDeque::with_capacity(window_size.max(5)),
            x: 1.0 / 60.0,
            p: 1e-4,
            q: 2.5e-7,
            r: 4.0e-6,
            initialized: false,
        }
    }

    pub fn push(&mut self, dt_sec: f64) -> f64 {
        if !dt_sec.is_finite() || dt_sec <= 0.0 {
            return self.x;
        }

        if self.samples.len() >= self.window_size {
            self.samples.pop_front();
        }
        self.samples.push_back(dt_sec);

        if !self.initialized {
            self.x = dt_sec;
            self.initialized = true;
            return self.x;
        }

        // Compute median and MAD for Huber outlier suppression
        let (median, mad) = self.median_and_mad();
        let scale = (mad * 1.4826).max(median * 0.04).max(0.0005);
        let huber_delta = 2.5 * scale;
        let residual = dt_sec - median;
        let clamped_obs = if residual.abs() > huber_delta {
            median + residual.signum() * huber_delta
        } else {
            dt_sec
        };

        // 1D Kalman predict + update
        let p_pred = self.p + self.q;
        let k = p_pred / (p_pred + self.r);
        self.x += k * (clamped_obs - self.x);
        self.p = (1.0 - k) * p_pred;

        self.x
    }

    pub fn estimate_sec(&self) -> f64 {
        self.x
    }

    fn median_and_mad(&self) -> (f64, f64) {
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mid = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 {
            0.5 * (sorted[mid - 1] + sorted[mid])
        } else {
            sorted[mid]
        };

        let mut devs: Vec<f64> = sorted.iter().map(|v| (v - median).abs()).collect();
        devs.sort_by(|a, b| a.total_cmp(b));
        let mad = if devs.len() % 2 == 0 {
            0.5 * (devs[mid - 1] + devs[mid])
        } else {
            devs[mid]
        };
        (median, mad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_single_shader_stutter_spike() {
        let mut est = RobustFrametimeEstimator::new(15);
        let nominal = 1.0 / 60.0; // 16.67ms
        for _ in 0..20 {
            est.push(nominal);
        }
        // Inject a huge 120ms shader compilation hitch
        let after_spike = est.push(0.120);
        // Should stay very close to 16.67ms (< 0.8ms drift) instead of jumping to ~23.5ms
        assert!(
            (after_spike - nominal).abs() < 0.0008,
            "expected spike rejection near 16.67ms, got {:.3}ms",
            after_spike * 1000.0
        );
    }

    #[test]
    fn tracks_genuine_framerate_step_change() {
        let mut est = RobustFrametimeEstimator::new(11);
        for _ in 0..20 {
            est.push(1.0 / 60.0);
        }
        // Game genuinely drops from 60fps to 45fps
        for _ in 0..30 {
            est.push(1.0 / 45.0);
        }
        let target = 1.0 / 45.0;
        assert!(
            (est.estimate_sec() - target).abs() < 0.001,
            "expected convergence to 22.22ms, got {:.3}ms",
            est.estimate_sec() * 1000.0
        );
    }
}
