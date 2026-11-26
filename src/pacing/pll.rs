/// PI Closed-Loop Phase-Locked Loop (PLL) Queue Watermark Controller.
///
/// Instead of abruptly dropping frames when the capture queue overflows or starving when it
/// empties, this controller continuously applies an imperceptible slew (±0.5% to ±0.8%) to the
/// output frame interval so queue depth stays locked around `target_depth`.
pub struct PllQueueController {
    target_depth: f64,
    kp: f64,
    ki: f64,
    max_slew: f64,
    integral: f64,
    last_slew: f64,
}

impl PllQueueController {
    pub fn new(target_depth: f64, kp: f64, ki: f64, max_slew: f64) -> Self {
        Self {
            target_depth: target_depth.max(0.5),
            kp,
            ki,
            max_slew: max_slew.clamp(0.001, 0.05),
            integral: 0.0,
            last_slew: 1.0,
        }
    }

    /// Update controller with current queue occupancy and nominal interval.
    ///
    /// Returns a multiplicative interval scale factor in `[1.0 - max_slew, 1.0 + max_slew]`:
    /// - Queue deeper than target (`error > 0`) -> factor `< 1.0` (present slightly faster to drain)
    /// - Queue shallower than target (`error < 0`) -> factor `> 1.0` (present slightly slower to refill)
    pub fn update(&mut self, current_depth: f64, dt_sec: f64) -> f64 {
        let error = current_depth - self.target_depth;
        let integral_limit = if self.ki > 1e-9 {
            self.max_slew / self.ki
        } else {
            0.0
        };
        self.integral = (self.integral + error * dt_sec).clamp(-integral_limit, integral_limit);

        let control = (self.kp * error + self.ki * self.integral).clamp(-self.max_slew, self.max_slew);
        self.last_slew = 1.0 - control;
        self.last_slew
    }

    pub fn last_slew(&self) -> f64 {
        self.last_slew
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pll_drains_high_queue_and_refills_low_queue_within_slew_bounds() {
        let mut pll = PllQueueController::new(1.5, 0.004, 0.001, 0.006);
        let fast = pll.update(3.5, 1.0 / 120.0);
        assert!(fast < 1.0 && fast >= 1.0 - 0.006);

        let slow = pll.update(0.2, 1.0 / 120.0);
        assert!(slow > 1.0 && slow <= 1.0 + 0.006);
    }

    #[test]
    fn pll_converges_simulated_queue_without_overflow() {
        let mut pll = PllQueueController::new(1.5, 0.005, 0.002, 0.008);
        let mut queue = 2.8_f64;
        // Simulate 300 output steps where producer arrives at 1.003x nominal rate
        for _ in 0..300 {
            let factor = pll.update(queue, 1.0 / 120.0);
            // Net queue change = arrival - consumption(1 / factor)
            queue = (queue + 1.003 - (1.0 / factor)).max(0.0);
        }
        assert!(
            (queue - 1.5).abs() < 0.35,
            "expected PLL to stabilize queue near 1.5, got {queue:.3}"
        );
    }
}
