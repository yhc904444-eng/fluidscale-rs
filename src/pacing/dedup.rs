/// Content-aware duplicate frame detector.
///
/// Games locked to 30/60 FPS or rendering menus often present byte-identical or near-identical
/// consecutive frames. Passing duplicates blindly into optical flow produces zero motion followed
/// by a 2x motion jump (1-stutter-1-smooth judder).
pub struct ContentDedup {
    enabled: bool,
    threshold: f32,
    consecutive_dups: u32,
    max_consecutive_drops: u32,
    total_dropped: u64,
}

impl ContentDedup {
    pub fn new(enabled: bool, threshold: f32) -> Self {
        Self {
            enabled,
            threshold,
            consecutive_dups: 0,
            max_consecutive_drops: 6,
            total_dropped: 0,
        }
    }

    /// Returns `true` if the frame carries meaningful new visual motion and should advance the
    /// optical-flow pair; `false` if it should be treated as a duplicate.
    pub fn accept(&mut self, mean_abs_diff: f32) -> bool {
        if !self.enabled {
            return true;
        }
        if mean_abs_diff < self.threshold && self.consecutive_dups < self.max_consecutive_drops {
            self.consecutive_dups += 1;
            self.total_dropped += 1;
            false
        } else {
            self.consecutive_dups = 0;
            true
        }
    }

    #[allow(dead_code)]
    pub fn total_dropped(&self) -> u64 {
        self.total_dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_short_duplicate_runs_but_never_stalls_forever() {
        let mut dedup = ContentDedup::new(true, 0.001);
        assert!(dedup.accept(0.05)); // genuine frame
        assert!(!dedup.accept(0.0001)); // dup 1
        assert!(!dedup.accept(0.0002)); // dup 2
        assert!(dedup.accept(0.02)); // genuine frame resumes
        assert_eq!(dedup.total_dropped(), 2);

        // Even on a completely static screen, periodically admit a heartbeat frame
        let mut admitted = false;
        for _ in 0..10 {
            if dedup.accept(0.0) {
                admitted = true;
                break;
            }
        }
        assert!(admitted);
    }
}
