use std::time::{Duration, Instant};

/// Hybrid high-precision frame pacer:
/// - Coarse sleep via Windows `CreateWaitableTimerExW(CREATE_WAITABLE_TIMER_HIGH_RESOLUTION)`
///   (zero busy-CPU burn during the bulk of the frame interval)
/// - Fine tail spin-wait for the last `spin_wait_us` microseconds using `Instant::now()` / QPC,
///   eliminating Windows scheduler ~0.5-1ms wake-up jitter.
pub struct HybridPacerTimer {
    spin_tail: Duration,
    #[cfg(windows)]
    timer_handle: Option<windows::Win32::Foundation::HANDLE>,
}

impl HybridPacerTimer {
    pub fn new(spin_wait_us: u64) -> Self {
        #[cfg(windows)]
        let timer_handle = unsafe {
            use windows::core::PCWSTR;
            use windows::Win32::System::Threading::{
                CreateWaitableTimerExW, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS,
            };
            CreateWaitableTimerExW(
                None,
                PCWSTR::null(),
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS.0,
            )
            .ok()
        };

        Self {
            spin_tail: Duration::from_micros(spin_wait_us),
            #[cfg(windows)]
            timer_handle,
        }
    }

    pub fn wait_until(&self, deadline: Instant) {
        let now = Instant::now();
        if deadline <= now {
            return;
        }
        let remaining = deadline - now;
        if remaining > self.spin_tail {
            let sleep_dur = remaining - self.spin_tail;
            self.sleep_high_res(sleep_dur);
        }
        // Final sub-millisecond spin tail for deterministic frame pacing
        while Instant::now() < deadline {
            std::hint::spin_loop();
        }
    }

    #[cfg(windows)]
    fn sleep_high_res(&self, dur: Duration) {
        use windows::Win32::System::Threading::{SetWaitableTimerEx, WaitForSingleObject, INFINITE};
        if let Some(handle) = self.timer_handle {
            // Negative relative time in 100-nanosecond intervals
            let due_time: i64 = -((dur.as_nanos() / 100).max(1) as i64);
            unsafe {
                if SetWaitableTimerEx(handle, &due_time, 0, None, None, None, 0).is_ok() {
                    let _ = WaitForSingleObject(handle, INFINITE);
                    return;
                }
            }
        }
        std::thread::sleep(dur);
    }

    #[cfg(not(windows))]
    fn sleep_high_res(&self, dur: Duration) {
        std::thread::sleep(dur);
    }
}

#[cfg(windows)]
impl Drop for HybridPacerTimer {
    fn drop(&mut self) {
        if let Some(handle) = self.timer_handle.take() {
            unsafe {
                let _ = windows::Win32::Foundation::CloseHandle(handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_timer_hits_deadline_accurately() {
        let timer = HybridPacerTimer::new(500);
        let start = Instant::now();
        let target = start + Duration::from_millis(8);
        timer.wait_until(target);
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(8));
        assert!(
            elapsed < Duration::from_millis(14),
            "overshot by too much: {:?}",
            elapsed
        );
    }
}
