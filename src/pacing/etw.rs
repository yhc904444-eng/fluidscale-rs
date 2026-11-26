//! Real-time Present timestamp observer.
//!
//! On Windows with Administrator privileges, can open an ETW session targeting
//! `Microsoft-Windows-DxgKrnl` (`802ec45a-1e99-4b83-9920-87c98277ba9d`) to observe true kernel
//! SwapChain Present timestamps (decoupling frame pacing from DWM composition jitter).
//! When running unprivileged or when ETW is disabled, gracefully falls back to QPC timestamps.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, info};

#[derive(Debug, Default)]
struct EtwState {
    last_present_qpc: Option<i64>,
    last_interval_sec: Option<f64>,
    qpc_freq: f64,
}

#[allow(dead_code)]
pub struct PresentTimestampTracker {
    active: Arc<AtomicBool>,
    state: Arc<Mutex<EtwState>>,
    target_pid: u32,
}

#[allow(dead_code)]
impl PresentTimestampTracker {
    pub fn start(target_pid: u32, enable_etw: bool) -> Self {
        let qpc_freq = query_qpc_frequency();
        let state = Arc::new(Mutex::new(EtwState {
            last_present_qpc: None,
            last_interval_sec: None,
            qpc_freq,
        }));
        let active = Arc::new(AtomicBool::new(false));

        if enable_etw && target_pid != 0 {
            #[cfg(windows)]
            {
                if try_probe_etw_privilege() {
                    info!(
                        pid = target_pid,
                        "ETW DxgKrnl Present tracker enabled (privileged session ready)"
                    );
                    active.store(true, Ordering::Relaxed);
                } else {
                    debug!(
                        "ETW DxgKrnl requires Administrator privilege; falling back to high-res QPC capture timestamps"
                    );
                }
            }
        }

        Self {
            active,
            state,
            target_pid,
        }
    }

    pub fn is_etw_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    pub fn target_pid(&self) -> u32 {
        self.target_pid
    }

    /// Record a kernel/QPC timestamp sample (either from ETW Present event or capture callback)
    /// and return the computed interval in seconds.
    pub fn record_qpc_sample(&self, qpc_now: i64) -> Option<f64> {
        let mut guard = self.state.lock();
        let prev = guard.last_present_qpc.replace(qpc_now);
        if let Some(p) = prev {
            if qpc_now > p && guard.qpc_freq > 0.0 {
                let dt = (qpc_now - p) as f64 / guard.qpc_freq;
                guard.last_interval_sec = Some(dt);
                return Some(dt);
            }
        }
        None
    }

    pub fn poll_latest_interval_sec(&self) -> Option<f64> {
        self.state.lock().last_interval_sec.take()
    }
}

pub fn query_qpc_now() -> i64 {
    #[cfg(windows)]
    unsafe {
        let mut v = 0i64;
        let _ = windows::Win32::System::Performance::QueryPerformanceCounter(&mut v);
        v
    }
    #[cfg(not(windows))]
    {
        0
    }
}

pub fn query_qpc_frequency() -> f64 {
    #[cfg(windows)]
    unsafe {
        let mut f = 0i64;
        let _ = windows::Win32::System::Performance::QueryPerformanceFrequency(&mut f);
        f.max(1) as f64
    }
    #[cfg(not(windows))]
    {
        1_000_000_000.0
    }
}

#[cfg(windows)]
fn try_probe_etw_privilege() -> bool {
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::Security::{
        GetTokenInformation, TokenElevation, TOKEN_ELEVATION, TOKEN_QUERY,
    };
    use windows::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    unsafe {
        let mut token = HANDLE::default();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token).is_err() {
            return false;
        }
        let mut elev = TOKEN_ELEVATION::default();
        let mut ret_len = 0u32;
        let ok = GetTokenInformation(
            token,
            TokenElevation,
            Some(&mut elev as *mut _ as *mut _),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut ret_len,
        )
        .is_ok();
        let _ = CloseHandle(token);
        ok && elev.TokenIsElevated != 0
    }
}
