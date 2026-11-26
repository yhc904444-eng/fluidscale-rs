//! Windows platform primitives:
//! - `WDA_EXCLUDEFROMCAPTURE` (prevents recursive self-capture of the overlay window)
//! - Click-through topmost borderless overlay styling with proper `SetLayeredWindowAttributes(255)`
//! - Target monitor bounds query (`MonitorFromWindow` + `GetMonitorInfoW`)
//! - Dynamic GPU scheduling priority elevation (`D3DKMTSetProcessSchedulingPriorityClass` + `HIGH_PRIORITY_CLASS`)
//! - Visible window & foreground window enumeration

use anyhow::{anyhow, Result};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct WindowInfo {
    pub hwnd: isize,
    pub pid: u32,
    pub title: String,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct MonitorRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// RAII guard that elevates process & GPU scheduling priority during active scaling and restores it on Drop.
pub struct GpuPriorityGuard {
    elevated: bool,
}

impl GpuPriorityGuard {
    pub fn elevate() -> Self {
        #[cfg(windows)]
        let elevated = unsafe {
            use windows::core::s;
            use windows::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
            use windows::Win32::System::Threading::{
                GetCurrentProcess, SetPriorityClass, HIGH_PRIORITY_CLASS,
            };

            let _ = SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

            let mut ok = false;
            if let Ok(gdi) = GetModuleHandleA(s!("gdi32.dll")) {
                if let Some(proc_addr) =
                    GetProcAddress(gdi, s!("D3DKMTSetProcessSchedulingPriorityClass"))
                {
                    type FnType = unsafe extern "system" fn(
                        windows::Win32::Foundation::HANDLE,
                        u32,
                    ) -> i32;
                    let func: FnType = std::mem::transmute(proc_addr);
                    let status = func(GetCurrentProcess(), 4);
                    if status == 0 {
                        info!("Elevated D3DKMT GPU scheduling priority class to HIGH");
                        ok = true;
                    } else {
                        warn!(
                            ntstatus = format!("0x{:08x}", status as u32),
                            "D3DKMTSetProcessSchedulingPriorityClass returned non-zero (run as Admin for realtime GPU priority)"
                        );
                    }
                }
            }
            ok
        };

        #[cfg(not(windows))]
        let elevated = false;

        Self { elevated }
    }
}

impl Drop for GpuPriorityGuard {
    fn drop(&mut self) {
        #[cfg(windows)]
        if self.elevated {
            unsafe {
                use windows::core::s;
                use windows::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
                use windows::Win32::System::Threading::{
                    GetCurrentProcess, SetPriorityClass, NORMAL_PRIORITY_CLASS,
                };
                let _ = SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
                if let Ok(gdi) = GetModuleHandleA(s!("gdi32.dll")) {
                    if let Some(proc_addr) =
                        GetProcAddress(gdi, s!("D3DKMTSetProcessSchedulingPriorityClass"))
                    {
                        type FnType = unsafe extern "system" fn(
                            windows::Win32::Foundation::HANDLE,
                            u32,
                        ) -> i32;
                        let func: FnType = std::mem::transmute(proc_addr);
                        let _ = func(GetCurrentProcess(), 2);
                    }
                }
            }
        }
    }
}

/// Enable Per-Monitor V2 DPI awareness so `GetWindowRect`, `GetMonitorInfoW`, and
/// `IDXGIOutput1::DuplicateOutput` operate in true physical pixels without Windows virtualization
/// or `E_ACCESSDENIED` failures on high-DPI laptops.
pub fn enable_per_monitor_dpi_awareness() {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::UI::HiDpi::{
            SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
        };
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
}

/// Apply `WDA_EXCLUDEFROMCAPTURE` so DXGI/WGC desktop capture transparently sees right through
/// our overlay window into the game window underneath (zero infinite feedback loop).
pub fn exclude_window_from_capture(hwnd_raw: isize) -> Result<()> {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Foundation::HWND;
        use windows::Win32::UI::WindowsAndMessaging::{
            SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE,
        };
        let hwnd = HWND(hwnd_raw as *mut _);
        SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            .map_err(|e| anyhow!("SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) failed: {e}"))?;
        info!(hwnd = hwnd_raw, "Applied WDA_EXCLUDEFROMCAPTURE to overlay window");
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let _ = hwnd_raw;
        Ok(())
    }
}

/// Configure an overlay window as click-through, topmost, and visibly opaque (`LWA_ALPHA = 255`).
/// CRITICAL: On Windows, setting `WS_EX_LAYERED` without calling `SetLayeredWindowAttributes`
/// leaves the window completely invisible (alpha = 0).
pub fn configure_clickthrough_overlay(hwnd_raw: isize) -> Result<()> {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Foundation::{COLORREF, HWND};
        use windows::Win32::UI::WindowsAndMessaging::{
            GetWindowLongPtrW, SetLayeredWindowAttributes, SetWindowLongPtrW, SetWindowPos,
            GWL_EXSTYLE, HWND_TOPMOST, LWA_ALPHA, SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_NOMOVE,
            SWP_NOSIZE, SWP_SHOWWINDOW, WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW,
            WS_EX_TOPMOST, WS_EX_TRANSPARENT,
        };
        let hwnd = HWND(hwnd_raw as *mut _);
        let ex_style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        let new_style = ex_style
            | (WS_EX_LAYERED.0 as isize)
            | (WS_EX_TRANSPARENT.0 as isize)
            | (WS_EX_TOPMOST.0 as isize)
            | (WS_EX_TOOLWINDOW.0 as isize)
            | (WS_EX_NOACTIVATE.0 as isize);
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, new_style);
        // Must initialize alpha to 255 so the D3D12/Vulkan SwapChain surface is fully visible!
        SetLayeredWindowAttributes(hwnd, COLORREF(0), 255, LWA_ALPHA)
            .map_err(|e| anyhow!("SetLayeredWindowAttributes(LWA_ALPHA=255) failed: {e}"))?;
        let _ = SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_SHOWWINDOW,
        );
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let _ = hwnd_raw;
        Ok(())
    }
}

/// Check if an HWND still refers to an existing, non-minimized window.
pub fn is_window_alive(hwnd_raw: isize) -> bool {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Foundation::HWND;
        use windows::Win32::UI::WindowsAndMessaging::IsWindow;
        IsWindow(HWND(hwnd_raw as *mut _)).as_bool()
    }
    #[cfg(not(windows))]
    {
        let _ = hwnd_raw;
        true
    }
}

/// Return the full physical monitor rectangle containing `hwnd_raw`.
pub fn get_monitor_rect_for_window(hwnd_raw: isize) -> Option<MonitorRect> {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Foundation::HWND;
        use windows::Win32::Graphics::Gdi::{
            GetMonitorInfoW, MonitorFromWindow, MONITORINFO, MONITOR_DEFAULTTONEAREST,
        };
        let hmon = MonitorFromWindow(HWND(hwnd_raw as *mut _), MONITOR_DEFAULTTONEAREST);
        if hmon.0.is_null() {
            return None;
        }
        let mut info = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if GetMonitorInfoW(hmon, &mut info).as_bool() {
            let r = info.rcMonitor;
            let w = (r.right - r.left).max(1) as u32;
            let h = (r.bottom - r.top).max(1) as u32;
            return Some(MonitorRect {
                x: r.left,
                y: r.top,
                width: w,
                height: h,
            });
        }
        None
    }
    #[cfg(not(windows))]
    {
        let _ = hwnd_raw;
        None
    }
}

/// Enumerate all visible top-level desktop windows with non-empty titles (excluding our own PID).
pub fn list_visible_windows() -> Vec<WindowInfo> {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Foundation::{BOOL, HWND, LPARAM, RECT};
        use windows::Win32::UI::WindowsAndMessaging::{
            EnumWindows, GetWindowRect, GetWindowTextLengthW, GetWindowTextW,
            GetWindowThreadProcessId, IsIconic, IsWindowVisible,
        };

        struct Ctx {
            self_pid: u32,
            list: Vec<WindowInfo>,
        }

        unsafe extern "system" fn enum_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
            if !IsWindowVisible(hwnd).as_bool() || IsIconic(hwnd).as_bool() {
                return BOOL(1);
            }
            let len = GetWindowTextLengthW(hwnd);
            if len <= 0 {
                return BOOL(1);
            }
            let mut buf = vec![0u16; (len + 1) as usize];
            let copied = GetWindowTextW(hwnd, &mut buf);
            if copied <= 0 {
                return BOOL(1);
            }
            let title = String::from_utf16_lossy(&buf[..copied as usize]);
            let mut rect = RECT::default();
            if GetWindowRect(hwnd, &mut rect).is_err() {
                return BOOL(1);
            }
            let w = rect.right - rect.left;
            let h = rect.bottom - rect.top;
            if w < 160 || h < 120 {
                return BOOL(1);
            }
            let mut pid = 0u32;
            GetWindowThreadProcessId(hwnd, Some(&mut pid));
            let ctx = &mut *(lparam.0 as *mut Ctx);
            if pid == ctx.self_pid {
                return BOOL(1);
            }
            ctx.list.push(WindowInfo {
                hwnd: hwnd.0 as isize,
                pid,
                title,
                x: rect.left,
                y: rect.top,
                width: w,
                height: h,
            });
            BOOL(1)
        }

        let mut ctx = Ctx {
            self_pid: std::process::id(),
            list: Vec::new(),
        };
        let _ = EnumWindows(Some(enum_proc), LPARAM(&mut ctx as *mut _ as isize));
        ctx.list
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

/// Get exact visible frame bounds (using `DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS)`
/// to strip Windows 10/11's invisible 7px drop-shadow border, falling back to `GetWindowRect`).
pub fn get_window_rect(hwnd_raw: isize) -> Option<(i32, i32, u32, u32)> {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Foundation::{HWND, RECT};
        use windows::Win32::Graphics::Dwm::{DwmGetWindowAttribute, DWMWA_EXTENDED_FRAME_BOUNDS};
        use windows::Win32::UI::WindowsAndMessaging::GetWindowRect;
        let hwnd = HWND(hwnd_raw as *mut _);
        let mut rect = RECT::default();
        let ok = DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            &mut rect as *mut _ as *mut _,
            std::mem::size_of::<RECT>() as u32,
        )
        .is_ok()
            || GetWindowRect(hwnd, &mut rect).is_ok();
        if ok {
            let w = (rect.right - rect.left).max(0) as u32;
            let h = (rect.bottom - rect.top).max(0) as u32;
            if w > 0 && h > 0 {
                return Some((rect.left, rect.top, w, h));
            }
        }
        None
    }
    #[cfg(not(windows))]
    {
        let _ = hwnd_raw;
        None
    }
}

/// Return information about the currently focused foreground window.
pub fn get_foreground_window_info() -> Option<WindowInfo> {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::UI::WindowsAndMessaging::{
            GetForegroundWindow, GetWindowRect, GetWindowTextLengthW, GetWindowTextW,
            GetWindowThreadProcessId,
        };
        let hwnd = GetForegroundWindow();
        if hwnd.0.is_null() {
            return None;
        }
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return None;
        }
        let mut buf = vec![0u16; (len + 1) as usize];
        let copied = GetWindowTextW(hwnd, &mut buf);
        if copied <= 0 {
            return None;
        }
        let title = String::from_utf16_lossy(&buf[..copied as usize]);
        let mut rect = RECT::default();
        if GetWindowRect(hwnd, &mut rect).is_err() {
            return None;
        }
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        Some(WindowInfo {
            hwnd: hwnd.0 as isize,
            pid,
            title,
            x: rect.left,
            y: rect.top,
            width: rect.right - rect.left,
            height: rect.bottom - rect.top,
        })
    }
    #[cfg(not(windows))]
    {
        None
    }
}
