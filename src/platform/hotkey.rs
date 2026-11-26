//! Zero-CPU background global hotkey listener using Win32 `RegisterHotKey` + `GetMessageW`.
//! Supports live hot-reload when the user changes hotkeys in the GUI and reports registration status.

use crate::config::CaptureConfig;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{info, warn};

pub struct GlobalHotkeyListener {
    triggered: Arc<AtomicBool>,
    registered: Arc<AtomicBool>,
    #[cfg(windows)]
    thread_id: u32,
}

impl GlobalHotkeyListener {
    pub fn start(cfg: &CaptureConfig, repaint_cb: impl Fn() + Send + 'static) -> Self {
        let triggered = Arc::new(AtomicBool::new(false));
        let registered = Arc::new(AtomicBool::new(false));

        #[cfg(windows)]
        let thread_id = {
            let trig = triggered.clone();
            let reg_flag = registered.clone();
            let use_ctrl = cfg.hotkey_ctrl;
            let use_alt = cfg.hotkey_alt;
            let key_str = cfg.hotkey_key.trim().to_ascii_uppercase();
            let vk = vk_from_key_name(&key_str);
            let (tx, rx) = std::sync::mpsc::channel::<(u32, bool)>();

            std::thread::spawn(move || unsafe {
                use windows::Win32::Foundation::HWND;
                use windows::Win32::System::Threading::GetCurrentThreadId;
                use windows::Win32::UI::Input::KeyboardAndMouse::{
                    RegisterHotKey, UnregisterHotKey, HOT_KEY_MODIFIERS, MOD_ALT, MOD_CONTROL,
                    MOD_NOREPEAT,
                };
                use windows::Win32::UI::WindowsAndMessaging::{GetMessageW, MSG, WM_HOTKEY};

                let tid = GetCurrentThreadId();
                let mut mods = MOD_NOREPEAT;
                if use_ctrl {
                    mods |= MOD_CONTROL;
                }
                if use_alt {
                    mods |= MOD_ALT;
                }
                const HOTKEY_ID: i32 = 0x4653; // "FS"

                if RegisterHotKey(HWND::default(), HOTKEY_ID, HOT_KEY_MODIFIERS(mods.0), vk).is_err()
                {
                    warn!(
                        key = %key_str,
                        "RegisterHotKey failed (hotkey may already be in use by another app)"
                    );
                    reg_flag.store(false, Ordering::SeqCst);
                    let _ = tx.send((0, false));
                    return;
                }
                reg_flag.store(true, Ordering::SeqCst);
                let _ = tx.send((tid, true));
                info!(
                    ctrl = use_ctrl,
                    alt = use_alt,
                    key = %key_str,
                    "Registered global scaling hotkey"
                );

                let mut msg = MSG::default();
                while GetMessageW(&mut msg, HWND::default(), 0, 0).as_bool() {
                    if msg.message == WM_HOTKEY && msg.wParam.0 as i32 == HOTKEY_ID {
                        trig.store(true, Ordering::SeqCst);
                        repaint_cb();
                    }
                }
                let _ = UnregisterHotKey(HWND::default(), HOTKEY_ID);
            });

            let (tid, _ok) = rx.recv().unwrap_or((0, false));
            tid
        };

        #[cfg(not(windows))]
        {
            let _ = (cfg, repaint_cb);
        }

        Self {
            triggered,
            registered,
            #[cfg(windows)]
            thread_id,
        }
    }

    /// Cleanly unregister the previous global hotkey and re-register with updated `CaptureConfig`.
    pub fn restart(&mut self, cfg: &CaptureConfig, repaint_cb: impl Fn() + Send + 'static) {
        self.stop_listener();
        *self = Self::start(cfg, repaint_cb);
    }

    pub fn is_registered(&self) -> bool {
        self.registered.load(Ordering::SeqCst)
    }

    pub fn take_triggered(&self) -> bool {
        self.triggered.swap(false, Ordering::SeqCst)
    }

    fn stop_listener(&mut self) {
        #[cfg(windows)]
        if self.thread_id != 0 {
            unsafe {
                use windows::Win32::Foundation::{LPARAM, WPARAM};
                use windows::Win32::UI::WindowsAndMessaging::{PostThreadMessageW, WM_QUIT};
                let _ = PostThreadMessageW(self.thread_id, WM_QUIT, WPARAM(0), LPARAM(0));
            }
            self.thread_id = 0;
            // Brief pause so Windows releases the previous hotkey binding before re-registering
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    }
}

/// Convert readable key name (`"S"`, `"F8"`, `"F10"`, `"0".."9"`) to Windows VK code.
pub fn vk_from_key_name(key: &str) -> u32 {
    let k = key.trim().to_ascii_uppercase();
    match k.as_str() {
        "F1" => 0x70,
        "F2" => 0x71,
        "F3" => 0x72,
        "F4" => 0x73,
        "F5" => 0x74,
        "F6" => 0x75,
        "F7" => 0x76,
        "F8" => 0x77,
        "F9" => 0x78,
        "F10" => 0x79,
        "F11" => 0x7A,
        "F12" => 0x7B,
        _ => k
            .chars()
            .next()
            .filter(|c| c.is_ascii_alphanumeric())
            .unwrap_or('S') as u32,
    }
}

#[cfg(windows)]
impl Drop for GlobalHotkeyListener {
    fn drop(&mut self) {
        self.stop_listener();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vk_key_mapping_supports_letters_and_function_keys() {
        assert_eq!(vk_from_key_name("s"), b'S' as u32);
        assert_eq!(vk_from_key_name("F8"), 0x77);
        assert_eq!(vk_from_key_name("f10"), 0x79);
    }
}
