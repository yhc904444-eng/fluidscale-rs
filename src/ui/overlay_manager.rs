//! Managed Overlay Process Controller.
//!
//! Spawns the borderless click-through `WDA_EXCLUDEFROMCAPTURE` scaling overlay in an isolated
//! child process (giving the overlay its own Win32 main-thread winit EventLoop without colliding
//! with the control panel's eframe EventLoop), handles 3-second countdowns, and guarantees clean
//! shutdown on Stop or App exit.

use crate::config::AppConfig;
use crate::platform::win32::get_foreground_window_info;
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use tracing::info;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayStatus {
    Idle,
    Countdown {
        remaining_secs: u64,
        target_hint: Option<String>,
    },
    Running {
        target_title: String,
        pid: u32,
    },
}

pub struct OverlayManager {
    child: Option<Child>,
    running_title: String,
    countdown_deadline: Option<Instant>,
    countdown_target: Option<String>,
}

impl OverlayManager {
    pub fn new() -> Self {
        Self {
            child: None,
            running_title: String::new(),
            countdown_deadline: None,
            countdown_target: None,
        }
    }

    pub fn is_running(&self) -> bool {
        self.child.is_some()
    }

    pub fn is_counting_down(&self) -> bool {
        self.countdown_deadline.is_some()
    }

    pub fn status(&self) -> OverlayStatus {
        if let Some(ref child) = self.child {
            OverlayStatus::Running {
                target_title: self.running_title.clone(),
                pid: child.id(),
            }
        } else if let Some(deadline) = self.countdown_deadline {
            let rem = deadline
                .saturating_duration_since(Instant::now())
                .as_secs()
                .saturating_add(1);
            OverlayStatus::Countdown {
                remaining_secs: rem,
                target_hint: self.countdown_target.clone(),
            }
        } else {
            OverlayStatus::Idle
        }
    }

    /// Start a 3-second countdown (gives user time to click into their game window) before attaching.
    pub fn start_countdown(&mut self, secs: u64, explicit_target: Option<String>) {
        self.stop();
        self.countdown_deadline = Some(Instant::now() + Duration::from_secs(secs));
        self.countdown_target = explicit_target;
    }

    /// Immediately launch the capture overlay on the target window (or current foreground window).
    pub fn launch_now(
        &mut self,
        cfg: &AppConfig,
        cfg_path: &Path,
        explicit_target: Option<String>,
    ) -> Result<String> {
        self.stop();
        cfg.save(cfg_path)?;

        let target_title = if let Some(t) = explicit_target.filter(|s| !s.trim().is_empty()) {
            t
        } else if let Some(fg) = get_foreground_window_info() {
            if fg.pid == std::process::id() {
                return Err(anyhow!(
                    "当前前台窗口仍是 FluidScale 自身，请点击缩放后在 3 秒内切换到目标游戏窗口"
                ));
            }
            fg.title
        } else {
            return Err(anyhow!(
                "未检测到可捕获的前台窗口，请在“窗口捕获”页手动选择目标窗口"
            ));
        };

        let exe = resolve_worker_executable()?;
        let mut cmd = Command::new(&exe);
        cmd.arg("--config")
            .arg(cfg_path)
            .arg("capture")
            .arg("--window")
            .arg(&target_title);

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let child = cmd.spawn().map_err(|e| {
            anyhow!(
                "启动覆盖层进程失败 ({}): {e}",
                exe.display()
            )
        })?;
        info!(
            pid = child.id(),
            target = %target_title,
            "Launched managed capture overlay process"
        );
        self.child = Some(child);
        self.running_title = target_title.clone();
        Ok(target_title)
    }

    /// Poll countdown completion and detect unexpected child exit.
    /// Returns an optional status message if a transition occurred.
    pub fn poll(&mut self, cfg: &AppConfig, cfg_path: &Path) -> Option<Result<String>> {
        if let Some(ref mut child) = self.child {
            if let Ok(Some(exit)) = child.try_wait() {
                let title = std::mem::take(&mut self.running_title);
                self.child = None;
                if exit.success() {
                    return Some(Ok(format!("目标窗口已关闭，覆盖层已自动退出 ({title})")));
                } else {
                    return Some(Err(anyhow!(
                        "覆盖层异常退出 ({title}, code={exit})，可切换“单窗口兼容模式”重试"
                    )));
                }
            }
        }

        if let Some(deadline) = self.countdown_deadline {
            if Instant::now() >= deadline {
                self.countdown_deadline = None;
                let target = self.countdown_target.take();
                return Some(
                    self.launch_now(cfg, cfg_path, target)
                        .map(|t| format!("已附着覆盖层到窗口: {t}")),
                );
            }
        }
        None
    }

    pub fn stop(&mut self) {
        self.countdown_deadline = None;
        self.countdown_target = None;
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
            info!(target = %self.running_title, "Stopped capture overlay process");
        }
        self.running_title.clear();
    }
}

impl Drop for OverlayManager {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Prefer sibling `fluidscale-rs.exe` (CLI+overlay worker) if launched via `FluidScale.exe`,
/// otherwise reuse current executable.
fn resolve_worker_executable() -> Result<PathBuf> {
    let cur = std::env::current_exe()?;
    if let Some(dir) = cur.parent() {
        let cli_worker = dir.join("fluidscale-rs.exe");
        if cli_worker.exists() {
            return Ok(cli_worker);
        }
    }
    Ok(cur)
}
