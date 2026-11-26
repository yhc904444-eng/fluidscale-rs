//! Standalone Windows GUI Desktop Application (`FluidScale.exe`) — zero console window flash!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::Result;
use fluidscale_rs::config::AppConfig;
use fluidscale_rs::ui;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    fluidscale_rs::platform::win32::enable_per_monitor_dpi_awareness();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("fluidscale_rs=info,wgpu=warn,naga=warn")),
        )
        .init();

    let (read_path, write_path) = resolve_config_paths();
    let cfg = AppConfig::load_or_default(&read_path)?;
    ui::run_desktop_app(cfg, write_path)
}

/// Prefer portable `config.toml` beside `FluidScale.exe` when the directory is writable;
/// if extracted to a read-only location (e.g. `C:\Program Files\FluidScale`), gracefully
/// fall back to `%APPDATA%\FluidScale\config.toml` for saving while still seeding from any
/// bundled read-only default `config.toml`.
fn resolve_config_paths() -> (PathBuf, PathBuf) {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));

    if let Some(dir) = exe_dir {
        let portable = dir.join("config.toml");
        if is_config_writable(&dir, &portable) {
            return (portable.clone(), portable);
        }
        if let Ok(appdata) = std::env::var("APPDATA") {
            let user_cfg = PathBuf::from(appdata).join("FluidScale").join("config.toml");
            let read_src = if user_cfg.exists() || !portable.exists() {
                user_cfg.clone()
            } else {
                portable
            };
            return (read_src, user_cfg);
        }
        return (portable.clone(), portable);
    }

    let fallback = PathBuf::from("config.toml");
    (fallback.clone(), fallback)
}

fn is_config_writable(dir: &Path, cfg_file: &Path) -> bool {
    if cfg_file.exists() {
        OpenOptions::new().append(true).open(cfg_file).is_ok()
    } else {
        let probe = dir.join(format!(".fluidscale_probe_{}.tmp", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&probe) {
            Ok(_) => {
                let _ = fs::remove_file(&probe);
                true
            }
            Err(_) => false,
        }
    }
}
