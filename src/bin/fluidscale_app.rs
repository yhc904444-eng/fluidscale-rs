//! Standalone Windows GUI Desktop Application (`FluidScale.exe`) — zero console window flash!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::Result;
use fluidscale_rs::config::AppConfig;
use fluidscale_rs::ui;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    fluidscale_rs::platform::win32::enable_per_monitor_dpi_awareness();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("fluidscale_rs=info,wgpu=warn,naga=warn")),
        )
        .init();

    // Always place config.toml beside FluidScale.exe so shortcut/Start-Menu launches save safely
    let cfg_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("config.toml")))
        .unwrap_or_else(|| PathBuf::from("config.toml"));

    let cfg = AppConfig::load_or_default(&cfg_path)?;
    ui::run_desktop_app(cfg, cfg_path)
}
