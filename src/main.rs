use anyhow::Result;
use clap::{Parser, Subcommand};
use fluidscale_rs::app;
use fluidscale_rs::config::AppConfig;
use fluidscale_rs::gpu::context::GpuContext;
use fluidscale_rs::platform::win32::list_visible_windows;
use fluidscale_rs::ui;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "fluidscale-rs",
    version,
    about = "FluidScale-RS: Clean-room Rust screen-space upscaler & frame-generation desktop app (DX12/Vulkan, Tile-Skip IndirectDispatch, RenderGraph VRAM Aliasing, PI-PLL Pacing, Motion-Gated UI Anti-Jello)"
)]
struct Cli {
    /// Path to config.toml
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the interactive Desktop GUI Application (default when no subcommand is given)
    Gui,
    /// List available DX12/Vulkan GPU adapters (for Dual-GPU offload selection)
    ListGpus,
    /// List visible Windows desktop windows available for capture
    ListWindows,
    /// Run headless GPU benchmark verifying Tile-Skip, VRAM Aliasing, UI/Cut detection & throughput
    Benchmark {
        #[arg(long, default_value_t = 1920)]
        src_width: u32,
        #[arg(long, default_value_t = 1080)]
        src_height: u32,
        #[arg(long, default_value_t = 2560)]
        dst_width: u32,
        #[arg(long, default_value_t = 1440)]
        dst_height: u32,
        #[arg(long, default_value_t = 60)]
        frames: u32,
    },
    /// Launch standalone Demo window (30fps synthetic scene -> 120fps Adaptive FG)
    Demo,
    /// Attach click-through WDA_EXCLUDEFROMCAPTURE overlay to a live window
    Capture {
        /// Substring of target window title (defaults to first visible window)
        #[arg(short, long)]
        window: Option<String>,
    },
}

fn main() -> Result<()> {
    fluidscale_rs::platform::win32::enable_per_monitor_dpi_awareness();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("fluidscale_rs=info,wgpu=warn,naga=warn")),
        )
        .init();

    let cli = Cli::parse();
    let cfg = AppConfig::load_or_default(&cli.config)?;

    match cli.command.unwrap_or(Commands::Gui) {
        Commands::Gui => {
            ui::run_desktop_app(cfg, cli.config)?;
        }
        Commands::ListGpus => {
            let gpus = GpuContext::enumerate_adapters(&cfg.gpu.backend);
            println!("Detected DX12/Vulkan GPU Adapters ({}):", gpus.len());
            for g in gpus {
                println!(
                    "  [{}] {} | backend={} | type={} | vendor=0x{:04x}",
                    g.index, g.name, g.backend, g.device_type, g.vendor
                );
            }
        }
        Commands::ListWindows => {
            let wins = list_visible_windows();
            println!("Visible Desktop Windows ({}):", wins.len());
            for w in wins {
                println!(
                    "  PID={:<6} {}x{} at ({},{}) | {}",
                    w.pid, w.width, w.height, w.x, w.y, w.title
                );
            }
        }
        Commands::Benchmark {
            src_width,
            src_height,
            dst_width,
            dst_height,
            frames,
        } => {
            app::run_benchmark(&cfg, src_width, src_height, dst_width, dst_height, frames)?;
        }
        Commands::Demo => {
            app::run_windowed(cfg, None, true)?;
        }
        Commands::Capture { window } => {
            app::run_windowed(cfg, window, false)?;
        }
    }

    Ok(())
}
