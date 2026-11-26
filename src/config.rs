use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub gpu: GpuConfig,
    pub upscaling: UpscalingConfig,
    pub frame_gen: FrameGenConfig,
    pub ui_protection: UiProtectionConfig,
    pub pacing: PacingConfig,
    pub osd: OsdConfig,
    pub capture: CaptureConfig,
    #[serde(default)]
    pub profiles: Vec<GameProfile>,
}

fn default_aspect_mode() -> String {
    "preserve".into()
}

fn default_countdown_secs() -> u64 {
    3
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub backend: String,
    pub preferred_adapter: String,
    #[serde(default = "default_true")]
    pub compact_formats: bool,
    pub dynamic_flow_scale: bool,
    pub base_flow_scale: f32,
    pub min_flow_scale: f32,
    pub max_flow_scale: f32,
    pub tile_skip_threshold: f32,
    pub enable_vram_aliasing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpscalingConfig {
    pub algorithm: String,
    pub sharpness: f32,
    #[serde(default = "default_aspect_mode")]
    pub aspect_ratio_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameGenConfig {
    pub mode: String,
    pub multiplier: u32,
    pub target_fps: f32,
    pub scene_cut_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiProtectionConfig {
    pub motion_gate_threshold: f32,
    pub static_diff_threshold: f32,
    pub attack_rate: f32,
    pub decay_rate: f32,
    pub edge_prior_weight: f32,
    pub dilation_radius: u32,
    pub top_layer_composite: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacingConfig {
    pub queue_target: f64,
    pub filter_window: usize,
    pub pll_max_slew: f64,
    pub pll_kp: f64,
    pub pll_ki: f64,
    pub enable_dedup: bool,
    pub dedup_threshold: f32,
    pub spin_wait_us: u64,
    pub enable_etw: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsdConfig {
    pub show_fps: bool,
    pub scale: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    pub api: String,
    pub present_mode: String,
    pub hotkey_ctrl: bool,
    pub hotkey_alt: bool,
    pub hotkey_key: String,
    #[serde(default = "default_countdown_secs")]
    pub countdown_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameProfile {
    pub name: String,
    pub match_pattern: String,
    #[serde(default)]
    pub auto_scale: bool,
    /// Normalized [x, y, w, h] static HUD hint rects (up to 4 passed directly to GPU UI mask shader)
    #[serde(default)]
    pub hud_hints: Vec<[f32; 4]>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            gpu: GpuConfig {
                backend: "auto".into(),
                preferred_adapter: String::new(),
                compact_formats: true,
                dynamic_flow_scale: true,
                base_flow_scale: 0.5,
                min_flow_scale: 0.25,
                max_flow_scale: 0.85,
                tile_skip_threshold: 0.004,
                enable_vram_aliasing: true,
            },
            upscaling: UpscalingConfig {
                algorithm: "bicubic_cas".into(),
                sharpness: 0.45,
                aspect_ratio_mode: "preserve".into(),
            },
            frame_gen: FrameGenConfig {
                mode: "adaptive".into(),
                multiplier: 2,
                target_fps: 120.0,
                scene_cut_threshold: 0.65,
            },
            ui_protection: UiProtectionConfig {
                motion_gate_threshold: 0.75,
                static_diff_threshold: 0.015,
                attack_rate: 0.35,
                decay_rate: 0.06,
                edge_prior_weight: 0.35,
                dilation_radius: 2,
                top_layer_composite: true,
            },
            pacing: PacingConfig {
                queue_target: 1.5,
                filter_window: 15,
                pll_max_slew: 0.006,
                pll_kp: 0.004,
                pll_ki: 0.0005,
                enable_dedup: true,
                dedup_threshold: 0.0008,
                spin_wait_us: 600,
                enable_etw: true,
            },
            osd: OsdConfig {
                show_fps: true,
                scale: 2,
            },
            capture: CaptureConfig {
                api: "dxgi".into(),
                present_mode: "mailbox".into(),
                hotkey_ctrl: true,
                hotkey_alt: true,
                hotkey_key: "S".into(),
                countdown_secs: 3,
            },
            profiles: vec![GameProfile {
                name: "Default".into(),
                match_pattern: "*".into(),
                auto_scale: false,
                hud_hints: vec![
                    [0.02, 0.82, 0.24, 0.14],
                    [0.485, 0.485, 0.03, 0.03],
                ],
            }],
        }
    }
}

impl AppConfig {
    pub fn load_or_default(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read config file {}", path.display()))?;
        let cfg: Self = toml::from_str(&text)
            .with_context(|| format!("failed to parse TOML config {}", path.display()))?;
        Ok(cfg)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let toml_text = toml::to_string_pretty(self).context("failed to serialize AppConfig")?;
        fs::write(path, toml_text)
            .with_context(|| format!("failed to write config file {}", path.display()))?;
        Ok(())
    }

    pub fn match_profile(&self, window_title_or_exe: &str) -> Option<&GameProfile> {
        // Prefer specific non-wildcard profiles first, then fallback to wildcard "*"
        let mut fallback = None;
        for profile in &self.profiles {
            if profile.match_pattern == "*" {
                fallback = Some(profile);
                continue;
            }
            if wildcard_match(&profile.match_pattern, window_title_or_exe) {
                return Some(profile);
            }
        }
        fallback
    }

    pub fn upscaling_mode_id(&self) -> u32 {
        match self.upscaling.algorithm.to_lowercase().as_str() {
            "off" => 0,
            "bicubic_cas" => 1,
            "lanczos" => 2,
            "sharp_bilinear" => 3,
            "nearest" => 4,
            _ => 1,
        }
    }

    pub fn preserve_aspect_ratio(&self) -> bool {
        !self.upscaling.aspect_ratio_mode.eq_ignore_ascii_case("stretch")
    }

    /// Apply one of the curated quick presets while keeping user profiles & GPU selection intact.
    pub fn apply_preset(&mut self, preset: &str) {
        match preset {
            "esports" => {
                self.frame_gen.mode = "fixed".into();
                self.frame_gen.multiplier = 2;
                self.frame_gen.scene_cut_threshold = 0.50;
                self.upscaling.algorithm = "bicubic_cas".into();
                self.upscaling.sharpness = 0.35;
                self.ui_protection.top_layer_composite = true;
                self.ui_protection.dilation_radius = 2;
                self.gpu.base_flow_scale = 0.50;
                self.gpu.tile_skip_threshold = 0.004;
                self.gpu.dynamic_flow_scale = true;
                self.pacing.queue_target = 1.0;
                self.capture.present_mode = "immediate".into();
            }
            "cinematic" => {
                self.frame_gen.mode = "adaptive".into();
                self.frame_gen.target_fps = 120.0;
                self.frame_gen.scene_cut_threshold = 0.65;
                self.upscaling.algorithm = "lanczos".into();
                self.upscaling.sharpness = 0.55;
                self.ui_protection.top_layer_composite = true;
                self.ui_protection.dilation_radius = 2;
                self.gpu.base_flow_scale = 0.65;
                self.gpu.tile_skip_threshold = 0.003;
                self.gpu.dynamic_flow_scale = true;
                self.pacing.queue_target = 1.5;
                self.capture.present_mode = "mailbox".into();
            }
            "eco" => {
                self.frame_gen.mode = "fixed".into();
                self.frame_gen.multiplier = 2;
                self.frame_gen.scene_cut_threshold = 0.65;
                self.upscaling.algorithm = "bicubic_cas".into();
                self.upscaling.sharpness = 0.40;
                self.ui_protection.top_layer_composite = true;
                self.ui_protection.dilation_radius = 2;
                self.gpu.base_flow_scale = 0.40;
                self.gpu.tile_skip_threshold = 0.008;
                self.gpu.dynamic_flow_scale = true;
                self.gpu.enable_vram_aliasing = true;
                self.pacing.queue_target = 1.5;
                self.capture.present_mode = "fifo".into();
            }
            _ => {
                let saved_profiles = self.profiles.clone();
                let saved_adapter = self.gpu.preferred_adapter.clone();
                *self = Self::default();
                if !saved_profiles.is_empty() {
                    self.profiles = saved_profiles;
                }
                self.gpu.preferred_adapter = saved_adapter;
            }
        }
    }
}

/// Simple case-insensitive glob matching supporting `*` and `?`.
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.to_lowercase().chars().collect();
    let t: Vec<char> = text.to_lowercase().chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_idx, mut match_idx) = (None, 0usize);

    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star_idx = Some(pi);
            match_idx = ti;
            pi += 1;
        } else if let Some(si) = star_idx {
            pi = si + 1;
            match_idx += 1;
            ti = match_idx;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcard_matching_works() {
        assert!(wildcard_match("*elden*ring*", "ELDEN RING™"));
        assert!(wildcard_match("game_?.exe", "Game_1.exe"));
        assert!(!wildcard_match("cyberpunk*", "Starfield.exe"));
    }

    #[test]
    fn presets_and_reset_preserve_user_profiles() {
        let mut cfg = AppConfig::default();
        cfg.profiles.push(GameProfile {
            name: "Elden".into(),
            match_pattern: "*Elden*".into(),
            auto_scale: true,
            hud_hints: vec![[0.1, 0.8, 0.2, 0.1]],
        });
        cfg.apply_preset("esports");
        assert_eq!(cfg.frame_gen.mode, "fixed");
        assert_eq!(cfg.pacing.queue_target, 1.0);
        cfg.apply_preset("cinematic");
        assert_eq!(cfg.upscaling.algorithm, "lanczos");
        cfg.apply_preset("eco");
        assert!(cfg.gpu.base_flow_scale <= 0.41);
        cfg.apply_preset("reset");
        assert_eq!(cfg.profiles.len(), 2, "reset must preserve custom user profiles");
        assert!(cfg.preserve_aspect_ratio());
    }
}
