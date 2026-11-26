use crate::config::GpuConfig;
use anyhow::{anyhow, Result};
use tracing::info;

#[derive(Debug, Clone)]
pub struct AdapterSummary {
    pub index: usize,
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub vendor: u32,
}

#[allow(dead_code)]
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    pub fn enumerate_adapters(backend_pref: &str) -> Vec<AdapterSummary> {
        let backends = parse_backends(backend_pref);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        instance
            .enumerate_adapters(backends)
            .into_iter()
            .enumerate()
            .map(|(index, a)| {
                let info = a.get_info();
                AdapterSummary {
                    index,
                    name: info.name,
                    backend: format!("{:?}", info.backend),
                    device_type: format!("{:?}", info.device_type),
                    vendor: info.vendor,
                }
            })
            .collect()
    }

    /// Initialize DX12 / Vulkan wgpu device. Supports explicit Dual-GPU adapter selection
    /// (e.g., offloading frame generation to a secondary iGPU/dGPU via `preferred_adapter`).
    pub fn new(cfg: &GpuConfig, compatible_surface: Option<&wgpu::Surface<'_>>) -> Result<Self> {
        let backends = parse_backends(&cfg.backend);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = if !cfg.preferred_adapter.trim().is_empty() {
            let needle = cfg.preferred_adapter.to_lowercase();
            let candidates = instance.enumerate_adapters(backends);
            candidates
                .into_iter()
                .find(|a| {
                    let ok_name = a.get_info().name.to_lowercase().contains(&needle);
                    let ok_surf = compatible_surface
                        .map(|s| a.is_surface_supported(s))
                        .unwrap_or(true);
                    ok_name && ok_surf
                })
                .ok_or_else(|| {
                    anyhow!(
                        "No GPU adapter matching '{}' on backend '{}'",
                        cfg.preferred_adapter,
                        cfg.backend
                    )
                })?
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface,
                force_fallback_adapter: false,
            }))
            .ok_or_else(|| anyhow!("Failed to acquire any compatible DX12/Vulkan GPU adapter"))?
        };

        let adapter_info = adapter.get_info();
        info!(
            name = %adapter_info.name,
            backend = ?adapter_info.backend,
            device_type = ?adapter_info.device_type,
            "Initialized GPU backend"
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fluidscale-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| anyhow!("request_device failed: {e}"))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_info,
        })
    }
}

pub fn parse_backends(name: &str) -> wgpu::Backends {
    match name.to_lowercase().as_str() {
        "dx12" => wgpu::Backends::DX12,
        "vulkan" | "vk" => wgpu::Backends::VULKAN,
        _ => wgpu::Backends::DX12 | wgpu::Backends::VULKAN,
    }
}

/// [Opt 1] Dynamic FlowScale controller:
/// Adjusts optical-flow working resolution smoothly based on measured GPU pass latency
/// against the target frame budget so high-FPS output never stutters under load spikes.
pub struct DynamicFlowScale {
    enabled: bool,
    current: f32,
    min_scale: f32,
    max_scale: f32,
    smoothed_gpu_ms: f32,
}

impl DynamicFlowScale {
    pub fn new(cfg: &GpuConfig) -> Self {
        let base = cfg
            .base_flow_scale
            .clamp(cfg.min_flow_scale, cfg.max_flow_scale);
        Self {
            enabled: cfg.dynamic_flow_scale,
            current: base,
            min_scale: cfg.min_flow_scale.clamp(0.20, 1.0),
            max_scale: cfg.max_flow_scale.clamp(0.25, 1.0),
            smoothed_gpu_ms: 2.0,
        }
    }

    pub fn current(&self) -> f32 {
        self.current
    }

    /// Update scale using measured compute time (ms) vs frame budget (ms).
    /// Returns `Some(new_scale)` when the scale steps by >= 0.05 (to avoid thrashing textures).
    pub fn update(&mut self, gpu_time_ms: f32, budget_ms: f32) -> Option<f32> {
        if !self.enabled || budget_ms <= 0.1 {
            return None;
        }
        self.smoothed_gpu_ms = self.smoothed_gpu_ms * 0.85 + gpu_time_ms * 0.15;
        let load = self.smoothed_gpu_ms / budget_ms;
        let prev = self.current;

        if load > 0.72 {
            self.current = (self.current - 0.05).max(self.min_scale);
        } else if load < 0.42 {
            self.current = (self.current + 0.025).min(self.max_scale);
        }

        if (self.current - prev).abs() >= 0.049 {
            Some(self.current)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;

    #[test]
    fn dynamic_flow_scale_reduces_under_overload_and_recovers() {
        let cfg = AppConfig::default().gpu;
        let mut dfs = DynamicFlowScale::new(&cfg);
        let initial = dfs.current();
        for _ in 0..20 {
            dfs.update(7.5, 8.33); // 90% budget pressure
        }
        assert!(dfs.current() < initial);
        let low = dfs.current();
        for _ in 0..40 {
            dfs.update(1.2, 8.33); // light load
        }
        assert!(dfs.current() > low);
    }
}
