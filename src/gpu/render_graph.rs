//! [Opt 2] RenderGraph Lifetime Analyzer & Transient Texture Aliasing Pool.
//!
//! Instead of dedicating a permanent VRAM allocation to every intermediate pass texture
//! (as naive D3D11 pipelines do), this RenderGraph:
//! 1. Computes each logical transient resource's exact `[first_pass, last_pass]` lifetime.
//! 2. Greedily packs non-overlapping lifetimes with compatible `(width, height, format, usage)`
//!    into shared physical GPU textures (memory aliasing).
//! 3. Reports exact naive vs. aliased VRAM footprint and percentage saved.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSpec {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
}

impl TextureSpec {
    pub fn bytes_per_pixel(&self) -> u64 {
        match self.format {
            wgpu::TextureFormat::R8Unorm => 1,
            wgpu::TextureFormat::Rg8Unorm => 2,
            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Rgb10a2Unorm
            | wgpu::TextureFormat::R32Float
            | wgpu::TextureFormat::Rg16Float => 4,
            wgpu::TextureFormat::Rgba16Float => 8,
            wgpu::TextureFormat::Rgba32Float => 16,
            _ => 4,
        }
    }

    pub fn estimated_bytes(&self) -> u64 {
        self.width as u64 * self.height as u64 * self.bytes_per_pixel()
    }
}

#[derive(Debug, Clone)]
pub struct LogicalResource {
    pub name: String,
    pub spec: TextureSpec,
    pub first_pass: usize,
    pub last_pass: usize,
    pub persistent: bool,
}

#[derive(Debug, Clone)]
pub struct AliasingPlan {
    pub logical_to_physical: HashMap<String, usize>,
    pub physical_specs: Vec<TextureSpec>,
    pub naive_vram_bytes: u64,
    pub aliased_vram_bytes: u64,
}

impl AliasingPlan {
    pub fn saved_bytes(&self) -> u64 {
        self.naive_vram_bytes.saturating_sub(self.aliased_vram_bytes)
    }

    pub fn saved_percent(&self) -> f64 {
        if self.naive_vram_bytes == 0 {
            0.0
        } else {
            (self.saved_bytes() as f64 / self.naive_vram_bytes as f64) * 100.0
        }
    }
}

pub struct RenderGraphBuilder {
    resources: Vec<LogicalResource>,
}

impl RenderGraphBuilder {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
        }
    }

    pub fn add_transient(
        &mut self,
        name: impl Into<String>,
        spec: TextureSpec,
        first_pass: usize,
        last_pass: usize,
    ) {
        self.resources.push(LogicalResource {
            name: name.into(),
            spec,
            first_pass,
            last_pass,
            persistent: false,
        });
    }

    pub fn add_persistent(&mut self, name: impl Into<String>, spec: TextureSpec) {
        self.resources.push(LogicalResource {
            name: name.into(),
            spec,
            first_pass: 0,
            last_pass: usize::MAX,
            persistent: true,
        });
    }

    /// Compile logical resource lifetimes into physical aliased slots.
    /// If `enable_aliasing == false`, allocates 1:1 physical textures (useful for A/B comparison).
    pub fn compile(&self, enable_aliasing: bool) -> AliasingPlan {
        let mut naive_vram_bytes = 0u64;
        let mut physical_specs: Vec<TextureSpec> = Vec::new();
        let mut physical_busy_until: Vec<usize> = Vec::new();
        let mut logical_to_physical = HashMap::new();

        // Sort by first_pass so greedy interval graph coloring is optimal for interval graphs
        let mut ordered: Vec<&LogicalResource> = self.resources.iter().collect();
        ordered.sort_by_key(|r| (r.persistent, r.first_pass, r.last_pass));

        for res in ordered {
            naive_vram_bytes += res.spec.estimated_bytes();

            let mut chosen_slot = None;
            if enable_aliasing && !res.persistent {
                for (slot_idx, slot_spec) in physical_specs.iter().enumerate() {
                    if *slot_spec == res.spec && physical_busy_until[slot_idx] < res.first_pass {
                        chosen_slot = Some(slot_idx);
                        break;
                    }
                }
            }

            let slot = if let Some(idx) = chosen_slot {
                physical_busy_until[idx] = res.last_pass;
                idx
            } else {
                let idx = physical_specs.len();
                physical_specs.push(res.spec);
                physical_busy_until.push(if res.persistent {
                    usize::MAX
                } else {
                    res.last_pass
                });
                idx
            };

            logical_to_physical.insert(res.name.clone(), slot);
        }

        let aliased_vram_bytes: u64 = physical_specs.iter().map(|s| s.estimated_bytes()).sum();

        AliasingPlan {
            logical_to_physical,
            physical_specs,
            naive_vram_bytes,
            aliased_vram_bytes,
        }
    }
}

#[allow(dead_code)]
pub struct PhysicalTextureSlot {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub spec: TextureSpec,
}

pub struct TransientResourcePool {
    pub plan: AliasingPlan,
    pub slots: Vec<PhysicalTextureSlot>,
}

impl TransientResourcePool {
    pub fn realize(device: &wgpu::Device, plan: AliasingPlan) -> Self {
        let slots = plan
            .physical_specs
            .iter()
            .enumerate()
            .map(|(idx, spec)| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("rg-physical-slot-{idx}")),
                    size: wgpu::Extent3d {
                        width: spec.width.max(1),
                        height: spec.height.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: spec.format,
                    usage: spec.usage,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                PhysicalTextureSlot {
                    texture,
                    view,
                    spec: *spec,
                }
            })
            .collect();

        Self { plan, slots }
    }

    pub fn view(&self, logical_name: &str) -> &wgpu::TextureView {
        let idx = self.plan.logical_to_physical[logical_name];
        &self.slots[idx].view
    }

    pub fn texture(&self, logical_name: &str) -> &wgpu::Texture {
        let idx = self.plan.logical_to_physical[logical_name];
        &self.slots[idx].texture
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_graph_aliases_non_overlapping_transients_and_saves_vram() {
        let mut rg = RenderGraphBuilder::new();
        let src_spec = TextureSpec {
            width: 1920,
            height: 1080,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        };
        let flow_spec = TextureSpec {
            width: 960,
            height: 540,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        };

        // Pass 1-2: pyramids live during [1, 3]
        rg.add_transient("prev_pyr", flow_spec, 1, 3);
        rg.add_transient("curr_pyr", flow_spec, 2, 3);
        // Pass 3: flow written at 3, read until 6
        rg.add_transient("flow_tex", flow_spec, 3, 6);
        // Pass 4: ui_raw written at 4, consumed at 5 -> dead after pass 5!
        rg.add_transient("ui_raw", src_spec, 4, 5);
        // Pass 6: synth_tex written at 6, consumed at 7 -> CAN ALIAS ui_raw!
        rg.add_transient("synth_tex", src_spec, 6, 7);
        // Pass 7: output_tex written at 7, presented at 8
        rg.add_transient("output_tex", src_spec, 7, 8);
        // Extra post-process scratch at pass 8 -> CAN ALIAS synth_tex (dead after 7)!
        rg.add_transient("post_scratch", src_spec, 8, 9);

        let plan = rg.compile(true);
        assert_eq!(
            plan.logical_to_physical["ui_raw"],
            plan.logical_to_physical["synth_tex"],
            "synth_tex should alias dead ui_raw slot"
        );
        assert_eq!(
            plan.logical_to_physical["synth_tex"],
            plan.logical_to_physical["post_scratch"],
            "post_scratch should reuse dead synth_tex slot"
        );
        assert!(plan.saved_bytes() > 0);
        assert!(plan.saved_percent() > 25.0);
    }
}
