//! Real Windows DXGI Desktop Duplication capturer (`IDXGIOutputDuplication`).
//!
//! Optimus / Multi-Monitor resilient:
//! - Enumerates ALL DXGI adapters and outputs via `CreateDXGIFactory1` to locate the exact
//!   adapter + output containing the target window coordinates (fixes hybrid RTX+iGPU laptops!).
//! - Translates global desktop coordinates into monitor-local crop coordinates cleanly.
//! - Automatically re-initializes duplication on `DXGI_ERROR_ACCESS_LOST` (Alt+Tab / UAC / mode switch).

use crate::capture::{CapturedFrame, FrameCapturer};
use crate::pacing::etw::query_qpc_now;
use anyhow::{anyhow, Result};
use tracing::{info, warn};

#[cfg(windows)]
pub struct DxgiDesktopCapturer {
    adapter_index: u32,
    output_index: u32,
    device: windows::Win32::Graphics::Direct3D11::ID3D11Device,
    context: windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
    duplication: windows::Win32::Graphics::Dxgi::IDXGIOutputDuplication,
    staging: Option<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D>,
    desktop_width: u32,
    desktop_height: u32,
    crop: Option<(u32, u32, u32, u32)>,
}

#[cfg(windows)]
unsafe impl Send for DxgiDesktopCapturer {}

#[cfg(windows)]
impl DxgiDesktopCapturer {
    /// Locate the DXGI adapter & output covering `global_crop` `(x, y, w, h)` (or primary monitor
    /// if `None`) and create an `IDXGIOutputDuplication` session on that exact adapter.
    pub fn new_for_rect(global_crop: Option<(i32, i32, u32, u32)>) -> Result<Self> {
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
        crate::platform::win32::enable_per_monitor_dpi_awareness();

        unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1()
                .map_err(|e| anyhow!("CreateDXGIFactory1 failed: {e}"))?;

            let target_pt = global_crop.map(|(x, y, w, h)| (x + w as i32 / 2, y + h as i32 / 2));
            let mut chosen: Option<(u32, u32, i32, i32, u32, u32)> = None;
            let mut fallback: Option<(u32, u32, i32, i32, u32, u32)> = None;

            let mut a_idx = 0u32;
            while let Ok(adapter) = factory.EnumAdapters1(a_idx) {
                let mut o_idx = 0u32;
                while let Ok(output) = adapter.EnumOutputs(o_idx) {
                    if let Ok(desc) = output.GetDesc() {
                        let r = desc.DesktopCoordinates;
                        let mw = (r.right - r.left).max(1) as u32;
                        let mh = (r.bottom - r.top).max(1) as u32;
                        if fallback.is_none() {
                            fallback = Some((a_idx, o_idx, r.left, r.top, mw, mh));
                        }
                        if let Some((px, py)) = target_pt {
                            if px >= r.left && px < r.right && py >= r.top && py < r.bottom {
                                chosen = Some((a_idx, o_idx, r.left, r.top, mw, mh));
                                break;
                            }
                        } else {
                            chosen = Some((a_idx, o_idx, r.left, r.top, mw, mh));
                            break;
                        }
                    }
                    o_idx += 1;
                }
                if chosen.is_some() {
                    break;
                }
                a_idx += 1;
            }

            let (adapter_index, output_index, mon_x, mon_y, mon_w, mon_h) = chosen
                .or(fallback)
                .ok_or_else(|| anyhow!("No attached DXGI display output found on any adapter"))?;

            // Convert global desktop coordinates into monitor-local clamped crop rect
            let local_crop = global_crop.map(|(gx, gy, gw, gh)| {
                let lx = (gx - mon_x).clamp(0, mon_w.saturating_sub(1) as i32) as u32;
                let ly = (gy - mon_y).clamp(0, mon_h.saturating_sub(1) as i32) as u32;
                let lw = gw.min(mon_w.saturating_sub(lx)).max(16);
                let lh = gh.min(mon_h.saturating_sub(ly)).max(16);
                (lx, ly, lw, lh)
            });

            let (device, context, duplication) = Self::init_duplication(adapter_index, output_index)?;
            info!(
                adapter = adapter_index,
                output = output_index,
                monitor = format!("{mon_w}x{mon_h}"),
                crop = ?local_crop,
                "Initialized Optimus-safe DXGI Desktop Duplication"
            );

            Ok(Self {
                adapter_index,
                output_index,
                device,
                context,
                duplication,
                staging: None,
                desktop_width: mon_w,
                desktop_height: mon_h,
                crop: local_crop,
            })
        }
    }

    unsafe fn init_duplication(
        adapter_index: u32,
        output_index: u32,
    ) -> Result<(
        windows::Win32::Graphics::Direct3D11::ID3D11Device,
        windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
        windows::Win32::Graphics::Dxgi::IDXGIOutputDuplication,
    )> {
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::{
            D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_0,
        };
        use windows::Win32::Graphics::Direct3D11::{
            D3D11CreateDevice, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION,
        };
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory1, IDXGIOutput1,
        };

        let factory: IDXGIFactory1 = CreateDXGIFactory1()?;
        let adapter1 = factory.EnumAdapters1(adapter_index)?;
        let adapter: IDXGIAdapter = adapter1.cast()?;
        let output = adapter1.EnumOutputs(output_index)?;
        let output1: IDXGIOutput1 = output.cast()?;

        let mut device = None;
        let mut context = None;
        D3D11CreateDevice(
            &adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            None,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&[D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
        .map_err(|e| anyhow!("D3D11CreateDevice on adapter {adapter_index} failed: {e}"))?;

        let device = device.ok_or_else(|| anyhow!("No D3D11 device"))?;
        let context = context.ok_or_else(|| anyhow!("No D3D11 context"))?;
        let duplication = output1
            .DuplicateOutput(&device)
            .map_err(|e| anyhow!("IDXGIOutput1::DuplicateOutput failed: {e}"))?;
        Ok((device, context, duplication))
    }

    fn try_recover(&mut self) -> bool {
        unsafe {
            self.staging = None;
            if let Ok((dev, ctx, dupl)) = Self::init_duplication(self.adapter_index, self.output_index)
            {
                self.device = dev;
                self.context = ctx;
                self.duplication = dupl;
                warn!("Recovered DXGI Desktop Duplication after DXGI_ERROR_ACCESS_LOST");
                return true;
            }
            false
        }
    }
}

#[cfg(windows)]
impl FrameCapturer for DxgiDesktopCapturer {
    fn dimensions(&self) -> (u32, u32) {
        if let Some((_, _, w, h)) = self.crop {
            (w, h)
        } else {
            (self.desktop_width, self.desktop_height)
        }
    }

    fn capture_next(&mut self, timeout_ms: u32) -> Result<Option<CapturedFrame>> {
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D11::{
            ID3D11Texture2D, D3D11_CPU_ACCESS_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ,
            D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
        };
        use windows::Win32::Graphics::Dxgi::{
            DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_WAIT_TIMEOUT, DXGI_OUTDUPL_FRAME_INFO,
        };

        unsafe {
            let mut frame_info = DXGI_OUTDUPL_FRAME_INFO::default();
            let mut resource = None;
            let hr = self
                .duplication
                .AcquireNextFrame(timeout_ms, &mut frame_info, &mut resource);
            if let Err(e) = hr {
                if e.code() == DXGI_ERROR_WAIT_TIMEOUT {
                    return Ok(None);
                }
                if e.code() == DXGI_ERROR_ACCESS_LOST {
                    let _ = self.try_recover();
                    return Ok(None);
                }
                return Err(anyhow!("AcquireNextFrame failed: {e}"));
            }

            let qpc = query_qpc_now();
            let res = resource.ok_or_else(|| anyhow!("Missing DXGI frame resource"))?;
            let tex: ID3D11Texture2D = res.cast()?;
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            tex.GetDesc(&mut desc);

            if self.staging.is_none() {
                let mut sdesc = desc;
                sdesc.Usage = D3D11_USAGE_STAGING;
                sdesc.BindFlags = 0;
                sdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
                sdesc.MiscFlags = 0;
                let mut staging = None;
                self.device
                    .CreateTexture2D(&sdesc, None, Some(&mut staging))?;
                self.staging = staging;
            }

            let staging = self.staging.as_ref().unwrap();
            self.context.CopyResource(staging, &tex);

            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context
                .Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;

            let (cx, cy, cw, ch) = self.crop.unwrap_or((0, 0, desc.Width, desc.Height));
            let cx = cx.min(desc.Width.saturating_sub(1));
            let cy = cy.min(desc.Height.saturating_sub(1));
            let cw = cw.min(desc.Width - cx).max(1);
            let ch = ch.min(desc.Height - cy).max(1);

            let mut rgba = vec![0u8; (cw * ch * 4) as usize];
            let src_ptr = mapped.pData as *const u8;
            let row_pitch = mapped.RowPitch as usize;

            for y in 0..ch as usize {
                let src_row = std::slice::from_raw_parts(
                    src_ptr.add((cy as usize + y) * row_pitch + cx as usize * 4),
                    cw as usize * 4,
                );
                let dst_row = &mut rgba[y * cw as usize * 4..(y + 1) * cw as usize * 4];
                for (s, d) in src_row.chunks_exact(4).zip(dst_row.chunks_exact_mut(4)) {
                    d[0] = s[2];
                    d[1] = s[1];
                    d[2] = s[0];
                    d[3] = 255;
                }
            }

            self.context.Unmap(staging, 0);
            let _ = self.duplication.ReleaseFrame();

            Ok(Some(CapturedFrame {
                width: cw,
                height: ch,
                rgba,
                qpc,
            }))
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn dxgi_desktop_duplication_initializes_on_active_adapter() {
        match DxgiDesktopCapturer::new_for_rect(None) {
            Ok(mut cap) => {
                let (w, h) = cap.dimensions();
                println!("DXGI Desktop Duplication active monitor: {w}x{h}");
                assert!(w >= 640 && h >= 480);
                // Non-blocking poll (either timeout None or valid RGBA frame)
                let frame = cap.capture_next(50).expect("AcquireNextFrame should not error");
                if let Some(f) = frame {
                    assert_eq!(f.rgba.len(), (f.width * f.height * 4) as usize);
                    println!("DXGI captured real desktop frame: {}x{}", f.width, f.height);
                }
            }
            Err(e) => {
                // Report clearly if running in non-interactive/Session0 environment
                println!("DXGI duplication unavailable in this session (GDI fallback handles it): {e}");
            }
        }
    }
}
