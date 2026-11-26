//! Single-window GDI/PrintWindow capture fallback (captures a specific HWND cleanly).

use crate::capture::{CapturedFrame, FrameCapturer};
use crate::pacing::etw::query_qpc_now;
use crate::platform::win32::get_window_rect;
use anyhow::{anyhow, Result};

pub struct WindowGdiCapturer {
    hwnd: isize,
    width: u32,
    height: u32,
}

impl WindowGdiCapturer {
    pub fn new(hwnd: isize) -> Result<Self> {
        let (_, _, width, height) =
            get_window_rect(hwnd).ok_or_else(|| anyhow!("Target window rect unavailable"))?;
        Ok(Self {
            hwnd,
            width,
            height,
        })
    }
}

impl FrameCapturer for WindowGdiCapturer {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[cfg(windows)]
    fn capture_next(&mut self, _timeout_ms: u32) -> Result<Option<CapturedFrame>> {
        use windows::Win32::Foundation::HWND;
        use windows::Win32::Graphics::Gdi::{
            CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC, GetDIBits,
            ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS,
        };
        use windows::Win32::Storage::Xps::{PrintWindow, PRINT_WINDOW_FLAGS};

        unsafe {
            let hwnd = HWND(self.hwnd as *mut _);
            let w = self.width as i32;
            let h = self.height as i32;
            let hdc_win = GetDC(hwnd);
            let hdc_mem = CreateCompatibleDC(hdc_win);
            let hbm = CreateCompatibleBitmap(hdc_win, w, h);
            let old = SelectObject(hdc_mem, hbm);

            // PW_RENDERFULLCONTENT = 2 (captures DirectX/DWM accelerated window content)
            let ok = PrintWindow(hwnd, hdc_mem, PRINT_WINDOW_FLAGS(2)).as_bool();
            let qpc = query_qpc_now();

            let mut bgra = vec![0u8; (self.width * self.height * 4) as usize];
            if ok {
                let mut bi = BITMAPINFO {
                    bmiHeader: BITMAPINFOHEADER {
                        biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                        biWidth: w,
                        biHeight: -h, // top-down
                        biPlanes: 1,
                        biBitCount: 32,
                        biCompression: BI_RGB.0,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                GetDIBits(
                    hdc_mem,
                    hbm,
                    0,
                    self.height,
                    Some(bgra.as_mut_ptr() as *mut _),
                    &mut bi,
                    DIB_RGB_COLORS,
                );
            }

            SelectObject(hdc_mem, old);
            let _ = DeleteObject(hbm);
            let _ = DeleteDC(hdc_mem);
            ReleaseDC(hwnd, hdc_win);

            if !ok {
                return Ok(None);
            }

            for px in bgra.chunks_exact_mut(4) {
                px.swap(0, 2);
                px[3] = 255;
            }

            Ok(Some(CapturedFrame {
                width: self.width,
                height: self.height,
                rgba: bgra,
                qpc,
            }))
        }
    }

    #[cfg(not(windows))]
    fn capture_next(&mut self, _timeout_ms: u32) -> Result<Option<CapturedFrame>> {
        Ok(None)
    }
}
