pub mod dxgi;
pub mod window_gdi;

use anyhow::Result;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub qpc: i64,
}

pub trait FrameCapturer: Send {
    fn dimensions(&self) -> (u32, u32);
    fn capture_next(&mut self, timeout_ms: u32) -> Result<Option<CapturedFrame>>;
}

/// Bounded producer-consumer frame queue with watermark telemetry for the PI-PLL controller.
#[allow(dead_code)]
#[derive(Clone)]
pub struct BoundedFrameQueue {
    inner: Arc<Mutex<VecDeque<CapturedFrame>>>,
    capacity: usize,
}

#[allow(dead_code)]
impl BoundedFrameQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(capacity.max(2)))),
            capacity: capacity.max(2),
        }
    }

    pub fn push(&self, frame: CapturedFrame) -> bool {
        let mut q = self.inner.lock();
        let mut dropped = false;
        if q.len() >= self.capacity {
            q.pop_front();
            dropped = true;
        }
        q.push_back(frame);
        dropped
    }

    pub fn pop(&self) -> Option<CapturedFrame> {
        self.inner.lock().pop_front()
    }

    pub fn depth(&self) -> usize {
        self.inner.lock().len()
    }
}
