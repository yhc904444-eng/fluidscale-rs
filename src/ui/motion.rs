//! Restrained Spring-Physics Motion System for smooth navigation & page transitions.

/// Semi-implicit damped spring oscillator for silky, subtle navigation glides.
#[derive(Debug, Clone)]
pub struct Spring1D {
    pub value: f32,
    pub target: f32,
    pub velocity: f32,
    pub stiffness: f32,
    pub damping: f32,
}

impl Spring1D {
    pub fn new(initial: f32, stiffness: f32, damping: f32) -> Self {
        Self {
            value: initial,
            target: initial,
            velocity: 0.0,
            stiffness,
            damping,
        }
    }

    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    pub fn step(&mut self, dt: f32) -> bool {
        let dt = dt.clamp(0.001, 0.04);
        let force = (self.target - self.value) * self.stiffness;
        let damp = self.velocity * self.damping;
        self.velocity += (force - damp) * dt;
        self.value += self.velocity * dt;

        let settled = (self.target - self.value).abs() < 0.002 && self.velocity.abs() < 0.01;
        if settled {
            self.value = self.target;
            self.velocity = 0.0;
            false
        } else {
            true
        }
    }
}

/// Clean subtle directional glide + opacity fade when switching pages (zero flashy gimmicks).
#[derive(Debug, Clone)]
pub struct PageTransition {
    pub progress: f32,
    pub direction: f32,
    pub duration_sec: f32,
}

impl PageTransition {
    pub fn new() -> Self {
        Self {
            progress: 1.0,
            direction: 1.0,
            duration_sec: 0.22,
        }
    }

    pub fn trigger(&mut self, from_idx: usize, to_idx: usize) {
        if from_idx != to_idx {
            self.progress = 0.0;
            self.direction = if to_idx > from_idx { 1.0 } else { -1.0 };
        }
    }

    pub fn step(&mut self, dt: f32) -> bool {
        if self.progress < 1.0 {
            self.progress = (self.progress + dt.clamp(0.001, 0.05) / self.duration_sec).min(1.0);
            true
        } else {
            false
        }
    }

    pub fn eased(&self) -> f32 {
        let inv = 1.0 - self.progress.clamp(0.0, 1.0);
        1.0 - inv * inv * inv
    }

    pub fn slide_offset_y(&self) -> f32 {
        (1.0 - self.eased()) * 14.0 * self.direction
    }

    pub fn opacity(&self) -> f32 {
        (0.15 + self.eased() * 0.85).clamp(0.0, 1.0)
    }
}
