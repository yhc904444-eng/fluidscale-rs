//! Minimalist crisp geometric vector icons painted directly via `egui::Painter`.

use egui::{Color32, CornerRadius, Painter, Pos2, Rect, Stroke, StrokeKind, Vec2};
use std::f32::consts::TAU;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavIconKind {
    Scaling,
    LiveLab,
    HudEditor,
    Capture,
    Diagnostics,
}

/// Clean minimalist monochrome brand mark (precision ring + 3 subtle frame bars).
pub fn draw_brand_emblem(painter: &Painter, center: Pos2, radius: f32, accent: Color32) {
    painter.circle_filled(center, radius, Color32::from_rgb(24, 28, 36));
    painter.circle_stroke(
        center,
        radius,
        Stroke::new(1.3_f32, Color32::from_rgb(56, 65, 82)),
    );

    let offsets = [-0.34_f32, 0.0, 0.34];
    let heights = [0.56_f32, 0.86, 0.56];
    for i in 0..3 {
        let h = radius * heights[i];
        let x = center.x + offsets[i] * radius;
        let col = if i == 1 {
            accent
        } else {
            Color32::from_rgb(215, 222, 232)
        };
        let bar = Rect::from_center_size(Pos2::new(x, center.y), Vec2::new(2.4, h));
        painter.rect_filled(bar, CornerRadius::same(2), col);
    }
}

/// Crisp geometric icons for the 5 navigation tabs.
pub fn draw_nav_icon(
    painter: &Painter,
    center: Pos2,
    size: f32,
    kind: NavIconKind,
    color: Color32,
) {
    let r = size * 0.44;
    let stroke = Stroke::new(1.5_f32, color);

    match kind {
        NavIconKind::Scaling => {
            let left = Rect::from_center_size(
                center + Vec2::new(-r * 0.26, r * 0.14),
                Vec2::splat(r * 1.02),
            );
            let right = Rect::from_center_size(
                center + Vec2::new(r * 0.26, -r * 0.14),
                Vec2::splat(r * 1.02),
            );
            painter.rect_stroke(left, CornerRadius::same(2), stroke, StrokeKind::Middle);
            painter.rect_stroke(right, CornerRadius::same(2), stroke, StrokeKind::Middle);
        }
        NavIconKind::LiveLab => {
            painter.circle_stroke(center, r * 0.86, stroke);
            painter.line_segment(
                [
                    Pos2::new(center.x, center.y - r * 0.92),
                    Pos2::new(center.x, center.y + r * 0.92),
                ],
                stroke,
            );
            painter.circle_filled(center + Vec2::new(-r * 0.36, 0.0), r * 0.18, color);
        }
        NavIconKind::HudEditor => {
            let b = r * 0.84;
            let t = r * 0.36;
            for (sx, sy) in [(-1.0_f32, -1.0_f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let corner = center + Vec2::new(sx * b, sy * b);
                painter.line_segment([corner, corner + Vec2::new(-sx * t, 0.0)], stroke);
                painter.line_segment([corner, corner + Vec2::new(0.0, -sy * t)], stroke);
            }
            painter.circle_filled(center, r * 0.2, color);
        }
        NavIconKind::Capture => {
            let screen = Rect::from_center_size(
                center + Vec2::new(0.0, -r * 0.12),
                Vec2::new(r * 1.72, r * 1.22),
            );
            painter.rect_stroke(screen, CornerRadius::same(2), stroke, StrokeKind::Middle);
            painter.line_segment(
                [
                    Pos2::new(center.x - r * 0.42, center.y + r * 0.82),
                    Pos2::new(center.x + r * 0.42, center.y + r * 0.82),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(center.x, center.y + r * 0.50),
                    Pos2::new(center.x, center.y + r * 0.82),
                ],
                stroke,
            );
        }
        NavIconKind::Diagnostics => {
            let frame = Rect::from_center_size(center, Vec2::splat(r * 1.68));
            painter.rect_stroke(frame, CornerRadius::same(3), stroke, StrokeKind::Middle);
            let heights = [0.42_f32, 0.92, 0.65];
            for (i, &h) in heights.iter().enumerate() {
                let x = center.x + (i as f32 - 1.0) * r * 0.44;
                let y0 = center.y + r * 0.46;
                let y1 = y0 - h * r * 0.88;
                painter.line_segment(
                    [Pos2::new(x, y0), Pos2::new(x, y1)],
                    Stroke::new(1.8_f32, color),
                );
            }
        }
    }
}

/// Play / Stop / Countdown transport glyph.
pub fn draw_transport_glyph(
    painter: &Painter,
    center: Pos2,
    radius: f32,
    running: bool,
    countdown_frac: Option<f32>,
    color: Color32,
) {
    if let Some(frac) = countdown_frac {
        painter.circle_stroke(
            center,
            radius,
            Stroke::new(
                1.5_f32,
                Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 65),
            ),
        );
        let segs = 20;
        let sweep = TAU * frac.clamp(0.05, 1.0);
        let pts: Vec<Pos2> = (0..=segs)
            .map(|i| {
                let a = -std::f32::consts::FRAC_PI_2 + (i as f32 / segs as f32) * sweep;
                center + Vec2::new(a.cos() * radius, a.sin() * radius)
            })
            .collect();
        for w in pts.windows(2) {
            painter.line_segment([w[0], w[1]], Stroke::new(2.0_f32, color));
        }
    } else if running {
        let sq = Rect::from_center_size(center, Vec2::splat(radius * 1.1));
        painter.rect_filled(sq, CornerRadius::same(2), color);
    } else {
        let tri = vec![
            center + Vec2::new(-radius * 0.5, -radius * 0.7),
            center + Vec2::new(radius * 0.82, 0.0),
            center + Vec2::new(-radius * 0.5, radius * 0.7),
        ];
        painter.add(egui::Shape::convex_polygon(tri, color, Stroke::NONE));
    }
}

/// Minimalist save icon.
pub fn draw_save_glyph(painter: &Painter, center: Pos2, size: f32, color: Color32) {
    let r = size * 0.42;
    let stroke = Stroke::new(1.4_f32, color);
    let box_r = Rect::from_center_size(center, Vec2::splat(r * 1.75));
    painter.rect_stroke(box_r, CornerRadius::same(2), stroke, StrokeKind::Middle);
    let slot = Rect::from_min_max(
        Pos2::new(center.x - r * 0.42, center.y - r * 0.82),
        Pos2::new(center.x + r * 0.34, center.y - r * 0.24),
    );
    painter.rect_filled(slot, CornerRadius::same(1), color);
    painter.line_segment(
        [
            Pos2::new(center.x - r * 0.42, center.y + r * 0.42),
            Pos2::new(center.x + r * 0.42, center.y + r * 0.42),
        ],
        stroke,
    );
}
