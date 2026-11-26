//! Minimalist Monochrome Graphite Design System + Restrained Glacier-Blue Accent.
//!
//! Calm, distraction-free industrial precision aesthetic:
//! - Pure matte charcoal backgrounds (zero neon halos or rainbow clutter)
//! - Single glacier-blue accent (#4F9CF9) + semantic green/red only where meaningful
//! - Crisp spring toggle switches, smooth gliding segmented pills, clean slim sliders,
//!   understated area sparklines, and clear RenderGraph slot diagram.

use egui::{
    Align2, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Frame, Id,
    Margin, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Style, Ui, Vec2, Visuals,
};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

// Minimalist Monochrome Graphite Palette (deep architectural contrast)
pub const BG_CANVAS: Color32 = Color32::from_rgb(15, 17, 22);
pub const BG_SIDEBAR: Color32 = Color32::from_rgb(20, 23, 30);
pub const BG_CARD: Color32 = Color32::from_rgb(26, 30, 39);
pub const BG_ELEVATED: Color32 = Color32::from_rgb(35, 41, 53);
pub const BORDER_SUBTLE: Color32 = Color32::from_rgb(44, 51, 65);
pub const BORDER_SPECULAR: Color32 = Color32::from_rgba_premultiplied(255, 255, 255, 16);

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(240, 243, 248);
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(176, 185, 199);
pub const TEXT_MUTED: Color32 = Color32::from_rgb(120, 130, 147);

// Single Primary Accent + Minimal Semantic Colors
pub const ACCENT: Color32 = Color32::from_rgb(79, 156, 249);
pub const SEMANTIC_OK: Color32 = Color32::from_rgb(66, 196, 126);
pub const SEMANTIC_WARN: Color32 = Color32::from_rgb(228, 168, 56);
pub const SEMANTIC_DANGER: Color32 = Color32::from_rgb(229, 83, 101);

/// Load Windows Chinese system font (Microsoft YaHei / SimHei).
pub fn install_cjk_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyhl.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ];

    let mut loaded = false;
    for path_str in candidates {
        let p = Path::new(path_str);
        if p.exists() {
            if let Ok(bytes) = std::fs::read(p) {
                fonts
                    .font_data
                    .insert("cjk_sys".to_owned(), Arc::new(FontData::from_owned(bytes)));
                fonts
                    .families
                    .entry(FontFamily::Proportional)
                    .or_default()
                    .insert(0, "cjk_sys".to_owned());
                fonts
                    .families
                    .entry(FontFamily::Monospace)
                    .or_default()
                    .push("cjk_sys".to_owned());
                info!(font = path_str, "Loaded Windows CJK UI font");
                loaded = true;
                break;
            }
        }
    }
    if !loaded {
        warn!("No Windows CJK font found; Chinese glyphs may fall back");
    }
    ctx.set_fonts(fonts);
}

/// Apply clean minimalist monochrome theme.
pub fn apply_minimal_theme(ctx: &egui::Context) {
    let mut style: Style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(10.0, 9.0);
    style.spacing.button_padding = Vec2::new(12.0, 6.0);
    style.animation_time = 0.16;

    let mut visuals = Visuals::dark();
    visuals.window_fill = BG_CANVAS;
    visuals.panel_fill = BG_CANVAS;
    visuals.faint_bg_color = BG_ELEVATED;
    visuals.extreme_bg_color = Color32::from_rgb(13, 15, 19);
    visuals.override_text_color = Some(TEXT_PRIMARY);
    visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(79, 156, 249, 48);
    visuals.selection.stroke = Stroke::new(1.0_f32, ACCENT);

    let r = CornerRadius::same(7);
    visuals.widgets.noninteractive.bg_fill = BG_CARD;
    visuals.widgets.noninteractive.weak_bg_fill = BG_CARD;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER_SUBTLE);
    visuals.widgets.noninteractive.corner_radius = r;
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);

    visuals.widgets.inactive.bg_fill = BG_ELEVATED;
    visuals.widgets.inactive.weak_bg_fill = BG_ELEVATED;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER_SUBTLE);
    visuals.widgets.inactive.corner_radius = r;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);

    visuals.widgets.hovered.bg_fill = Color32::from_rgb(43, 49, 61);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(43, 49, 61);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(84, 96, 118));
    visuals.widgets.hovered.corner_radius = r;
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, Color32::WHITE);

    visuals.widgets.active.bg_fill = Color32::from_rgb(35, 62, 98);
    visuals.widgets.active.bg_stroke = Stroke::new(1.2_f32, ACCENT);
    visuals.widgets.active.corner_radius = r;
    visuals.widgets.active.fg_stroke = Stroke::new(1.2_f32, Color32::WHITE);

    style.visuals = visuals;
    ctx.set_style(style);
}

/// Strictly bounded horizontal sub-column (clamps `max_rect` so inner `available_width()`
/// never bleeds past `width` inside `ui.horizontal` / `ui.horizontal_top`).
pub fn fixed_box(ui: &mut Ui, width: f32, add_contents: impl FnOnce(&mut Ui)) {
    let w = width.max(40.0);
    ui.allocate_ui_with_layout(
        Vec2::new(w, 0.0),
        egui::Layout::top_down(egui::Align::LEFT),
        |ui| {
            ui.set_width(w);
            ui.set_max_width(w);
            add_contents(ui);
        },
    );
}

/// Clean elevated matte card with crisp 1px border and subtle specular top-edge bevel.
pub fn panel_card(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui)) {
    let resp = Frame::NONE
        .fill(BG_CARD)
        .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::same(16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add_contents(ui);
        });
    let r = resp.response.rect;
    if r.width() > 28.0 {
        ui.painter().line_segment(
            [
                Pos2::new(r.left() + 12.0, r.top() + 1.0),
                Pos2::new(r.right() - 12.0, r.top() + 1.0),
            ],
            Stroke::new(1.0_f32, BORDER_SPECULAR),
        );
    }
}

/// Unified minimalist tactile pill/ghost button matching the design system.
pub fn ghost_button(ui: &mut Ui, label: &str, active: bool, enabled: bool) -> bool {
    let galley =
        ui.painter()
            .layout_no_wrap(label.to_string(), FontId::proportional(12.2), TEXT_PRIMARY);
    let size = Vec2::new((galley.size().x + 22.0).max(52.0), 29.0);
    let sense = if enabled { Sense::click() } else { Sense::hover() };
    let (rect, resp) = ui.allocate_exact_size(size, sense);
    let hovered = enabled && resp.hovered();

    let (fill, border, txt_col) = if !enabled {
        (
            Color32::from_rgb(28, 32, 40),
            Color32::from_rgb(38, 43, 54),
            TEXT_MUTED,
        )
    } else if active {
        (
            Color32::from_rgb(31, 48, 72),
            Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), 185),
            Color32::WHITE,
        )
    } else if hovered {
        (
            Color32::from_rgb(45, 53, 68),
            Color32::from_rgb(76, 88, 110),
            Color32::WHITE,
        )
    } else {
        (BG_ELEVATED, BORDER_SUBTLE, TEXT_SECONDARY)
    };

    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(7), fill);
    p.rect_stroke(
        rect,
        CornerRadius::same(7),
        Stroke::new(1.0_f32, border),
        StrokeKind::Inside,
    );
    if enabled && (active || hovered) {
        p.line_segment(
            [
                Pos2::new(rect.left() + 8.0, rect.top() + 1.0),
                Pos2::new(rect.right() - 8.0, rect.top() + 1.0),
            ],
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 24)),
        );
    }
    p.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(12.2),
        txt_col,
    );
    enabled && resp.clicked()
}

/// Understated destructive-action ghost button (turns subtle crimson on hover).
pub fn danger_ghost_button(ui: &mut Ui, label: &str) -> bool {
    let galley =
        ui.painter()
            .layout_no_wrap(label.to_string(), FontId::proportional(12.0), TEXT_SECONDARY);
    let size = Vec2::new((galley.size().x + 20.0).max(48.0), 29.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let hov = resp.hovered();
    let p = ui.painter();
    p.rect_filled(
        rect,
        CornerRadius::same(7),
        if hov {
            Color32::from_rgba_unmultiplied(SEMANTIC_DANGER.r(), SEMANTIC_DANGER.g(), SEMANTIC_DANGER.b(), 36)
        } else {
            BG_ELEVATED
        },
    );
    p.rect_stroke(
        rect,
        CornerRadius::same(7),
        Stroke::new(
            1.0_f32,
            if hov {
                Color32::from_rgba_unmultiplied(SEMANTIC_DANGER.r(), SEMANTIC_DANGER.g(), SEMANTIC_DANGER.b(), 150)
            } else {
                BORDER_SUBTLE
            },
        ),
        StrokeKind::Inside,
    );
    p.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(12.0),
        if hov { SEMANTIC_DANGER } else { TEXT_MUTED },
    );
    resp.clicked()
}

/// Compact inline keycap/tag chip for summaries and metadata.
pub fn inline_tag(ui: &mut Ui, label: &str, value: &str, highlight: bool) {
    Frame::NONE
        .fill(if highlight {
            Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), 22)
        } else {
            BG_ELEVATED
        })
        .stroke(Stroke::new(
            1.0_f32,
            if highlight {
                Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), 90)
            } else {
                BORDER_SUBTLE
            },
        ))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(8, 3))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.label(RichText::new(label).size(11.0).color(TEXT_MUTED));
                ui.label(
                    RichText::new(value)
                        .size(11.2)
                        .strong()
                        .color(if highlight { ACCENT } else { TEXT_PRIMARY }),
                );
            });
        });
}

/// Minimalist Chinese-only section heading with plain 1-line explanation.
pub fn section_heading(ui: &mut Ui, title: &str, desc: &str) {
    ui.label(RichText::new(title).strong().size(14.5).color(TEXT_PRIMARY));
    if !desc.is_empty() {
        ui.label(RichText::new(desc).size(11.5).color(TEXT_MUTED));
    }
    ui.add_space(10.0);
}

/// Restrained status pill with subtle breathing dot when active.
pub fn status_pill(ui: &mut Ui, text: &str, dot_color: Color32, pulse_phase: Option<f32>) {
    Frame::NONE
        .fill(BG_ELEVATED)
        .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
        .corner_radius(CornerRadius::same(14))
        .inner_margin(Margin::symmetric(11, 4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let (dot_rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                let c = dot_rect.center();
                if let Some(t) = pulse_phase {
                    let wave = (t * 1.8).fract();
                    let halo_a = ((1.0 - wave) * 95.0) as u8;
                    ui.painter().circle_stroke(
                        c,
                        3.0 + wave * 4.0,
                        Stroke::new(
                            1.1_f32,
                            Color32::from_rgba_unmultiplied(
                                dot_color.r(),
                                dot_color.g(),
                                dot_color.b(),
                                halo_a,
                            ),
                        ),
                    );
                }
                ui.painter().circle_filled(c, 3.2, dot_color);
                ui.label(RichText::new(text).size(11.8).color(TEXT_SECONDARY));
            });
        });
}

/// Clean minimalist toggle switch with smooth thumb glide.
pub fn clean_toggle(
    ui: &mut Ui,
    id_salt: &str,
    value: &mut bool,
    label: &str,
    desc: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            ui.label(RichText::new(label).size(13.0).color(TEXT_PRIMARY));
            if !desc.is_empty() {
                ui.label(RichText::new(desc).size(11.2).color(TEXT_MUTED));
            }
        });

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (rect, resp) = ui.allocate_exact_size(Vec2::new(40.0, 22.0), Sense::click());
            if resp.clicked() {
                *value = !*value;
                changed = true;
            }
            let t = ui.ctx().animate_bool_with_time(Id::new(id_salt), *value, 0.16);
            let painter = ui.painter();

            let track = lerp_color(Color32::from_rgb(46, 52, 65), ACCENT, t);
            painter.rect_filled(rect, CornerRadius::same(11), track);

            let cx = egui::lerp((rect.left() + 11.0)..=(rect.right() - 11.0), t);
            painter.circle_filled(Pos2::new(cx, rect.center().y), 7.6, Color32::WHITE);
        });
    });
    changed
}

/// Clean gliding segmented control (single accent, zero rainbow clutter).
pub fn segmented_control<T: PartialEq + Clone>(
    ui: &mut Ui,
    id_salt: &str,
    current: &mut T,
    options: &[(T, &str)],
) -> bool {
    let mut changed = false;
    let count = options.len().max(1);
    let selected_idx = options.iter().position(|(v, _)| v == current).unwrap_or(0);
    let anim_idx = ui
        .ctx()
        .animate_value_with_time(Id::new(id_salt), selected_idx as f32, 0.18);

    let avail_w = ui.available_width().max(160.0);
    let h = 32.0;
    let (track_rect, _) = ui.allocate_exact_size(Vec2::new(avail_w, h), Sense::hover());

    ui.painter()
        .rect_filled(track_rect, CornerRadius::same(8), Color32::from_rgb(17, 20, 26));
    ui.painter().rect_stroke(
        track_rect,
        CornerRadius::same(8),
        Stroke::new(1.0_f32, BORDER_SUBTLE),
        StrokeKind::Inside,
    );

    let pad = 3.0;
    let seg_w = (track_rect.width() - pad * 2.0) / count as f32;
    let capsule = Rect::from_min_size(
        Pos2::new(track_rect.left() + pad + anim_idx * seg_w, track_rect.top() + pad),
        Vec2::new(seg_w, h - pad * 2.0),
    );
    ui.painter()
        .rect_filled(capsule, CornerRadius::same(6), Color32::from_rgb(42, 51, 66));
    ui.painter().rect_stroke(
        capsule,
        CornerRadius::same(6),
        Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(79, 156, 249, 165)),
        StrokeKind::Inside,
    );
    ui.painter().line_segment(
        [
            Pos2::new(capsule.left() + 6.0, capsule.top() + 1.0),
            Pos2::new(capsule.right() - 6.0, capsule.top() + 1.0),
        ],
        Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 22)),
    );

    for (idx, (val, label)) in options.iter().enumerate() {
        let seg_rect = Rect::from_min_size(
            Pos2::new(track_rect.left() + pad + idx as f32 * seg_w, track_rect.top()),
            Vec2::new(seg_w, h),
        );
        let resp = ui.interact(seg_rect, Id::new((id_salt, idx)), Sense::click());
        if resp.clicked() && *current != *val {
            *current = val.clone();
            changed = true;
        }
        let is_sel = idx == selected_idx;
        let col = if is_sel {
            Color32::WHITE
        } else if resp.hovered() {
            TEXT_PRIMARY
        } else {
            TEXT_MUTED
        };
        ui.painter().text(
            seg_rect.center(),
            Align2::CENTER_CENTER,
            *label,
            FontId::proportional(12.2),
            col,
        );
    }

    changed
}

/// Clean slim slider with plain-language description and crisp tabular value readout.
pub fn clean_slider_f32(
    ui: &mut Ui,
    id_salt: &str,
    label: &str,
    desc: &str,
    value: &mut f32,
    min: f32,
    max: f32,
    decimals: usize,
    suffix: &str,
) -> bool {
    let mut changed = false;
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 3.0;
        ui.horizontal(|ui| {
            ui.label(RichText::new(label).size(12.8).color(TEXT_PRIMARY));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let text = format!("{:.*} {}", decimals, *value, suffix);
                ui.label(
                    RichText::new(text.trim())
                        .monospace()
                        .strong()
                        .size(12.0)
                        .color(ACCENT),
                );
            });
        });
        if !desc.is_empty() {
            ui.label(RichText::new(desc).size(11.0).color(TEXT_MUTED));
        }

        let w = ui.available_width().max(120.0);
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, 18.0), Sense::click_and_drag());
        if resp.dragged() || resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let norm = ((pos.x - (rect.left() + 7.0)) / (rect.width() - 14.0).max(1.0))
                    .clamp(0.0, 1.0);
                let new_v = min + norm * (max - min);
                if (new_v - *value).abs() > 1e-6 {
                    *value = new_v;
                    changed = true;
                }
            }
        }

        let norm = ((*value - min) / (max - min).max(1e-6)).clamp(0.0, 1.0);
        let hov = ui
            .ctx()
            .animate_bool_with_time(Id::new((id_salt, "h")), resp.hovered() || resp.dragged(), 0.12);
        let painter = ui.painter();
        let groove = Rect::from_min_max(
            Pos2::new(rect.left() + 7.0, rect.center().y - 2.0),
            Pos2::new(rect.right() - 7.0, rect.center().y + 2.0),
        );
        painter.rect_filled(groove, CornerRadius::same(2), Color32::from_rgb(42, 48, 60));
        let thumb_x = groove.left() + groove.width() * norm;
        let fill = Rect::from_min_max(groove.min, Pos2::new(thumb_x, groove.max.y));
        if fill.width() > 1.0 {
            painter.rect_filled(fill, CornerRadius::same(2), ACCENT);
        }
        if hov > 0.05 {
            painter.circle_filled(
                Pos2::new(thumb_x, rect.center().y),
                9.5,
                Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), (hov * 52.0) as u8),
            );
        }
        painter.circle_filled(Pos2::new(thumb_x, rect.center().y), 6.0 + hov * 1.1, Color32::WHITE);
    });
    changed
}

pub fn clean_slider_int(
    ui: &mut Ui,
    id_salt: &str,
    label: &str,
    desc: &str,
    value: &mut u32,
    min: u32,
    max: u32,
    suffix: &str,
) -> bool {
    let mut f = *value as f32;
    let _ = clean_slider_f32(ui, id_salt, label, desc, &mut f, min as f32, max as f32, 0, suffix);
    let rounded = f.round().clamp(min as f32, max as f32) as u32;
    if rounded != *value {
        *value = rounded;
        true
    } else {
        false
    }
}

/// Compact calm metric box for telemetry readouts (fills its column width cleanly).
pub fn stat_box(ui: &mut Ui, label: &str, value: &str, sub: &str, emphasize: bool) {
    Frame::NONE
        .fill(BG_ELEVATED)
        .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(label).size(11.0).color(TEXT_MUTED));
            ui.label(
                RichText::new(value)
                    .monospace()
                    .strong()
                    .size(16.5)
                    .color(if emphasize { ACCENT } else { TEXT_PRIMARY }),
            );
            if !sub.is_empty() {
                ui.label(RichText::new(sub).size(10.5).color(TEXT_MUTED));
            }
        });
}

/// Understated monochrome-blue area sparkline chart.
pub fn clean_sparkline(
    ui: &mut Ui,
    title: &str,
    values: &VecDeque<f32>,
    min_v: f32,
    max_v: f32,
    unit: &str,
    height: f32,
) {
    let latest = values.back().copied().unwrap_or(0.0);
    ui.horizontal(|ui| {
        ui.label(RichText::new(title).size(11.8).color(TEXT_MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(format!("{latest:.1}{unit}"))
                    .monospace()
                    .strong()
                    .size(12.0)
                    .color(TEXT_PRIMARY),
            );
        });
    });

    let avail_w = ui.available_width().max(120.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(avail_w, height), Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, CornerRadius::same(6), Color32::from_rgb(17, 20, 26));
    painter.rect_stroke(
        rect,
        CornerRadius::same(6),
        Stroke::new(1.0_f32, BORDER_SUBTLE),
        StrokeKind::Inside,
    );

    if values.len() < 2 {
        return;
    }
    let range = (max_v - min_v).max(1e-4);
    let pad = 6.0;
    let inner = Rect::from_min_max(
        Pos2::new(rect.left() + pad, rect.top() + pad),
        Pos2::new(rect.right() - pad, rect.bottom() - pad),
    );
    let n = values.len();
    let pts: Vec<Pos2> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = inner.left() + inner.width() * (i as f32 / (n - 1) as f32);
            let norm = ((v - min_v) / range).clamp(0.0, 1.0);
            let y = inner.bottom() - inner.height() * norm;
            Pos2::new(x, y)
        })
        .collect();

    let fill = Color32::from_rgba_unmultiplied(ACCENT.r(), ACCENT.g(), ACCENT.b(), 26);
    for w in pts.windows(2) {
        let poly = vec![
            w[0],
            w[1],
            Pos2::new(w[1].x, inner.bottom()),
            Pos2::new(w[0].x, inner.bottom()),
        ];
        painter.add(egui::Shape::convex_polygon(poly, fill, Stroke::NONE));
        painter.line_segment([w[0], w[1]], Stroke::new(1.6_f32, ACCENT));
    }
}

/// Clean monochrome RenderGraph Lifetime & VRAM Slot Aliasing diagram.
pub fn render_graph_gantt(ui: &mut Ui, aliasing_enabled: bool) {
    let avail_w = ui.available_width().max(520.0);
    let h = 166.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(avail_w, h), Sense::hover());
    let painter = ui.painter();

    painter.rect_filled(rect, CornerRadius::same(8), Color32::from_rgb(17, 20, 26));
    painter.rect_stroke(
        rect,
        CornerRadius::same(8),
        Stroke::new(1.0_f32, BORDER_SUBTLE),
        StrokeKind::Inside,
    );

    let left_w = 108.0;
    let top_h = 24.0;
    let passes = ["捕获", "金字塔A", "金字塔B", "光流", "UI检测", "膨胀", "插帧合成", "超分锐化", "上屏"];
    let grid_left = rect.left() + left_w;
    let col_w = (rect.width() - left_w - 12.0) / passes.len() as f32;

    for (i, &name) in passes.iter().enumerate() {
        let x = grid_left + i as f32 * col_w;
        painter.line_segment(
            [Pos2::new(x, rect.top() + top_h), Pos2::new(x, rect.bottom() - 6.0)],
            Stroke::new(0.7_f32, Color32::from_rgb(32, 37, 46)),
        );
        painter.text(
            Pos2::new(x + col_w * 0.5, rect.top() + 13.0),
            Align2::CENTER_CENTER,
            name,
            FontId::proportional(10.0),
            TEXT_MUTED,
        );
    }

    let neutral_bar = Color32::from_rgb(76, 88, 108);
    let rows: Vec<(&str, Vec<(&str, usize, usize, Color32)>)> = if aliasing_enabled {
        vec![
            ("显存槽 1", vec![("前帧金字塔", 1, 3, neutral_bar)]),
            ("显存槽 2", vec![("当前金字塔", 2, 3, neutral_bar)]),
            ("显存槽 3", vec![("光流矢量场", 3, 6, neutral_bar)]),
            (
                "显存槽 4 (复用)",
                vec![
                    ("UI 掩码", 4, 5, ACCENT),
                    ("插帧缓冲 (共享同槽)", 6, 7, ACCENT),
                ],
            ),
            ("显存槽 5", vec![("最终输出", 7, 8, neutral_bar)]),
        ]
    } else {
        vec![
            ("显存槽 1", vec![("前帧金字塔", 1, 3, neutral_bar)]),
            ("显存槽 2", vec![("当前金字塔", 2, 3, neutral_bar)]),
            ("显存槽 3", vec![("光流矢量场", 3, 6, neutral_bar)]),
            ("显存槽 4 (独立)", vec![("UI 掩码", 4, 5, neutral_bar)]),
            ("显存槽 5 (独立)", vec![("插帧缓冲", 6, 7, neutral_bar)]),
            ("显存槽 6", vec![("最终输出", 7, 8, neutral_bar)]),
        ]
    };

    let row_h = ((rect.height() - top_h - 10.0) / rows.len() as f32).min(25.0);
    for (r_idx, (slot_lbl, segs)) in rows.iter().enumerate() {
        let y0 = rect.top() + top_h + 3.0 + r_idx as f32 * row_h;
        let cy = y0 + row_h * 0.5;
        let is_shared = aliasing_enabled && r_idx == 3;

        painter.text(
            Pos2::new(rect.left() + 10.0, cy),
            Align2::LEFT_CENTER,
            *slot_lbl,
            FontId::proportional(10.8),
            if is_shared { ACCENT } else { TEXT_MUTED },
        );

        for &(res_name, s, e, col) in segs {
            let bx0 = grid_left + s as f32 * col_w + 2.0;
            let bx1 = grid_left + (e + 1) as f32 * col_w - 2.0;
            let bar = Rect::from_min_max(Pos2::new(bx0, y0 + 3.0), Pos2::new(bx1, y0 + row_h - 3.0));
            painter.rect_filled(
                bar,
                CornerRadius::same(4),
                Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 56),
            );
            painter.rect_stroke(
                bar,
                CornerRadius::same(4),
                Stroke::new(1.0_f32, col),
                StrokeKind::Inside,
            );
            painter.text(
                bar.center(),
                Align2::CENTER_CENTER,
                res_name,
                FontId::proportional(10.0),
                TEXT_PRIMARY,
            );
        }
    }
}

/// Crisp keycap badge.
pub fn keycap_badge(ui: &mut Ui, key: &str) {
    Frame::NONE
        .fill(BG_ELEVATED)
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(66, 74, 89)))
        .corner_radius(CornerRadius::same(5))
        .inner_margin(Margin::symmetric(8, 3))
        .show(ui, |ui| {
            ui.label(RichText::new(key).monospace().strong().size(11.5).color(TEXT_PRIMARY));
        });
}

pub fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let l = |x: u8, y: u8| -> u8 { (x as f32 + (y as f32 - x as f32) * t).round() as u8 };
    Color32::from_rgba_unmultiplied(
        l(a.r(), b.r()),
        l(a.g(), b.g()),
        l(a.b(), b.b()),
        l(a.a(), b.a()),
    )
}

/// Clean minimalist 64x64 monochrome + glacier-blue window icon.
pub fn build_app_icon() -> egui::IconData {
    let size = 64u32;
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    let c = (size as f32) * 0.5;
    for y in 0..size {
        for x in 0..size {
            let idx = ((y * size + x) * 4) as usize;
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let r = dx.hypot(dy);
            let in_box = dx.abs() <= 29.0
                && dy.abs() <= 29.0
                && (dx.abs() <= 21.0
                    || dy.abs() <= 21.0
                    || (dx.abs() - 21.0).hypot(dy.abs() - 21.0) <= 8.0);
            if !in_box {
                rgba[idx + 3] = 0;
                continue;
            }
            let mut cr = 20.0_f32;
            let mut cg = 23.0_f32;
            let mut cb = 30.0_f32;
            if (r - 19.5).abs() < 1.8 {
                cr = 92.0;
                cg = 104.0;
                cb = 124.0;
            }
            let bar1 = (dx + 8.5).abs() < 2.2 && dy.abs() < 9.0;
            let bar2 = dx.abs() < 2.4 && dy.abs() < 13.5;
            let bar3 = (dx - 8.5).abs() < 2.2 && dy.abs() < 9.0;
            if bar2 {
                cr = 79.0;
                cg = 156.0;
                cb = 249.0;
            } else if bar1 || bar3 {
                cr = 225.0;
                cg = 230.0;
                cb = 238.0;
            }
            rgba[idx] = cr as u8;
            rgba[idx + 1] = cg as u8;
            rgba[idx + 2] = cb as u8;
            rgba[idx + 3] = 255;
        }
    }
    egui::IconData {
        rgba,
        width: size,
        height: size,
    }
}
