pub mod icons;
pub mod live_lab;
pub mod motion;
pub mod overlay_manager;
pub mod theme;

use crate::app::{run_benchmark_report, BenchmarkReport};
use crate::config::{wildcard_match, AppConfig, GameProfile};
use crate::gpu::context::{AdapterSummary, GpuContext};
use crate::platform::hotkey::GlobalHotkeyListener;
use crate::platform::win32::{get_foreground_window_info, list_visible_windows, WindowInfo};
use crate::ui::icons::{
    draw_brand_emblem, draw_nav_icon, draw_save_glyph, draw_transport_glyph, NavIconKind,
};
use crate::ui::live_lab::LiveGpuLab;
use crate::ui::motion::{PageTransition, Spring1D};
use crate::ui::overlay_manager::{OverlayManager, OverlayStatus};
use crate::ui::theme::{
    clean_slider_f32, clean_slider_int, clean_sparkline, clean_toggle, danger_ghost_button,
    fixed_box, ghost_button, inline_tag, keycap_badge, panel_card, render_graph_gantt,
    section_heading, segmented_control, stat_box, status_pill, ACCENT, BG_CANVAS, BG_CARD,
    BG_ELEVATED, BG_SIDEBAR, BORDER_SUBTLE, SEMANTIC_DANGER, SEMANTIC_OK, SEMANTIC_WARN,
    TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY,
};
use anyhow::{anyhow, Result};
use egui::{
    Align, Align2, Color32, CornerRadius, FontId, Frame, Id, Layout, Margin, Pos2, Rect, RichText,
    ScrollArea, Sense, Stroke, StrokeKind, TopBottomPanel, Ui, Vec2,
};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NavTab {
    Scaling,
    LiveLab,
    HudEditor,
    Capture,
    Diagnostics,
}

impl NavTab {
    const ALL: [(NavTab, NavIconKind, &'static str); 5] = [
        (NavTab::Scaling, NavIconKind::Scaling, "缩放与补帧"),
        (NavTab::LiveLab, NavIconKind::LiveLab, "实时效果对比"),
        (NavTab::HudEditor, NavIconKind::HudEditor, "HUD 保护框选"),
        (NavTab::Capture, NavIconKind::Capture, "窗口捕获"),
        (NavTab::Diagnostics, NavIconKind::Diagnostics, "显卡与测速"),
    ];

    fn index(self) -> usize {
        match self {
            NavTab::Scaling => 0,
            NavTab::LiveLab => 1,
            NavTab::HudEditor => 2,
            NavTab::Capture => 3,
            NavTab::Diagnostics => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetSelectMode {
    ForegroundCountdown,
    ExplicitWindow,
}

pub fn run_desktop_app(cfg: AppConfig, cfg_path: PathBuf) -> Result<()> {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title("FluidScale")
            .with_inner_size([1240.0, 790.0])
            .with_min_inner_size([980.0, 640.0])
            .with_icon(Arc::new(theme::build_app_icon())),
        ..Default::default()
    };

    eframe::run_native(
        "FluidScale",
        options,
        Box::new(move |cc| Ok(Box::new(FluidScaleApp::new(cc, cfg, cfg_path)))),
    )
    .map_err(|e| anyhow!("eframe launch failed: {e}"))
}

pub struct FluidScaleApp {
    cfg: AppConfig,
    cfg_path: PathBuf,
    dirty: bool,

    active_tab: NavTab,
    nav_spring: Spring1D,
    page_transition: PageTransition,
    app_start: Instant,
    last_frame_instant: Instant,
    show_advanced: bool,

    selected_profile_idx: usize,
    new_profile_name: String,
    new_profile_pattern: String,
    show_new_profile_box: bool,

    gpu_adapters: Vec<AdapterSummary>,
    active_gpu_label: String,
    visible_windows: Vec<WindowInfo>,
    target_mode: TargetSelectMode,
    selected_window_title: Option<String>,

    live_lab: Option<LiveGpuLab>,
    live_lab_error: Option<String>,
    hud_drag_start: Option<Pos2>,

    overlay: OverlayManager,
    _hotkey: GlobalHotkeyListener,
    last_auto_scale_poll: Instant,
    suppressed_auto_hwnd: Option<isize>,
    active_preset: Option<&'static str>,

    bench_rx: Option<Receiver<Result<BenchmarkReport>>>,
    bench_running: bool,
    bench_src_w: u32,
    bench_src_h: u32,
    bench_dst_w: u32,
    bench_dst_h: u32,
    bench_frames: u32,
    last_bench: Option<BenchmarkReport>,

    toast: Option<(String, Color32, Instant)>,
}

impl FluidScaleApp {
    fn new(cc: &eframe::CreationContext<'_>, mut cfg: AppConfig, cfg_path: PathBuf) -> Self {
        theme::install_cjk_fonts(&cc.egui_ctx);
        theme::apply_minimal_theme(&cc.egui_ctx);

        if cfg.profiles.is_empty() {
            cfg.profiles.push(GameProfile {
                name: "默认配置".into(),
                match_pattern: "*".into(),
                auto_scale: false,
                hud_hints: vec![
                    [0.025, 0.80, 0.245, 0.15],
                    [0.475, 0.465, 0.05, 0.07],
                ],
            });
        }

        let gpu_adapters = GpuContext::enumerate_adapters("auto");
        let active_gpu_label = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| rs.adapter.get_info().name)
            .unwrap_or_else(|| {
                gpu_adapters
                    .first()
                    .map(|g| g.name.clone())
                    .unwrap_or_else(|| "未检测到独立显卡".into())
            });

        let (live_lab, live_lab_error) = match cc.wgpu_render_state.as_ref() {
            Some(rs) => match LiveGpuLab::new(rs, &cfg) {
                Ok(lab) => (Some(lab), None),
                Err(e) => (None, Some(format!("实时预览初始化失败: {e}"))),
            },
            None => (None, Some("未检测到 GPU 渲染上下文".into())),
        };

        let ctx_clone = cc.egui_ctx.clone();
        let hotkey = GlobalHotkeyListener::start(&cfg.capture, move || {
            ctx_clone.request_repaint();
        });

        let visible_windows = list_visible_windows();
        let now = Instant::now();

        Self {
            cfg,
            cfg_path,
            dirty: false,
            active_tab: NavTab::Scaling,
            nav_spring: Spring1D::new(0.0, 320.0, 26.0),
            page_transition: PageTransition::new(),
            app_start: now,
            last_frame_instant: now,
            show_advanced: false,
            selected_profile_idx: 0,
            new_profile_name: String::new(),
            new_profile_pattern: "*.exe".into(),
            show_new_profile_box: false,
            gpu_adapters,
            active_gpu_label,
            visible_windows,
            target_mode: TargetSelectMode::ForegroundCountdown,
            selected_window_title: None,
            live_lab,
            live_lab_error,
            hud_drag_start: None,
            overlay: OverlayManager::new(),
            _hotkey: hotkey,
            last_auto_scale_poll: now,
            suppressed_auto_hwnd: None,
            active_preset: None,
            bench_rx: None,
            bench_running: false,
            bench_src_w: 1920,
            bench_src_h: 1080,
            bench_dst_w: 2560,
            bench_dst_h: 1440,
            bench_frames: 60,
            last_bench: None,
            toast: None,
        }
    }

    fn switch_tab(&mut self, next: NavTab) {
        if self.active_tab != next {
            let old_idx = self.active_tab.index();
            let new_idx = next.index();
            self.active_tab = next;
            self.nav_spring.set_target(new_idx as f32);
            self.page_transition.trigger(old_idx, new_idx);
        }
    }

    fn set_toast(&mut self, msg: impl Into<String>, color: Color32) {
        self.toast = Some((msg.into(), color, Instant::now() + Duration::from_secs(3)));
    }

    fn current_profile(&self) -> Option<&GameProfile> {
        self.cfg.profiles.get(self.selected_profile_idx)
    }

    fn save_config(&mut self) {
        match self.cfg.save(&self.cfg_path) {
            Ok(()) => {
                self.dirty = false;
                self.set_toast("配置已保存", SEMANTIC_OK);
            }
            Err(e) => self.set_toast(format!("保存失败: {e}"), SEMANTIC_DANGER),
        }
    }

    fn toggle_scaling(&mut self, from_hotkey: bool) {
        if self.overlay.is_running() || self.overlay.is_counting_down() {
            self.overlay.stop();
            self.suppressed_auto_hwnd = get_foreground_window_info().map(|w| w.hwnd);
            self.set_toast("已停止缩放", TEXT_SECONDARY);
            return;
        }

        match self.target_mode {
            TargetSelectMode::ExplicitWindow => {
                let target = self.selected_window_title.clone();
                match self.overlay.launch_now(&self.cfg, &self.cfg_path, target) {
                    Ok(title) => {
                        self.dirty = false;
                        self.set_toast(format!("已对「{title}」启用缩放"), SEMANTIC_OK);
                    }
                    Err(e) => self.set_toast(format!("{e}"), SEMANTIC_DANGER),
                }
            }
            TargetSelectMode::ForegroundCountdown => {
                if from_hotkey {
                    match self.overlay.launch_now(&self.cfg, &self.cfg_path, None) {
                        Ok(title) => {
                            self.dirty = false;
                            self.set_toast(format!("热键启动成功：{title}"), SEMANTIC_OK);
                        }
                        Err(e) => self.set_toast(format!("{e}"), SEMANTIC_DANGER),
                    }
                } else {
                    let secs = self.cfg.capture.countdown_secs.clamp(1, 15);
                    self.overlay.start_countdown(secs, None);
                    self.set_toast(
                        format!("{secs} 秒后自动缩放当前窗口，请切到目标游戏"),
                        ACCENT,
                    );
                }
            }
        }
    }

    fn format_hotkey_label(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if self.cfg.capture.hotkey_ctrl {
            parts.push("Ctrl");
        }
        if self.cfg.capture.hotkey_alt {
            parts.push("Alt");
        }
        let k = self.cfg.capture.hotkey_key.trim();
        let key_part = if k.is_empty() { "S" } else { k };
        parts.push(key_part);
        parts.join("+")
    }

    fn reload_hotkey(&mut self, ctx: &egui::Context) {
        let ctx_clone = ctx.clone();
        self._hotkey.restart(&self.cfg.capture, move || {
            ctx_clone.request_repaint();
        });
        if self._hotkey.is_registered() {
            self.set_toast(
                format!("全局快捷键已更新为 {}", self.format_hotkey_label()),
                SEMANTIC_OK,
            );
        } else {
            self.set_toast(
                format!(
                    "快捷键 {} 可能已被其他程序占用，建议换一个按键",
                    self.format_hotkey_label()
                ),
                SEMANTIC_WARN,
            );
        }
    }

    fn poll_auto_scale_profiles(&mut self, now: Instant) {
        if self.overlay.is_running() || self.overlay.is_counting_down() {
            return;
        }
        if now.duration_since(self.last_auto_scale_poll) < Duration::from_millis(800) {
            return;
        }
        self.last_auto_scale_poll = now;
        if !self.cfg.profiles.iter().any(|p| p.auto_scale && p.match_pattern != "*") {
            return;
        }
        let Some(fg) = get_foreground_window_info() else {
            return;
        };
        if fg.pid == std::process::id() {
            return;
        }
        if self.suppressed_auto_hwnd == Some(fg.hwnd) {
            return;
        }
        // User switched to a different window -> clear manual stop suppression
        if self.suppressed_auto_hwnd.is_some() {
            self.suppressed_auto_hwnd = None;
        }
        let matched = self
            .cfg
            .profiles
            .iter()
            .find(|p| p.auto_scale && p.match_pattern != "*" && wildcard_match(&p.match_pattern, &fg.title))
            .map(|p| p.name.clone());
        if let Some(profile_name) = matched {
            match self.overlay.launch_now(&self.cfg, &self.cfg_path, Some(fg.title.clone())) {
                Ok(title) => {
                    self.dirty = false;
                    self.set_toast(
                        format!("配置档「{profile_name}」已自动接管「{}」", truncate_text(&title, 22)),
                        SEMANTIC_OK,
                    );
                }
                Err(e) => self.set_toast(format!("自动缩放启动失败: {e}"), SEMANTIC_DANGER),
            }
        }
    }

    fn start_async_benchmark(&mut self, ctx: egui::Context) {
        if self.bench_running {
            return;
        }
        let cfg_clone = self.cfg.clone();
        let (sw, sh, dw, dh, frames) = (
            self.bench_src_w,
            self.bench_src_h,
            self.bench_dst_w,
            self.bench_dst_h,
            self.bench_frames,
        );
        let (tx, rx) = mpsc::channel();
        self.bench_rx = Some(rx);
        self.bench_running = true;
        self.set_toast("正在运行显卡测速...", ACCENT);

        std::thread::spawn(move || {
            let res = run_benchmark_report(&cfg_clone, sw, sh, dw, dh, frames);
            let _ = tx.send(res);
            ctx.request_repaint();
        });
    }

    fn render_top_bar(&mut self, ctx: &egui::Context, time: f32) {
        TopBottomPanel::top("minimal_top_bar")
            .exact_height(60.0)
            .frame(
                Frame::NONE
                    .fill(BG_SIDEBAR)
                    .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
                    .inner_margin(Margin::symmetric(20, 10)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    let profile_name = self
                        .current_profile()
                        .map(|p| p.name.as_str())
                        .unwrap_or("默认配置");

                    ui.label(
                        RichText::new(profile_name)
                            .strong()
                            .size(15.5)
                            .color(TEXT_PRIMARY),
                    );
                    if self.dirty {
                        ui.label(RichText::new("· 未保存").size(12.0).color(SEMANTIC_WARN));
                    }

                    ui.add_space(12.0);
                    match self.overlay.status() {
                        OverlayStatus::Idle => status_pill(ui, "未运行", TEXT_MUTED, None),
                        OverlayStatus::Countdown { remaining_secs, .. } => status_pill(
                            ui,
                            &format!("{remaining_secs} 秒后开始 · 请切至游戏窗口"),
                            SEMANTIC_WARN,
                            Some(time),
                        ),
                        OverlayStatus::Running {
                            ref target_title,
                            ..
                        } => status_pill(
                            ui,
                            &format!("缩放中 · {}", truncate_text(target_title, 24)),
                            SEMANTIC_OK,
                            Some(time),
                        ),
                    }

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let status = self.overlay.status();
                        let hotkey_txt = self.format_hotkey_label();
                        let cd_total = self.cfg.capture.countdown_secs.max(1) as f32;
                        let (label, chip_txt, fill, txt_col, running, cd_frac) = match status {
                            OverlayStatus::Idle => (
                                "开始缩放".to_string(),
                                Some(hotkey_txt),
                                ACCENT,
                                Color32::WHITE,
                                false,
                                None,
                            ),
                            OverlayStatus::Countdown { remaining_secs, .. } => (
                                format!("取消倒计时 ({remaining_secs}s)"),
                                None,
                                SEMANTIC_WARN,
                                Color32::from_rgb(20, 16, 8),
                                false,
                                Some((remaining_secs as f32 / cd_total).clamp(0.1, 1.0)),
                            ),
                            OverlayStatus::Running { .. } => (
                                "停止缩放".into(),
                                None,
                                SEMANTIC_DANGER,
                                Color32::WHITE,
                                true,
                                None,
                            ),
                        };

                        let cta_w = if chip_txt.is_some() { 208.0 } else { 154.0 };
                        let (cta_rect, cta_resp) =
                            ui.allocate_exact_size(Vec2::new(cta_w, 36.0), Sense::click());
                        let cta_p = ui.painter();
                        cta_p.rect_filled(cta_rect, CornerRadius::same(8), fill);
                        // Subtle architectural top-edge highlight bevel
                        cta_p.line_segment(
                            [
                                Pos2::new(cta_rect.left() + 8.0, cta_rect.top() + 1.0),
                                Pos2::new(cta_rect.right() - 8.0, cta_rect.top() + 1.0),
                            ],
                            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 68)),
                        );
                        if cta_resp.hovered() {
                            cta_p.rect_filled(
                                cta_rect,
                                CornerRadius::same(8),
                                Color32::from_rgba_unmultiplied(255, 255, 255, 22),
                            );
                        }
                        draw_transport_glyph(
                            cta_p,
                            Pos2::new(cta_rect.left() + 19.0, cta_rect.center().y),
                            5.8,
                            running,
                            cd_frac,
                            txt_col,
                        );
                        cta_p.text(
                            Pos2::new(cta_rect.left() + 33.0, cta_rect.center().y),
                            Align2::LEFT_CENTER,
                            label,
                            FontId::proportional(13.0),
                            txt_col,
                        );
                        if let Some(hk) = chip_txt {
                            let galley = cta_p.layout_no_wrap(
                                hk,
                                FontId::monospace(10.5),
                                Color32::from_rgba_unmultiplied(255, 255, 255, 230),
                            );
                            let cw = galley.size().x + 12.0;
                            let chip_rect = Rect::from_min_size(
                                Pos2::new(cta_rect.right() - cw - 8.0, cta_rect.center().y - 10.0),
                                Vec2::new(cw, 20.0),
                            );
                            cta_p.rect_filled(
                                chip_rect,
                                CornerRadius::same(5),
                                Color32::from_rgba_unmultiplied(12, 18, 28, 78),
                            );
                            cta_p.galley(
                                Pos2::new(
                                    chip_rect.left() + 6.0,
                                    chip_rect.center().y - galley.size().y * 0.5,
                                ),
                                galley,
                                Color32::WHITE,
                            );
                        }
                        if cta_resp.clicked() {
                            self.toggle_scaling(false);
                        }

                        ui.add_space(8.0);

                        let (s_rect, s_resp) =
                            ui.allocate_exact_size(Vec2::new(98.0, 36.0), Sense::click());
                        let s_p = ui.painter();
                        s_p.rect_filled(
                            s_rect,
                            CornerRadius::same(8),
                            if s_resp.hovered() {
                                Color32::from_rgb(44, 51, 65)
                            } else {
                                BG_ELEVATED
                            },
                        );
                        s_p.rect_stroke(
                            s_rect,
                            CornerRadius::same(8),
                            Stroke::new(1.0_f32, if self.dirty { ACCENT } else { BORDER_SUBTLE }),
                            StrokeKind::Inside,
                        );
                        draw_save_glyph(
                            s_p,
                            Pos2::new(s_rect.left() + 18.0, s_rect.center().y),
                            12.5,
                            if self.dirty { ACCENT } else { TEXT_SECONDARY },
                        );
                        s_p.text(
                            Pos2::new(s_rect.left() + 32.0, s_rect.center().y),
                            Align2::LEFT_CENTER,
                            "保存设置",
                            FontId::proportional(12.5),
                            TEXT_PRIMARY,
                        );
                        if s_resp.clicked() {
                            self.save_config();
                        }
                    });
                });
            });
    }

    fn render_sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("minimal_sidebar")
            .exact_width(232.0)
            .frame(
                Frame::NONE
                    .fill(BG_SIDEBAR)
                    .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
                    .inner_margin(Margin::same(16)),
            )
            .show(ctx, |ui| {
                // Clean monochrome brand header
                ui.horizontal(|ui| {
                    let (er, _) = ui.allocate_exact_size(Vec2::splat(34.0), Sense::hover());
                    draw_brand_emblem(ui.painter(), er.center(), 15.0, ACCENT);
                    ui.add_space(4.0);
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ui.label(
                            RichText::new("FluidScale")
                                .strong()
                                .size(17.0)
                                .color(Color32::WHITE),
                        );
                        ui.label(
                            RichText::new(truncate_text(&self.active_gpu_label, 22))
                                .size(10.5)
                                .color(TEXT_MUTED),
                        );
                    });
                });

                ui.add_space(20.0);

                // Smooth spring-gliding navigation items
                let item_h = 38.0;
                let gap = 4.0;
                let stride = item_h + gap;
                let nav_w = ui.available_width();
                let total_h = stride * NavTab::ALL.len() as f32 - gap;
                let (nav_rect, _) =
                    ui.allocate_exact_size(Vec2::new(nav_w, total_h), Sense::hover());

                let spring_y = nav_rect.top() + self.nav_spring.value * stride;
                let pill_rect = Rect::from_min_size(
                    Pos2::new(nav_rect.left(), spring_y),
                    Vec2::new(nav_w, item_h),
                );
                ui.painter()
                    .rect_filled(pill_rect, CornerRadius::same(7), Color32::from_rgb(33, 39, 50));
                let bar = Rect::from_min_size(
                    Pos2::new(pill_rect.left() + 3.0, pill_rect.top() + 9.0),
                    Vec2::new(3.0, item_h - 18.0),
                );
                ui.painter().rect_filled(bar, CornerRadius::same(2), ACCENT);

                let mut clicked_tab = None;
                for (idx, (tab, icon_kind, title)) in NavTab::ALL.iter().enumerate() {
                    let r = Rect::from_min_size(
                        Pos2::new(nav_rect.left(), nav_rect.top() + idx as f32 * stride),
                        Vec2::new(nav_w, item_h),
                    );
                    let resp = ui.interact(r, Id::new(("nav_item", idx)), Sense::click());
                    let is_sel = *tab == self.active_tab;
                    if resp.hovered() && !is_sel {
                        ui.painter().rect_filled(
                            r,
                            CornerRadius::same(7),
                            Color32::from_rgba_unmultiplied(255, 255, 255, 8),
                        );
                    }
                    if resp.clicked() {
                        clicked_tab = Some(*tab);
                    }
                    let icon_col = if is_sel {
                        ACCENT
                    } else if resp.hovered() {
                        TEXT_PRIMARY
                    } else {
                        TEXT_MUTED
                    };
                    draw_nav_icon(
                        ui.painter(),
                        Pos2::new(r.left() + 22.0, r.center().y),
                        15.0,
                        *icon_kind,
                        icon_col,
                    );
                    ui.painter().text(
                        Pos2::new(r.left() + 40.0, r.center().y),
                        Align2::LEFT_CENTER,
                        *title,
                        FontId::proportional(13.2),
                        if is_sel { Color32::WHITE } else { TEXT_SECONDARY },
                    );
                }
                if let Some(next) = clicked_tab {
                    self.switch_tab(next);
                }

                ui.add_space(20.0);
                let (div, _) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 1.0), Sense::hover());
                ui.painter().rect_filled(div, 0.0, BORDER_SUBTLE);
                ui.add_space(12.0);

                // Game Profiles
                ui.horizontal(|ui| {
                    ui.label(RichText::new("游戏配置档").size(12.0).color(TEXT_MUTED));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ghost_button(ui, "+ 新建", self.show_new_profile_box, true) {
                            self.show_new_profile_box = !self.show_new_profile_box;
                        }
                    });
                });

                if self.show_new_profile_box {
                    ui.add_space(6.0);
                    Frame::NONE
                        .fill(BG_CARD)
                        .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
                        .corner_radius(CornerRadius::same(8))
                        .inner_margin(Margin::same(10))
                        .show(ui, |ui| {
                            ui.label(RichText::new("名称").size(11.0).color(TEXT_MUTED));
                            ui.text_edit_singleline(&mut self.new_profile_name);
                            ui.label(
                                RichText::new("窗口关键词 (如 *Elden*)")
                                    .size(11.0)
                                    .color(TEXT_MUTED),
                            );
                            ui.text_edit_singleline(&mut self.new_profile_pattern);
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                if ghost_button(
                                    ui,
                                    "确定",
                                    true,
                                    !self.new_profile_name.trim().is_empty(),
                                ) {
                                    self.cfg.profiles.push(GameProfile {
                                        name: self.new_profile_name.trim().to_string(),
                                        match_pattern: if self.new_profile_pattern.trim().is_empty()
                                        {
                                            "*".into()
                                        } else {
                                            self.new_profile_pattern.trim().to_string()
                                        },
                                        auto_scale: false,
                                        hud_hints: Vec::new(),
                                    });
                                    self.selected_profile_idx = self.cfg.profiles.len() - 1;
                                    self.new_profile_name.clear();
                                    self.show_new_profile_box = false;
                                    self.dirty = true;
                                }
                                if ghost_button(ui, "取消", false, true) {
                                    self.show_new_profile_box = false;
                                }
                            });
                        });
                }

                ui.add_space(6.0);
                let mut remove_idx: Option<usize> = None;
                let list_h = (ui.available_height() - 42.0).max(120.0);
                ScrollArea::vertical()
                    .id_salt("min_profile_scroll")
                    .max_height(list_h)
                    .show(ui, |ui| {
                        for (idx, p) in self.cfg.profiles.iter().enumerate() {
                            let is_sel = idx == self.selected_profile_idx;
                            let row_resp = Frame::NONE
                                .fill(if is_sel { BG_ELEVATED } else { Color32::TRANSPARENT })
                                .stroke(Stroke::new(
                                    1.0_f32,
                                    if is_sel {
                                        Color32::from_rgba_unmultiplied(
                                            ACCENT.r(),
                                            ACCENT.g(),
                                            ACCENT.b(),
                                            110,
                                        )
                                    } else {
                                        Color32::TRANSPARENT
                                    },
                                ))
                                .corner_radius(CornerRadius::same(7))
                                .inner_margin(Margin::symmetric(9, 6))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.horizontal(|ui| {
                                        ui.vertical(|ui| {
                                            ui.spacing_mut().item_spacing.y = 1.0;
                                            ui.horizontal(|ui| {
                                                ui.label(
                                                    RichText::new(truncate_text(&p.name, 14))
                                                        .size(12.5)
                                                        .color(if is_sel {
                                                            Color32::WHITE
                                                        } else {
                                                            TEXT_SECONDARY
                                                        }),
                                                );
                                                if p.auto_scale {
                                                    ui.label(
                                                        RichText::new("自动")
                                                            .size(10.0)
                                                            .color(SEMANTIC_OK),
                                                    );
                                                }
                                            });
                                            ui.label(
                                                RichText::new(truncate_text(&p.match_pattern, 18))
                                                    .monospace()
                                                    .size(10.0)
                                                    .color(TEXT_MUTED),
                                            );
                                        });
                                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                            if idx > 0
                                                && ui
                                                    .small_button(RichText::new("×").color(TEXT_MUTED))
                                                    .clicked()
                                            {
                                                remove_idx = Some(idx);
                                            }
                                        });
                                    });
                                })
                                .response
                                .interact(Sense::click());
                            if row_resp.clicked() {
                                self.selected_profile_idx = idx;
                            }
                            ui.add_space(3.0);
                        }
                    });
                if let Some(rm) = remove_idx {
                    self.cfg.profiles.remove(rm);
                    self.selected_profile_idx = self
                        .selected_profile_idx
                        .min(self.cfg.profiles.len().saturating_sub(1));
                    self.dirty = true;
                }

                // Clean sidebar bottom version footer
                ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
                    ui.label(
                        RichText::new("FluidScale v0.1.0 · Rust + wgpu")
                            .size(10.5)
                            .color(TEXT_MUTED),
                    );
                });
            });
    }

    /// Tab 1: Ultra-clean, plain-language parameter panel with Hero Quick-Preset & Pipeline Summary
    fn render_scaling_tab(&mut self, ui: &mut Ui) {
        ScrollArea::vertical()
            .id_salt("min_scaling_scroll")
            .show(ui, |ui| {
                // Hero Preset & Active Pipeline Summary Strip
                panel_card(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            RichText::new("快捷预设")
                                .strong()
                                .size(12.5)
                                .color(TEXT_PRIMARY),
                        );
                        for (id, name, desc) in [
                            ("esports", "竞技低延迟", "2× 补帧 + 最低缓冲延迟"),
                            ("cinematic", "3A 丝滑画质", "锁定 120 FPS + 电影锐利"),
                            ("eco", "核显省电", "高跳过省电 + 低负载光流"),
                        ] {
                            let is_cur = self.active_preset == Some(id);
                            if ghost_button(ui, name, is_cur, true) {
                                self.cfg.apply_preset(id);
                                self.active_preset = Some(id);
                                self.dirty = true;
                                self.set_toast(format!("已切换「{name}」预设：{desc}"), ACCENT);
                            }
                        }
                        if danger_ghost_button(ui, "恢复默认") {
                            self.cfg.apply_preset("reset");
                            self.active_preset = None;
                            self.dirty = true;
                            self.set_toast("已恢复默认参数（保留自定义游戏配置档）", TEXT_SECONDARY);
                        }
                    });

                    ui.add_space(8.0);

                    // Clean 1-line live pipeline overview chips
                    let fg_summary = match self.cfg.frame_gen.mode.as_str() {
                        "off" => "关闭".to_string(),
                        "fixed" => format!("固定 {}×", self.cfg.frame_gen.multiplier),
                        _ => format!("锁定 {:.0} FPS", self.cfg.frame_gen.target_fps),
                    };
                    let up_summary = match self.cfg.upscaling.algorithm.as_str() {
                        "off" => "不放大".to_string(),
                        "lanczos" => format!("电影锐利 {:.0}%", self.cfg.upscaling.sharpness * 100.0),
                        "sharp_bilinear" => "像素风".to_string(),
                        "nearest" => "整数无损".to_string(),
                        _ => format!("通用清晰 {:.0}%", self.cfg.upscaling.sharpness * 100.0),
                    };
                    let ar_summary = if self.cfg.preserve_aspect_ratio() {
                        "保持比例"
                    } else {
                        "拉伸铺满"
                    };
                    let ui_summary = if self.cfg.ui_protection.top_layer_composite {
                        "已开启"
                    } else {
                        "关闭"
                    };

                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("当前管线:").size(11.5).color(TEXT_MUTED));
                        inline_tag(ui, "补帧", &fg_summary, self.cfg.frame_gen.mode != "off");
                        inline_tag(ui, "放大", &up_summary, self.cfg.upscaling.algorithm != "off");
                        inline_tag(ui, "比例", ar_summary, false);
                        inline_tag(ui, "防果冻", ui_summary, self.cfg.ui_protection.top_layer_composite);
                        inline_tag(
                            ui,
                            "省电跳过",
                            if self.cfg.gpu.tile_skip_threshold > 0.0001 { "开启" } else { "关闭" },
                            false,
                        );
                    });
                });

                ui.add_space(12.0);

                ui.columns(2, |cols| {
                    // LEFT COLUMN: Frame Generation + Upscaling
                    cols[0].vertical(|ui| {
                        panel_card(ui, |ui| {
                            section_heading(
                                ui,
                                "补帧与流畅度",
                                "在两帧画面之间自动生成过渡帧，让低帧率游戏变得丝滑",
                            );
                            ui.label(RichText::new("补帧方式").size(12.2).color(TEXT_SECONDARY));
                            ui.add_space(4.0);
                            if segmented_control(
                                ui,
                                "fg_mode_seg",
                                &mut self.cfg.frame_gen.mode,
                                &[
                                    ("off".into(), "关闭"),
                                    ("fixed".into(), "固定倍数"),
                                    ("adaptive".into(), "锁定目标帧率"),
                                ],
                            ) {
                                self.dirty = true;
                            }
                            ui.add_space(12.0);

                            if self.cfg.frame_gen.mode == "fixed" {
                                ui.label(
                                    RichText::new("帧率翻倍档位（原帧率 × 倍数）")
                                        .size(12.2)
                                        .color(TEXT_SECONDARY),
                                );
                                ui.add_space(4.0);
                                if segmented_control(
                                    ui,
                                    "fg_mult_seg",
                                    &mut self.cfg.frame_gen.multiplier,
                                    &[(2, "2 倍流畅"), (3, "3 倍流畅"), (4, "4 倍流畅")],
                                ) {
                                    self.dirty = true;
                                }
                                ui.add_space(12.0);
                            } else if self.cfg.frame_gen.mode == "adaptive" {
                                if clean_slider_f32(
                                    ui,
                                    "fg_target_slider",
                                    "目标输出帧率",
                                    "自动补齐差值帧，建议设为显示器刷新率或略低于刷新率",
                                    &mut self.cfg.frame_gen.target_fps,
                                    48.0,
                                    240.0,
                                    0,
                                    "FPS",
                                ) {
                                    self.dirty = true;
                                }
                                ui.add_space(12.0);
                            }

                            // Plain-language 3-level scene-cut preset
                            let mut cut_level = if self.cfg.frame_gen.scene_cut_threshold <= 0.50 {
                                0u8
                            } else if self.cfg.frame_gen.scene_cut_threshold <= 0.72 {
                                1u8
                            } else {
                                2u8
                            };
                            ui.label(
                                RichText::new("切换镜头 / 菜单防撕裂")
                                    .size(12.2)
                                    .color(TEXT_SECONDARY),
                            );
                            ui.label(
                                RichText::new("画面突然大面积切换时自动暂停补帧 1 帧，避免转场撕裂")
                                    .size(11.0)
                                    .color(TEXT_MUTED),
                            );
                            ui.add_space(4.0);
                            if segmented_control(
                                ui,
                                "scene_cut_preset",
                                &mut cut_level,
                                &[(0, "灵敏"), (1, "标准（推荐）"), (2, "宽松")],
                            ) {
                                self.cfg.frame_gen.scene_cut_threshold = match cut_level {
                                    0 => 0.45,
                                    1 => 0.65,
                                    _ => 0.82,
                                };
                                self.dirty = true;
                            }
                        });

                        ui.add_space(12.0);

                        panel_card(ui, |ui| {
                            section_heading(
                                ui,
                                "画质放大与清晰度",
                                "将低分辨率游戏窗口放大至全屏并增强细节轮廓",
                            );
                            ui.label(RichText::new("放大方式").size(12.2).color(TEXT_SECONDARY));
                            ui.add_space(4.0);
                            if segmented_control(
                                ui,
                                "upscale_seg",
                                &mut self.cfg.upscaling.algorithm,
                                &[
                                    ("off".into(), "不放大"),
                                    ("bicubic_cas".into(), "通用清晰"),
                                    ("lanczos".into(), "电影锐利"),
                                    ("sharp_bilinear".into(), "像素风"),
                                    ("nearest".into(), "整数无损"),
                                ],
                            ) {
                                self.dirty = true;
                            }
                            ui.add_space(12.0);

                            let mut sharp_pct = self.cfg.upscaling.sharpness * 100.0;
                            if clean_slider_f32(
                                ui,
                                "sharp_pct_slider",
                                "边缘锐化强度",
                                "拉高可让文字和材质更清晰，过高可能出现白边",
                                &mut sharp_pct,
                                0.0,
                                100.0,
                                0,
                                "%",
                            ) {
                                self.cfg.upscaling.sharpness = sharp_pct / 100.0;
                                self.active_preset = None;
                                self.dirty = true;
                            }
                            ui.add_space(10.0);

                            ui.label(
                                RichText::new("全屏画面比例（防止 16:9 画面在 16:10 屏幕上拉变形）")
                                    .size(12.0)
                                    .color(TEXT_SECONDARY),
                            );
                            ui.add_space(4.0);
                            if segmented_control(
                                ui,
                                "aspect_ratio_seg",
                                &mut self.cfg.upscaling.aspect_ratio_mode,
                                &[
                                    ("preserve".into(), "保持原画比例（黑边）"),
                                    ("stretch".into(), "拉伸铺满全屏"),
                                ],
                            ) {
                                self.dirty = true;
                            }

                            ui.add_space(10.0);
                            if clean_toggle(
                                ui,
                                "osd_fps_sw",
                                &mut self.cfg.osd.show_fps,
                                "左上角显示帧率",
                                "显示「原始帧率 / 补帧帧率」，纯着色器绘制不增加延迟",
                            ) {
                                self.dirty = true;
                            }
                            if self.cfg.osd.show_fps {
                                ui.add_space(6.0);
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new("帧率字号:").size(11.8).color(TEXT_MUTED));
                                    fixed_box(ui, 168.0, |ui| {
                                        if segmented_control(
                                            ui,
                                            "osd_scale_seg",
                                            &mut self.cfg.osd.scale,
                                            &[(1, "小"), (2, "中"), (3, "大")],
                                        ) {
                                            self.dirty = true;
                                        }
                                    });
                                });
                            }
                        });
                    });

                    // RIGHT COLUMN: UI Protection + Performance Optimization (Plain Language)
                    cols[1].vertical(|ui| {
                        panel_card(ui, |ui| {
                            section_heading(
                                ui,
                                "界面与准星防果冻",
                                "自动识别血条、字幕和准星，防止背景高速移动时把界面拉扯变形",
                            );
                            if clean_toggle(
                                ui,
                                "ui_shield_sw",
                                &mut self.cfg.ui_protection.top_layer_composite,
                                "开启 HUD / 准星防变形保护",
                                "背景插帧时绕开界面边缘，并将清晰原画面界面叠加在最顶层",
                            ) {
                                self.dirty = true;
                            }

                            if self.cfg.ui_protection.top_layer_composite {
                                ui.add_space(12.0);
                                let mut ui_strength = match self.cfg.ui_protection.dilation_radius {
                                    0 | 1 => 0u8,
                                    2 => 1u8,
                                    _ => 2u8,
                                };
                                ui.label(
                                    RichText::new("保护强度（防边缘光晕）")
                                        .size(12.2)
                                        .color(TEXT_SECONDARY),
                                );
                                ui.add_space(4.0);
                                if segmented_control(
                                    ui,
                                    "ui_strength_seg",
                                    &mut ui_strength,
                                    &[(0, "轻度"), (1, "标准（推荐）"), (2, "强力")],
                                ) {
                                    match ui_strength {
                                        0 => {
                                            self.cfg.ui_protection.dilation_radius = 1;
                                            self.cfg.ui_protection.attack_rate = 0.25;
                                            self.cfg.ui_protection.decay_rate = 0.10;
                                            self.cfg.ui_protection.motion_gate_threshold = 1.0;
                                        }
                                        1 => {
                                            self.cfg.ui_protection.dilation_radius = 2;
                                            self.cfg.ui_protection.attack_rate = 0.35;
                                            self.cfg.ui_protection.decay_rate = 0.06;
                                            self.cfg.ui_protection.motion_gate_threshold = 0.75;
                                        }
                                        _ => {
                                            self.cfg.ui_protection.dilation_radius = 3;
                                            self.cfg.ui_protection.attack_rate = 0.50;
                                            self.cfg.ui_protection.decay_rate = 0.04;
                                            self.cfg.ui_protection.motion_gate_threshold = 0.55;
                                        }
                                    }
                                    self.dirty = true;
                                }
                            }
                        });

                        ui.add_space(12.0);

                        panel_card(ui, |ui| {
                            section_heading(
                                ui,
                                "省电、显存与防卡顿优化",
                                "在保证画质的前提下降低显卡占用、显存消耗和微卡顿",
                            );

                            let mut tile_skip_on = self.cfg.gpu.tile_skip_threshold > 0.0001;
                            if clean_toggle(
                                ui,
                                "tile_skip_sw",
                                &mut tile_skip_on,
                                "静止画面智能省电",
                                "自动跳过天空、菜单等静止不动的区块，降低显卡负载和发热",
                            ) {
                                self.cfg.gpu.tile_skip_threshold =
                                    if tile_skip_on { 0.004 } else { 0.0 };
                                self.dirty = true;
                            }
                            ui.add_space(10.0);

                            // 3-level FlowScale preset
                            let mut flow_preset = if self.cfg.gpu.base_flow_scale <= 0.48 {
                                0u8
                            } else if self.cfg.gpu.base_flow_scale <= 0.70 {
                                1u8
                            } else {
                                2u8
                            };
                            ui.label(
                                RichText::new("动态追踪精度")
                                    .size(12.2)
                                    .color(TEXT_SECONDARY),
                            );
                            ui.label(
                                RichText::new("越高高速画面越不易糊；选「省电」更适合核显或 4K 分辨率")
                                    .size(11.0)
                                    .color(TEXT_MUTED),
                            );
                            ui.add_space(4.0);
                            if segmented_control(
                                ui,
                                "flow_preset_seg",
                                &mut flow_preset,
                                &[(0, "省电 (40%)"), (1, "均衡 (55%)"), (2, "精细 (80%)")],
                            ) {
                                self.cfg.gpu.base_flow_scale = match flow_preset {
                                    0 => 0.40,
                                    1 => 0.55,
                                    _ => 0.80,
                                };
                                self.dirty = true;
                            }
                            ui.add_space(10.0);

                            if clean_toggle(
                                ui,
                                "vram_alias_sw",
                                &mut self.cfg.gpu.enable_vram_aliasing,
                                "显存复用压缩",
                                "自动合并复用中间缓存显存，减少显存峰值占用",
                            ) {
                                self.dirty = true;
                            }
                            ui.add_space(8.0);
                            if clean_toggle(
                                ui,
                                "dedup_smooth_sw",
                                &mut self.cfg.pacing.enable_dedup,
                                "帧节奏防卡顿平滑",
                                "自动过滤掉帧尖峰与游戏锁帧产生的重复画面，输出更均匀",
                            ) {
                                self.dirty = true;
                            }
                        });

                        ui.add_space(12.0);

                        // Collapsible Advanced Tweaks (Cleanly tucked away by default!)
                        panel_card(ui, |ui| {
                            let arrow = if self.show_advanced { "▾" } else { "▸" };
                            let hdr = ui.add(
                                egui::Button::new(
                                    RichText::new(format!("{arrow}  高级微调参数（一般保持默认即可）"))
                                        .size(12.5)
                                        .color(TEXT_SECONDARY),
                                )
                                .fill(Color32::TRANSPARENT)
                                .stroke(Stroke::NONE),
                            );
                            if hdr.clicked() {
                                self.show_advanced = !self.show_advanced;
                            }

                            if self.show_advanced {
                                ui.add_space(8.0);
                                ui.label(RichText::new("图形接口").size(12.0).color(TEXT_MUTED));
                                ui.add_space(3.0);
                                if segmented_control(
                                    ui,
                                    "adv_backend_seg",
                                    &mut self.cfg.gpu.backend,
                                    &[
                                        ("auto".into(), "自动"),
                                        ("vulkan".into(), "Vulkan"),
                                        ("dx12".into(), "DirectX 12"),
                                    ],
                                ) {
                                    self.dirty = true;
                                }
                                ui.add_space(8.0);
                                if clean_toggle(
                                    ui,
                                    "adv_dyn_flow",
                                    &mut self.cfg.gpu.dynamic_flow_scale,
                                    "显卡满载时自动降精度保帧",
                                    "检测到单帧耗时过高时临时略降光流分辨率，防止掉帧",
                                ) {
                                    self.dirty = true;
                                }
                                ui.add_space(6.0);
                                if clean_toggle(
                                    ui,
                                    "adv_etw",
                                    &mut self.cfg.pacing.enable_etw,
                                    "高精度时钟与高优先级调度",
                                    "启用高精度 QPC 帧间隔采样并提升显卡调度优先级（管理员模式效果更佳）",
                                ) {
                                    self.dirty = true;
                                }
                                ui.add_space(6.0);
                                let mut q = self.cfg.pacing.queue_target as f32;
                                if clean_slider_f32(
                                    ui,
                                    "adv_queue_depth",
                                    "缓冲队列深度",
                                    "越低延迟越小，越高画面节奏越稳",
                                    &mut q,
                                    1.0,
                                    3.0,
                                    1,
                                    "帧",
                                ) {
                                    self.cfg.pacing.queue_target = q as f64;
                                    self.dirty = true;
                                }
                                ui.add_space(6.0);
                                if clean_slider_f32(
                                    ui,
                                    "adv_tile_thr",
                                    "静止块跳过灵敏度",
                                    "值越大跳过越多省电越多，过大可能导致细微动态变卡",
                                    &mut self.cfg.gpu.tile_skip_threshold,
                                    0.0,
                                    0.025,
                                    4,
                                    "",
                                ) {
                                    self.dirty = true;
                                }
                            }
                        });
                    });
                });
            });
    }

    /// Tab 2: Upgraded 4-Scene Pure-GPU Interactive A/B Comparison Lab!
    fn render_live_lab_tab(&mut self, ui: &mut Ui) {
        let Some(lab) = self.live_lab.as_mut() else {
            panel_card(ui, |ui| {
                ui.label(
                    RichText::new(
                        self.live_lab_error
                            .as_deref()
                            .unwrap_or("实时演示暂不可用"),
                    )
                    .color(SEMANTIC_DANGER),
                );
            });
            return;
        };

        let visible_h = ui.available_height();
        ScrollArea::vertical()
            .id_salt("min_livelab_scroll")
            .show(ui, |ui| {
                let avail_w = ui.available_width();
                let wide = avail_w >= 820.0;

                // Top Clean Control Strip: 4 Scenes + Compare Mode + Input FPS + Speed
                panel_card(ui, |ui| {
                    let inner_w = ui.available_width();
                    if wide {
                        let gap = 12.0;
                        let left_w = ((inner_w - gap) * 0.51).max(260.0);
                        let right_w = (inner_w - gap - left_w).max(240.0);
                        ui.horizontal_top(|ui| {
                            ui.spacing_mut().item_spacing.x = gap;
                            fixed_box(ui, left_w, |ui| {
                                ui.label(
                                    RichText::new("测试场景（切换体验不同极限工况）")
                                        .size(11.8)
                                        .color(TEXT_MUTED),
                                );
                                ui.add_space(3.0);
                                if segmented_control(
                                    ui,
                                    "lab_scene_seg",
                                    &mut lab.scene_id,
                                    &[
                                        (0, "高速夜行公路"),
                                        (1, "FPS 甩枪准星"),
                                        (2, "RPG 半静止省电"),
                                        (3, "清晰度靶图"),
                                    ],
                                ) {
                                    lab.compare_mode = match lab.scene_id {
                                        0 => 2,
                                        1 => 3,
                                        2 => 1,
                                        _ => 2,
                                    };
                                }
                            });

                            fixed_box(ui, right_w, |ui| {
                                ui.label(
                                    RichText::new("左右拉帘对比模式（拖动画面中线直接对比）")
                                        .size(11.8)
                                        .color(TEXT_MUTED),
                                );
                                ui.add_space(3.0);
                                segmented_control(
                                    ui,
                                    "lab_compare_seg",
                                    &mut lab.compare_mode,
                                    &[
                                        (2, "原生低帧 vs 补帧"),
                                        (3, "无保护 vs 防果冻"),
                                        (1, "省算力热力透视"),
                                        (0, "全屏效果"),
                                    ],
                                );
                            });
                        });
                    } else {
                        ui.label(
                            RichText::new("测试场景（切换体验不同极限工况）")
                                .size(11.8)
                                .color(TEXT_MUTED),
                        );
                        ui.add_space(3.0);
                        if segmented_control(
                            ui,
                            "lab_scene_seg",
                            &mut lab.scene_id,
                            &[
                                (0, "高速夜行公路"),
                                (1, "FPS 甩枪准星"),
                                (2, "RPG 半静止省电"),
                                (3, "清晰度靶图"),
                            ],
                        ) {
                            lab.compare_mode = match lab.scene_id {
                                0 => 2,
                                1 => 3,
                                2 => 1,
                                _ => 2,
                            };
                        }
                        ui.add_space(8.0);
                        ui.label(
                            RichText::new("左右拉帘对比模式（拖动画面中线直接对比）")
                                .size(11.8)
                                .color(TEXT_MUTED),
                        );
                        ui.add_space(3.0);
                        segmented_control(
                            ui,
                            "lab_compare_seg",
                            &mut lab.compare_mode,
                            &[
                                (2, "原生低帧 vs 补帧"),
                                (3, "无保护 vs 防果冻"),
                                (1, "省算力热力透视"),
                                (0, "全屏效果"),
                            ],
                        );
                    }

                    ui.add_space(8.0);

                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("模拟游戏原帧率:").size(12.0).color(TEXT_MUTED));
                        fixed_box(ui, 198.0, |ui| {
                            segmented_control(
                                ui,
                                "lab_in_fps_seg",
                                &mut lab.simulated_input_fps,
                                &[(15, "15 FPS"), (20, "20 FPS"), (30, "30 FPS")],
                            );
                        });

                        ui.add_space(8.0);
                        ui.label(RichText::new("运动速度:").size(12.0).color(TEXT_MUTED));
                        fixed_box(ui, 138.0, |ui| {
                            let mut spd = if lab.motion_speed < 0.8 {
                                0u8
                            } else if lab.motion_speed < 1.3 {
                                1u8
                            } else {
                                2u8
                            };
                            if segmented_control(
                                ui,
                                "lab_spd_seg",
                                &mut spd,
                                &[(0, "慢"), (1, "中"), (2, "快")],
                            ) {
                                lab.motion_speed = match spd {
                                    0 => 0.55,
                                    1 => 1.0,
                                    _ => 1.75,
                                };
                            }
                        });

                        ui.add_space(8.0);
                        if ghost_button(ui, "模拟镜头瞬切", false, true) {
                            lab.trigger_scene_cut = true;
                        }
                        if ghost_button(ui, if lab.paused { "继续" } else { "暂停" }, lab.paused, true) {
                            lab.paused = !lab.paused;
                        }
                        if self.cfg.frame_gen.mode.eq_ignore_ascii_case("off") {
                            inline_tag(ui, "状态", "补帧已关闭", false);
                        }
                    });
                });

                ui.add_space(10.0);

                // Compute remaining visible height so the entire 16:9 frame fits cleanly above the fold!
                let top_strip_used = if wide { 142.0 } else { 205.0 };
                let max_img_h = (visible_h - top_strip_used - 74.0).clamp(220.0, 560.0);

                let render_viewport = |ui: &mut Ui, lab: &mut LiveGpuLab, card_w: f32, max_h: f32| {
                    fixed_box(ui, card_w, |ui| {
                        panel_card(ui, |ui| {
                            let inner_w = ui.available_width().max(200.0);
                            let hint = match lab.scene_id {
                                0 => "观察点：左侧为原生卡顿低帧，右侧为实时补帧；拖动中线观察路灯、车道线与左下仪表盘流畅度差异",
                                1 => "观察点：高楼窗格高速横扫屏幕，拖动中线经过中心 1px 十字准星与左下血条，对比有无防果冻保护清晰度",
                                2 => "观察点：上方夜空与底部技能栏完全静止，热力模式下暗色区域已自动跳过光流计算，右侧看省算力比例",
                                _ => "观察点：观察旋转辐射靶图与右侧斜线光栅的流畅度与边缘锐利度（可在「缩放与补帧」调整锐化强度）",
                            };
                            ui.label(RichText::new(hint).size(11.6).color(TEXT_SECONDARY));
                            ui.add_space(6.0);

                            // Strict aspect-ratio fit within BOTH card width and remaining viewport height
                            let aspect = lab.dst_width as f32 / lab.dst_height.max(1) as f32;
                            let mut vp_w = inner_w;
                            let mut vp_h = vp_w / aspect;
                            if vp_h > max_h {
                                vp_h = max_h;
                                vp_w = (vp_h * aspect).min(inner_w);
                            }

                            let (slot_rect, resp) = ui.allocate_exact_size(
                                Vec2::new(inner_w, vp_h),
                                Sense::click_and_drag(),
                            );
                            let x_pad = ((inner_w - vp_w) * 0.5).max(0.0);
                            let img_rect = Rect::from_min_size(
                                Pos2::new(slot_rect.left() + x_pad, slot_rect.top()),
                                Vec2::new(vp_w, vp_h),
                            );

                            let painter = ui.painter();
                            painter.image(
                                lab.texture_id,
                                img_rect,
                                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                Color32::WHITE,
                            );
                            painter.rect_stroke(
                                img_rect,
                                CornerRadius::same(6),
                                Stroke::new(1.0_f32, BORDER_SUBTLE),
                                StrokeKind::Inside,
                            );

                            if lab.compare_mode != 0 {
                                if resp.dragged() || resp.clicked() {
                                    if let Some(pos) = resp.interact_pointer_pos() {
                                        lab.split_x = ((pos.x - img_rect.left()) / img_rect.width())
                                            .clamp(0.05, 0.95);
                                    }
                                }

                                let div_x = img_rect.left() + img_rect.width() * lab.split_x;
                                painter.line_segment(
                                    [
                                        Pos2::new(div_x, img_rect.top()),
                                        Pos2::new(div_x, img_rect.bottom()),
                                    ],
                                    Stroke::new(
                                        1.6_f32,
                                        Color32::from_rgba_unmultiplied(240, 244, 250, 220),
                                    ),
                                );

                                let hc = Pos2::new(div_x, img_rect.center().y);
                                painter.circle_filled(hc, 13.5, Color32::from_rgb(24, 28, 36));
                                painter.circle_stroke(hc, 13.5, Stroke::new(1.4_f32, ACCENT));
                                painter.text(
                                    hc,
                                    Align2::CENTER_CENTER,
                                    "◄ ►",
                                    FontId::proportional(9.2),
                                    Color32::WHITE,
                                );

                                let left_txt = match lab.compare_mode {
                                    2 => format!("原生未补帧 ({} FPS)", lab.simulated_input_fps),
                                    3 => "无界面防果冻保护".into(),
                                    _ => "透视：蓝=锁定界面 · 暗=已跳过".into(),
                                };
                                let right_txt = match lab.compare_mode {
                                    2 => format!("FluidScale 补帧 ({:.0} FPS)", lab.output_fps()),
                                    3 => "开启界面防果冻保护".into(),
                                    _ => "FluidScale 正常画面".into(),
                                };
                                paint_clean_chip(
                                    painter,
                                    Pos2::new(img_rect.left() + 8.0, img_rect.bottom() - 28.0),
                                    &left_txt,
                                    false,
                                );
                                paint_clean_chip(
                                    painter,
                                    Pos2::new(img_rect.right() - 8.0, img_rect.bottom() - 28.0),
                                    &right_txt,
                                    true,
                                );
                            }
                        });
                    });
                };

                let render_telemetry = |ui: &mut Ui, lab: &LiveGpuLab, card_w: f32| {
                    fixed_box(ui, card_w, |ui| {
                        panel_card(ui, |ui| {
                            section_heading(ui, "实时运行数据", "由当前显卡计算后实时回传");
                            ui.columns(2, |cols| {
                                stat_box(
                                    &mut cols[0],
                                    "静止块省算力",
                                    &format!("{:.1}%", lab.latest_skipped_pct),
                                    "跳过无变化区块",
                                    true,
                                );
                                stat_box(
                                    &mut cols[1],
                                    "原帧 / 补帧",
                                    &format!("{:.0}/{:.0}", lab.captured_fps(), lab.output_fps()),
                                    "实时帧率对比",
                                    false,
                                );
                            });
                            ui.add_space(6.0);
                            ui.columns(2, |cols| {
                                stat_box(
                                    &mut cols[0],
                                    "显存占用",
                                    &format!("{:.1} MB", lab.latest_aliased_mb),
                                    &format!("复用省 {:.1}%", lab.latest_vram_saved_pct),
                                    false,
                                );
                                stat_box(
                                    &mut cols[1],
                                    "GPU 耗时",
                                    &format!("{:.2} ms", lab.latest_gpu_ms),
                                    &format!(
                                        "光流 {:.0}% · 切镜 {} 次",
                                        lab.current_flow_scale() * 100.0,
                                        lab.latest_scene_cuts
                                    ),
                                    false,
                                );
                            });

                            ui.add_space(10.0);
                            clean_sparkline(
                                ui,
                                "静止块跳过比例 (%)",
                                &lab.tile_skip_history,
                                0.0,
                                100.0,
                                "%",
                                68.0,
                            );
                            ui.add_space(8.0);
                            clean_sparkline(
                                ui,
                                "单帧 GPU 计算耗时 (ms)",
                                &lab.gpu_ms_history,
                                0.0,
                                10.0,
                                " ms",
                                68.0,
                            );
                        });
                    });
                };

                if wide {
                    let gap = 12.0;
                    let side_w = (avail_w * 0.30).clamp(260.0, 304.0);
                    let left_card_w = (avail_w - side_w - gap).max(400.0);
                    ui.horizontal_top(|ui| {
                        ui.spacing_mut().item_spacing.x = gap;
                        render_viewport(ui, lab, left_card_w, max_img_h);
                        render_telemetry(ui, lab, side_w);
                    });
                } else {
                    render_viewport(ui, lab, avail_w, max_img_h);
                    ui.add_space(10.0);
                    render_telemetry(ui, lab, avail_w);
                }
            });
    }

    fn render_hud_editor_tab(&mut self, ui: &mut Ui) {
        if self.cfg.profiles.get(self.selected_profile_idx).is_none() {
            return;
        }

        ScrollArea::vertical()
            .id_salt("min_hud_editor_scroll")
            .show(ui, |ui| {
                if let Some(profile) = self.cfg.profiles.get_mut(self.selected_profile_idx) {
                    panel_card(ui, |ui| {
                        section_heading(
                            ui,
                            &format!("固定界面保护框选 · {}", profile.name),
                            "直接用鼠标在下方画面拖拽框出小地图、血条或准星，这些区域将被强制锁定不参与补帧变形（最多 4 个）",
                        );
                        ui.horizontal_wrapped(|ui| {
                            let can_add = profile.hud_hints.len() < 4;
                            for (lbl, rect_v) in [
                                ("+ 左下血条", [0.025_f32, 0.80, 0.245, 0.15]),
                                ("+ 右上小地图", [0.80, 0.04, 0.18, 0.24]),
                                ("+ 中心准星", [0.475, 0.465, 0.05, 0.07]),
                                ("+ 底部字幕", [0.20, 0.85, 0.60, 0.11]),
                            ] {
                                if ghost_button(ui, lbl, false, can_add) {
                                    profile.hud_hints.push(rect_v);
                                    self.dirty = true;
                                }
                            }
                            if danger_ghost_button(ui, "清空全部") {
                                profile.hud_hints.clear();
                                self.dirty = true;
                            }
                            inline_tag(
                                ui,
                                "保护区",
                                &format!("{}/4", profile.hud_hints.len()),
                                !profile.hud_hints.is_empty(),
                            );
                        });
                    });
                }

                ui.add_space(10.0);

                let avail_w = ui.available_width();
                let wide = avail_w >= 820.0;
                let gap = 12.0;

                if wide {
                    let side_w = (avail_w * 0.30).clamp(250.0, 296.0);
                    let left_card_w = (avail_w - side_w - gap).max(380.0);
                    ui.horizontal_top(|ui| {
                        ui.spacing_mut().item_spacing.x = gap;
                        fixed_box(ui, left_card_w, |ui| {
                            self.render_hud_canvas_card(ui);
                        });
                        fixed_box(ui, side_w, |ui| {
                            self.render_hud_list_card(ui);
                        });
                    });
                } else {
                    self.render_hud_canvas_card(ui);
                    ui.add_space(10.0);
                    self.render_hud_list_card(ui);
                }
            });
    }

    fn render_hud_canvas_card(&mut self, ui: &mut Ui) {
        panel_card(ui, |ui| {
            let canvas_w = ui.available_width().max(240.0);
            let canvas_h = canvas_w * 9.0 / 16.0;
            let (rect, response) =
                ui.allocate_exact_size(Vec2::new(canvas_w, canvas_h), Sense::drag());
            let painter = ui.painter();

            if let Some(lab) = &self.live_lab {
                painter.image(
                    lab.texture_id,
                    rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            } else {
                painter.rect_filled(rect, CornerRadius::same(6), Color32::from_rgb(17, 20, 26));
            }

            if response.drag_started() {
                self.hud_drag_start = response.interact_pointer_pos();
            }
            if let (Some(start), Some(curr)) =
                (self.hud_drag_start, response.interact_pointer_pos())
            {
                let dr = Rect::from_two_pos(start, curr).intersect(rect);
                painter.rect_filled(
                    dr,
                    CornerRadius::same(3),
                    Color32::from_rgba_unmultiplied(79, 156, 249, 44),
                );
                painter.rect_stroke(
                    dr,
                    CornerRadius::same(3),
                    Stroke::new(1.5_f32, ACCENT),
                    StrokeKind::Inside,
                );
            }
            if response.drag_stopped() {
                if let (Some(start), Some(end)) =
                    (self.hud_drag_start.take(), response.interact_pointer_pos())
                {
                    let r = Rect::from_two_pos(start, end).intersect(rect);
                    if r.width() > 8.0 && r.height() > 8.0 {
                        let nx = ((r.left() - rect.left()) / rect.width()).clamp(0.0, 1.0);
                        let ny = ((r.top() - rect.top()) / rect.height()).clamp(0.0, 1.0);
                        let nw = (r.width() / rect.width()).clamp(0.01, 1.0 - nx);
                        let nh = (r.height() / rect.height()).clamp(0.01, 1.0 - ny);
                        if let Some(p) = self.cfg.profiles.get_mut(self.selected_profile_idx) {
                            if p.hud_hints.len() < 4 {
                                p.hud_hints.push([nx, ny, nw, nh]);
                                self.dirty = true;
                            }
                        }
                    }
                }
            }

            if let Some(p) = self.cfg.profiles.get(self.selected_profile_idx) {
                for (idx, h) in p.hud_hints.iter().enumerate() {
                    let min = Pos2::new(
                        rect.left() + h[0] * rect.width(),
                        rect.top() + h[1] * rect.height(),
                    );
                    let max = Pos2::new(
                        rect.left() + (h[0] + h[2]) * rect.width(),
                        rect.top() + (h[1] + h[3]) * rect.height(),
                    );
                    let hr = Rect::from_min_max(min, max);
                    painter.rect_filled(
                        hr,
                        CornerRadius::same(3),
                        Color32::from_rgba_unmultiplied(79, 156, 249, 40),
                    );
                    painter.rect_stroke(
                        hr,
                        CornerRadius::same(3),
                        Stroke::new(1.5_f32, ACCENT),
                        StrokeKind::Inside,
                    );
                    painter.text(
                        min + Vec2::new(6.0, 4.0),
                        Align2::LEFT_TOP,
                        format!("保护区 {}", idx + 1),
                        FontId::proportional(11.0),
                        Color32::WHITE,
                    );
                }
            }
        });
    }

    fn render_hud_list_card(&mut self, ui: &mut Ui) {
        panel_card(ui, |ui| {
            section_heading(ui, "已添加的保护区", "坐标按画面比例自动适配不同分辨率");
            let mut del_idx = None;
            if let Some(p) = self.cfg.profiles.get_mut(self.selected_profile_idx) {
                if p.hud_hints.is_empty() {
                    ui.label(RichText::new("暂无保护区，可在左侧画面拖动框选。").color(TEXT_MUTED));
                }
                for (i, h) in p.hud_hints.iter().enumerate() {
                    Frame::NONE
                        .fill(BG_ELEVATED)
                        .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
                        .corner_radius(CornerRadius::same(7))
                        .inner_margin(Margin::same(9))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(format!("保护区 {}", i + 1)).strong());
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui
                                        .small_button(RichText::new("移除").color(SEMANTIC_DANGER))
                                        .clicked()
                                    {
                                        del_idx = Some(i);
                                    }
                                });
                            });
                            ui.label(
                                RichText::new(format!(
                                    "位置 ({:.0}%, {:.0}%) · 大小 {:.0}%×{:.0}%",
                                    h[0] * 100.0,
                                    h[1] * 100.0,
                                    h[2] * 100.0,
                                    h[3] * 100.0
                                ))
                                .size(11.2)
                                .color(TEXT_MUTED),
                            );
                        });
                    ui.add_space(4.0);
                }
                if let Some(d) = del_idx {
                    p.hud_hints.remove(d);
                    self.dirty = true;
                }
            }
        });
    }

    fn render_capture_tab(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        ScrollArea::vertical()
            .id_salt("min_capture_scroll")
            .show(ui, |ui| {
                ui.columns(2, |cols| {
                    cols[0].vertical(|ui| {
                        panel_card(ui, |ui| {
                            section_heading(
                                ui,
                                "游戏窗口捕获设置",
                                "不注入游戏进程，兼容各类反作弊；自动开启防套娃与置顶鼠标穿透",
                            );

                            ui.label(RichText::new("启动时捕获哪个窗口").size(12.2).color(TEXT_SECONDARY));
                            ui.add_space(4.0);
                            segmented_control(
                                ui,
                                "cap_mode_seg",
                                &mut self.target_mode,
                                &[
                                    (TargetSelectMode::ForegroundCountdown, "自动抓前台窗口 (倒计时)"),
                                    (TargetSelectMode::ExplicitWindow, "锁定右侧选中的窗口"),
                                ],
                            );

                            if self.target_mode == TargetSelectMode::ForegroundCountdown {
                                ui.add_space(10.0);
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new("切换窗口倒计时:").size(12.0).color(TEXT_MUTED));
                                    fixed_box(ui, 198.0, |ui| {
                                        if segmented_control(
                                            ui,
                                            "cap_cd_seg",
                                            &mut self.cfg.capture.countdown_secs,
                                            &[(3, "3 秒"), (5, "5 秒"), (8, "8 秒")],
                                        ) {
                                            self.dirty = true;
                                        }
                                    });
                                });
                            }

                            ui.add_space(12.0);
                            ui.label(RichText::new("画面捕获方式").size(12.2).color(TEXT_SECONDARY));
                            ui.add_space(4.0);
                            if segmented_control(
                                ui,
                                "cap_api_seg",
                                &mut self.cfg.capture.api,
                                &[
                                    ("dxgi".into(), "DXGI 桌面复制（推荐）"),
                                    ("window_gdi".into(), "单窗口兼容模式"),
                                ],
                            ) {
                                self.dirty = true;
                            }

                            ui.add_space(12.0);
                            ui.label(RichText::new("画面同步方式").size(12.2).color(TEXT_SECONDARY));
                            ui.add_space(4.0);
                            if segmented_control(
                                ui,
                                "present_seg",
                                &mut self.cfg.capture.present_mode,
                                &[
                                    ("mailbox".into(), "低延迟防撕裂"),
                                    ("fifo".into(), "垂直同步"),
                                    ("immediate".into(), "允许撕裂极限延迟"),
                                ],
                            ) {
                                self.dirty = true;
                            }

                            ui.add_space(14.0);
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new("全局开关快捷键（点击即可实时更换）")
                                        .size(12.2)
                                        .color(TEXT_SECONDARY),
                                );
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if self._hotkey.is_registered() {
                                        status_pill(ui, "热键已生效", SEMANTIC_OK, None);
                                    } else {
                                        status_pill(ui, "热键被占用，请换键", SEMANTIC_WARN, None);
                                    }
                                });
                            });
                            ui.add_space(6.0);

                            let mut hk_changed = false;
                            ui.horizontal_wrapped(|ui| {
                                if ghost_button(ui, "Ctrl", self.cfg.capture.hotkey_ctrl, true) {
                                    // Keep at least one modifier active so normal typing isn't hijacked
                                    if self.cfg.capture.hotkey_alt || !self.cfg.capture.hotkey_ctrl {
                                        self.cfg.capture.hotkey_ctrl = !self.cfg.capture.hotkey_ctrl;
                                        hk_changed = true;
                                    }
                                }
                                if ghost_button(ui, "Alt", self.cfg.capture.hotkey_alt, true) {
                                    if self.cfg.capture.hotkey_ctrl || !self.cfg.capture.hotkey_alt {
                                        self.cfg.capture.hotkey_alt = !self.cfg.capture.hotkey_alt;
                                        hk_changed = true;
                                    }
                                }
                                ui.label(RichText::new("+").color(TEXT_MUTED));
                                for k in ["S", "G", "F", "X", "F8", "F9", "F10"] {
                                    let sel = self.cfg.capture.hotkey_key.eq_ignore_ascii_case(k);
                                    if ghost_button(ui, k, sel, true) && !sel {
                                        self.cfg.capture.hotkey_key = k.into();
                                        hk_changed = true;
                                    }
                                }
                            });
                            if hk_changed {
                                self.dirty = true;
                                self.reload_hotkey(ctx);
                            }
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("当前组合:").size(11.5).color(TEXT_MUTED));
                                if self.cfg.capture.hotkey_ctrl {
                                    keycap_badge(ui, "Ctrl");
                                    ui.label(RichText::new("+").color(TEXT_MUTED));
                                }
                                if self.cfg.capture.hotkey_alt {
                                    keycap_badge(ui, "Alt");
                                    ui.label(RichText::new("+").color(TEXT_MUTED));
                                }
                                keycap_badge(ui, &self.cfg.capture.hotkey_key);
                            });
                        });

                        ui.add_space(12.0);

                        // Active Game Profile Auto-Scale Rules Card
                        panel_card(ui, |ui| {
                            let prof_name = self
                                .cfg
                                .profiles
                                .get(self.selected_profile_idx)
                                .map(|p| p.name.clone())
                                .unwrap_or_else(|| "默认配置".into());
                            section_heading(
                                ui,
                                &format!("配置档自动触发 · {prof_name}"),
                                "切到匹配的游戏窗口时自动启动缩放，无需手动按热键",
                            );
                            if let Some(p) = self.cfg.profiles.get_mut(self.selected_profile_idx) {
                                let is_default = self.selected_profile_idx == 0;
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new("窗口关键词:").size(12.0).color(TEXT_MUTED));
                                    if ui
                                        .add_enabled(
                                            !is_default,
                                            egui::TextEdit::singleline(&mut p.match_pattern)
                                                .desired_width(210.0),
                                        )
                                        .changed()
                                    {
                                        self.dirty = true;
                                    }
                                });
                                if is_default {
                                    ui.label(
                                        RichText::new("默认配置档作用于所有游戏；如需按游戏自动开启，请在左侧「+ 新建」专属配置档。")
                                            .size(11.0)
                                            .color(TEXT_MUTED),
                                    );
                                } else {
                                    ui.add_space(8.0);
                                    if clean_toggle(
                                        ui,
                                        "prof_auto_scale_sw",
                                        &mut p.auto_scale,
                                        "切到匹配窗口时自动开启缩放",
                                        "后台每 0.8 秒轻量检测前台窗口标题，命中关键词即自动接管",
                                    ) {
                                        self.dirty = true;
                                    }
                                }
                            }
                        });
                    });

                    cols[1].vertical(|ui| {
                        panel_card(ui, |ui| {
                            ui.horizontal(|ui| {
                                section_heading(
                                    ui,
                                    "当前打开的窗口",
                                    "点击任意一行即可锁定为目标窗口",
                                );
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ghost_button(ui, "刷新列表", false, true) {
                                        self.visible_windows = list_visible_windows();
                                    }
                                });
                            });

                            ScrollArea::vertical()
                                .id_salt("min_win_scroll")
                                .max_height(460.0)
                                .show(ui, |ui| {
                                    if self.visible_windows.is_empty() {
                                        ui.label(
                                            RichText::new("未检测到其他可见窗口，可点击「刷新列表」重试。")
                                                .size(12.0)
                                                .color(TEXT_MUTED),
                                        );
                                    }
                                    for w in &self.visible_windows {
                                        let is_sel = self.selected_window_title.as_deref()
                                            == Some(w.title.as_str());
                                        let resp = Frame::NONE
                                            .fill(if is_sel {
                                                Color32::from_rgb(31, 46, 68)
                                            } else {
                                                BG_ELEVATED
                                            })
                                            .stroke(Stroke::new(
                                                1.0_f32,
                                                if is_sel { ACCENT } else { BORDER_SUBTLE },
                                            ))
                                            .corner_radius(CornerRadius::same(7))
                                            .inner_margin(Margin::symmetric(12, 8))
                                            .show(ui, |ui| {
                                                ui.set_width(ui.available_width());
                                                ui.horizontal(|ui| {
                                                    ui.label(
                                                        RichText::new(truncate_text(&w.title, 38))
                                                            .strong()
                                                            .size(12.5),
                                                    );
                                                    ui.with_layout(
                                                        Layout::right_to_left(Align::Center),
                                                        |ui| {
                                                            ui.label(
                                                                RichText::new(format!(
                                                                    "{}×{}",
                                                                    w.width, w.height
                                                                ))
                                                                .monospace()
                                                                .size(11.0)
                                                                .color(TEXT_MUTED),
                                                            );
                                                        },
                                                    );
                                                });
                                            })
                                            .response
                                            .interact(Sense::click());
                                        if resp.clicked() {
                                            self.selected_window_title = Some(w.title.clone());
                                            self.target_mode = TargetSelectMode::ExplicitWindow;
                                        }
                                        ui.add_space(4.0);
                                    }
                                });
                        });
                    });
                });
            });
    }

    fn render_diagnostics_tab(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        ScrollArea::vertical()
            .id_salt("min_diag_scroll")
            .show(ui, |ui| {
                // Card 1: Dual-GPU Selector
                panel_card(ui, |ui| {
                    section_heading(
                        ui,
                        "运行显卡选择（支持双卡副卡补帧）",
                        "若电脑有核显或副显卡，可指定副卡专门负责补帧，不占用主游戏卡性能",
                    );
                    ui.horizontal_wrapped(|ui| {
                        let is_auto = self.cfg.gpu.preferred_adapter.trim().is_empty();
                        if ghost_button(ui, "自动选择高性能独显", is_auto, true) {
                            self.cfg.gpu.preferred_adapter.clear();
                            self.dirty = true;
                        }
                        for g in &self.gpu_adapters {
                            let sel = !is_auto
                                && self
                                    .cfg
                                    .gpu
                                    .preferred_adapter
                                    .eq_ignore_ascii_case(&g.name);
                            if ghost_button(
                                ui,
                                &format!("{} ({})", g.name, g.backend),
                                sel,
                                true,
                            ) {
                                self.cfg.gpu.preferred_adapter = g.name.clone();
                                self.dirty = true;
                            }
                        }
                    });
                });

                ui.add_space(12.0);

                // Card 2: Clean RenderGraph VRAM Aliasing Diagram
                panel_card(ui, |ui| {
                    ui.horizontal(|ui| {
                        section_heading(
                            ui,
                            "显存复用原理示意图",
                            "开启显存复用压缩后，不重叠的中间缓存将合并到同一块显存槽中",
                        );
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if clean_toggle(
                                ui,
                                "diag_vram_sw",
                                &mut self.cfg.gpu.enable_vram_aliasing,
                                "显存复用压缩",
                                "",
                            ) {
                                self.dirty = true;
                            }
                        });
                    });
                    render_graph_gantt(ui, self.cfg.gpu.enable_vram_aliasing);
                });

                ui.add_space(12.0);

                // Card 3: Real Hardware Benchmark
                panel_card(ui, |ui| {
                    ui.horizontal(|ui| {
                        section_heading(
                            ui,
                            "显卡实机测速",
                            "在当前显卡上真实跑一遍完整放大+补帧管线，测试极限吞吐与省电效果",
                        );
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let btn = egui::Button::new(
                                RichText::new(if self.bench_running {
                                    "测速中..."
                                } else {
                                    "开始测速"
                                })
                                .strong()
                                .color(Color32::WHITE),
                            )
                            .fill(ACCENT)
                            .corner_radius(CornerRadius::same(7))
                            .min_size(Vec2::new(112.0, 34.0));
                            if ui.add_enabled(!self.bench_running, btn).clicked() {
                                self.start_async_benchmark(ctx.clone());
                            }
                        });
                    });

                    ui.horizontal_top(|ui| {
                        fixed_box(ui, 260.0, |ui| {
                            let mut preset = if self.bench_dst_w == 3840 { 1u32 } else { 0u32 };
                            if segmented_control(
                                ui,
                                "bench_res_seg",
                                &mut preset,
                                &[(0, "1080p → 2K"), (1, "2K → 4K")],
                            ) {
                                if preset == 0 {
                                    self.bench_src_w = 1920;
                                    self.bench_src_h = 1080;
                                    self.bench_dst_w = 2560;
                                    self.bench_dst_h = 1440;
                                } else {
                                    self.bench_src_w = 2560;
                                    self.bench_src_h = 1440;
                                    self.bench_dst_w = 3840;
                                    self.bench_dst_h = 2160;
                                }
                            }
                        });
                        ui.add_space(16.0);
                        fixed_box(ui, 240.0, |ui| {
                            clean_slider_int(
                                ui,
                                "bench_frames_sl",
                                "测试帧数",
                                "",
                                &mut self.bench_frames,
                                20,
                                180,
                                "帧",
                            );
                        });
                    });

                    if let Some(ref r) = self.last_bench {
                        ui.add_space(12.0);
                        ui.label(
                            RichText::new(format!(
                                "{} ({}) · {}×{} → {}×{}",
                                r.gpu_name, r.backend, r.src_w, r.src_h, r.dst_w, r.dst_h
                            ))
                            .size(12.0)
                            .color(TEXT_MUTED),
                        );
                        ui.add_space(8.0);
                        ui.columns(4, |cols| {
                            stat_box(
                                &mut cols[0],
                                "静止块省算力",
                                &format!("{:.1}%", r.tile_skipped_pct),
                                "静态块自动跳过",
                                true,
                            );
                            stat_box(
                                &mut cols[1],
                                "显存占用",
                                &format!("{:.1} MB", r.aliased_vram_mb),
                                &format!("比未复用省 {:.1}%", r.vram_saved_pct),
                                false,
                            );
                            let cap_fps = 1000.0 / r.avg_gpu_ms_per_out_frame.max(0.01);
                            stat_box(
                                &mut cols[2],
                                "单帧耗时",
                                &format!("{:.2} ms", r.avg_gpu_ms_per_out_frame),
                                &format!("理论上限 ~{cap_fps:.0} FPS"),
                                true,
                            );
                            stat_box(
                                &mut cols[3],
                                "瞬切拦截",
                                &format!("{} 次", r.scene_cuts_caught),
                                "切镜防撕裂生效",
                                false,
                            );
                        });
                    }
                });
            });
    }
}

impl eframe::App for FluidScaleApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame_instant).as_secs_f32();
        self.last_frame_instant = now;
        let time = now.duration_since(self.app_start).as_secs_f32();

        let spring_moving = self.nav_spring.step(dt);
        let trans_moving = self.page_transition.step(dt);

        if self._hotkey.take_triggered() {
            self.toggle_scaling(true);
        }

        self.poll_auto_scale_profiles(now);

        if let Some(event) = self.overlay.poll(&self.cfg, &self.cfg_path) {
            match event {
                Ok(msg) => self.set_toast(msg, SEMANTIC_OK),
                Err(e) => self.set_toast(format!("{e}"), SEMANTIC_DANGER),
            }
        }

        if let Some(rx) = &self.bench_rx {
            if let Ok(result) = rx.try_recv() {
                self.bench_running = false;
                self.bench_rx = None;
                match result {
                    Ok(rep) => {
                        self.set_toast(
                            format!(
                                "测速完成：单帧 {:.2} ms · 省算力 {:.1}%",
                                rep.avg_gpu_ms_per_out_frame, rep.tile_skipped_pct
                            ),
                            SEMANTIC_OK,
                        );
                        self.last_bench = Some(rep);
                    }
                    Err(e) => self.set_toast(format!("测速失败: {e}"), SEMANTIC_DANGER),
                }
            }
        }

        let lab_visible = matches!(self.active_tab, NavTab::LiveLab | NavTab::HudEditor);
        if lab_visible {
            if let Some(rs) = frame.wgpu_render_state() {
                if let Some(lab) = self.live_lab.as_mut() {
                    lab.sync_pipeline_if_needed(rs, &self.cfg);
                    let profile = self.cfg.profiles.get(self.selected_profile_idx).cloned();
                    lab.step(&self.cfg, profile.as_ref());
                }
            }
        }

        self.render_top_bar(ctx, time);
        self.render_sidebar(ctx);

        if let Some((msg, color, expires)) = self.toast.clone() {
            if now < expires {
                TopBottomPanel::bottom("min_toast")
                    .exact_height(32.0)
                    .frame(
                        Frame::NONE
                            .fill(BG_ELEVATED)
                            .stroke(Stroke::new(1.0_f32, BORDER_SUBTLE))
                            .inner_margin(Margin::symmetric(18, 6)),
                    )
                    .show(ctx, |ui| {
                        ui.label(RichText::new(msg).size(12.2).color(color));
                    });
            } else {
                self.toast = None;
            }
        }

        let slide_y = self.page_transition.slide_offset_y();
        let reveal_alpha = self.page_transition.opacity();
        egui::CentralPanel::default()
            .frame(Frame::NONE.fill(BG_CANVAS).inner_margin(Margin::same(18)))
            .show(ctx, |ui| {
                ui.set_opacity(reveal_alpha);
                if slide_y.abs() > 0.1 {
                    ui.add_space(slide_y.max(0.0));
                }
                match self.active_tab {
                    NavTab::Scaling => self.render_scaling_tab(ui),
                    NavTab::LiveLab => self.render_live_lab_tab(ui),
                    NavTab::HudEditor => self.render_hud_editor_tab(ui),
                    NavTab::Capture => self.render_capture_tab(ui, ctx),
                    NavTab::Diagnostics => self.render_diagnostics_tab(ui, ctx),
                }
            });

        let has_auto_profiles = self
            .cfg
            .profiles
            .iter()
            .any(|p| p.auto_scale && p.match_pattern != "*");
        let lab_active = lab_visible && self.live_lab.as_ref().is_some_and(|l| !l.paused);
        if spring_moving || trans_moving || lab_active {
            ctx.request_repaint();
        } else if self.overlay.is_counting_down() || self.overlay.is_running() {
            ctx.request_repaint_after(Duration::from_millis(80));
        } else if has_auto_profiles {
            ctx.request_repaint_after(Duration::from_millis(800));
        }
    }
}

fn paint_clean_chip(painter: &egui::Painter, anchor: Pos2, text: &str, align_right: bool) {
    let galley =
        painter.layout_no_wrap(text.to_string(), FontId::proportional(11.2), Color32::WHITE);
    let pad = Vec2::new(9.0, 4.0);
    let size = galley.size() + pad * 2.0;
    let min = if align_right {
        Pos2::new(anchor.x - size.x, anchor.y)
    } else {
        anchor
    };
    let rect = Rect::from_min_size(min, size);
    painter.rect_filled(
        rect,
        CornerRadius::same(5),
        Color32::from_rgba_unmultiplied(16, 18, 22, 215),
    );
    painter.rect_stroke(
        rect,
        CornerRadius::same(5),
        Stroke::new(1.0_f32, BORDER_SUBTLE),
        StrokeKind::Inside,
    );
    painter.galley(rect.min + pad, galley, TEXT_PRIMARY);
}

fn truncate_text(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let collected: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{collected}…")
    } else {
        collected
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn live_lab_layout_never_clips_horizontally_or_vertically() {
        for (win_w, win_h) in [(1240.0_f32, 790.0_f32), (980.0_f32, 640.0_f32)] {
            let ctx = egui::Context::default();
            theme::apply_minimal_theme(&ctx);
            let mut scene_id = 0u32;
            let mut compare_mode = 2u32;
            let mut left_card_rect = Rect::NOTHING;
            let mut right_card_rect = Rect::NOTHING;
            let mut img_rect_out = Rect::NOTHING;
            let mut central_rect = Rect::NOTHING;

            let raw_input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(win_w, win_h))),
                ..Default::default()
            };

            let _ = ctx.run(raw_input, |ctx| {
                TopBottomPanel::top("t").exact_height(60.0).show(ctx, |_| {});
                egui::SidePanel::left("s").exact_width(232.0).show(ctx, |_| {});
                egui::CentralPanel::default()
                    .frame(Frame::NONE.fill(BG_CANVAS).inner_margin(Margin::same(18)))
                    .show(ctx, |ui| {
                        central_rect = ui.max_rect();
                        let avail_w = ui.available_width();
                        let visible_h = ui.available_height();
                        let wide = avail_w >= 820.0;

                        panel_card(ui, |ui| {
                            let inner_w = ui.available_width();
                            if wide {
                                let gap = 12.0;
                                let left_w = ((inner_w - gap) * 0.51).max(260.0);
                                let right_w = (inner_w - gap - left_w).max(240.0);
                                ui.horizontal_top(|ui| {
                                    ui.spacing_mut().item_spacing.x = gap;
                                    fixed_box(ui, left_w, |ui| {
                                        let r = ui.max_rect();
                                        assert!(r.width() <= left_w + 1.0);
                                        segmented_control(
                                            ui,
                                            "test_sc",
                                            &mut scene_id,
                                            &[(0, "A"), (1, "B"), (2, "C"), (3, "D")],
                                        );
                                    });
                                    fixed_box(ui, right_w, |ui| {
                                        segmented_control(
                                            ui,
                                            "test_cm",
                                            &mut compare_mode,
                                            &[(2, "1"), (3, "2"), (1, "3"), (0, "4")],
                                        );
                                    });
                                });
                            }
                        });

                        ui.add_space(10.0);
                        let top_strip_used = if wide { 142.0 } else { 205.0 };
                        let max_img_h = (visible_h - top_strip_used - 74.0).clamp(220.0, 560.0);

                        if wide {
                            let gap = 12.0;
                            let side_w = (avail_w * 0.30).clamp(260.0, 304.0);
                            let left_card_w = (avail_w - side_w - gap).max(400.0);
                            ui.horizontal_top(|ui| {
                                ui.spacing_mut().item_spacing.x = gap;
                                fixed_box(ui, left_card_w, |ui| {
                                    panel_card(ui, |ui| {
                                        let inner_w = ui.available_width();
                                        let aspect = 16.0_f32 / 9.0;
                                        let mut vp_w = inner_w;
                                        let mut vp_h = vp_w / aspect;
                                        if vp_h > max_img_h {
                                            vp_h = max_img_h;
                                            vp_w = (vp_h * aspect).min(inner_w);
                                        }
                                        let (slot, _) = ui.allocate_exact_size(
                                            Vec2::new(inner_w, vp_h),
                                            Sense::hover(),
                                        );
                                        let x_pad = ((inner_w - vp_w) * 0.5).max(0.0);
                                        img_rect_out = Rect::from_min_size(
                                            Pos2::new(slot.left() + x_pad, slot.top()),
                                            Vec2::new(vp_w, vp_h),
                                        );
                                    });
                                    left_card_rect = ui.min_rect();
                                });
                                fixed_box(ui, side_w, |ui| {
                                    panel_card(ui, |ui| {
                                        ui.columns(2, |cols| {
                                            stat_box(&mut cols[0], "A", "50%", "sub", true);
                                            stat_box(&mut cols[1], "B", "20/60", "sub", false);
                                        });
                                    });
                                    right_card_rect = ui.min_rect();
                                });
                            });
                        } else {
                            fixed_box(ui, avail_w, |ui| {
                                panel_card(ui, |ui| {
                                    let inner_w = ui.available_width();
                                    let aspect = 16.0_f32 / 9.0;
                                    let mut vp_w = inner_w;
                                    let mut vp_h = vp_w / aspect;
                                    if vp_h > max_img_h {
                                        vp_h = max_img_h;
                                        vp_w = (vp_h * aspect).min(inner_w);
                                    }
                                    let (slot, _) = ui.allocate_exact_size(
                                        Vec2::new(inner_w, vp_h),
                                        Sense::hover(),
                                    );
                                    let x_pad = ((inner_w - vp_w) * 0.5).max(0.0);
                                    img_rect_out = Rect::from_min_size(
                                        Pos2::new(slot.left() + x_pad, slot.top()),
                                        Vec2::new(vp_w, vp_h),
                                    );
                                });
                                left_card_rect = ui.min_rect();
                            });
                        }
                    });
            });

            println!(
                "Layout test {win_w}x{win_h}: central={central_rect:?} img={img_rect_out:?} left_card={left_card_rect:?} right_card={right_card_rect:?}"
            );
            assert!(
                img_rect_out.right() <= central_rect.right() + 1.5,
                "Video clipped horizontally at {win_w}x{win_h}: img={img_rect_out:?} central={central_rect:?}"
            );
            assert!(
                img_rect_out.bottom() <= central_rect.bottom() + 1.5,
                "Video clipped vertically at {win_w}x{win_h}: img={img_rect_out:?} central={central_rect:?}"
            );
            if right_card_rect != Rect::NOTHING {
                assert!(
                    right_card_rect.right() <= central_rect.right() + 1.5,
                    "Telemetry clipped horizontally at {win_w}x{win_h}: right={right_card_rect:?} central={central_rect:?}"
                );
            }
        }
    }
}
