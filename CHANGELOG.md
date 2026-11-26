# Changelog

## v0.1.0 (2026-11-26)

### 首次完整发布
- **GPU 计算与补帧管线**：实现 8 个原创 WGSL 着色器（16×16 静态 Tile 跳过与间接分发、特征金字塔、两级层次光流 + 亚像素 Lucas-Kanade、运动门控时序滞回 UI 检测、形态学膨胀、防果冻单侧插帧合成、Bicubic+CAS / Lanczos-2 / SharpBilinear / Nearest 超分 + 内联位图 FPS 计数、等比黑边/拉伸呈现）。
- **RenderGraph 显存池**：按 Pass 生命周期自动复用瞬态显存槽并统计节省比例。
- **帧节奏引擎**：Huber/中位数 + 1D Kalman 滤波、PI-PLL 水位调节、512 点前置内容去重、混合高精度定时器与瞬切防撕裂。
- **Windows 捕获与覆盖层**：多适配器 DXGI 桌面复制（含 `DXGI_ERROR_ACCESS_LOST` 自动恢复）+ `WindowGdiCapturer` 自动兜底、`WDA_EXCLUDEFROMCAPTURE` 防套娃、`WS_EX_LAYERED` (`LWA_ALPHA=255`) 置顶穿透覆盖层、Per-Monitor V2 DPI 物理像素对齐与目标窗口退出自动清理。
- **现代极简桌面 GUI (`FluidScale.exe`)**：5 大页面、一键快捷预设、4 场景纯 GPU 实时左右拉帘对比实验室、鼠标拖拽 HUD 保护框选、全局热键实时自定义与热重载、游戏配置档切窗自动触发 (`auto_scale`) 与实机 GPU 测速。
- **质量验证**：18 项单元测试 + 实机 GPU 着色器精度测试 + 无头 egui 布局零裁切验证全部通过。
