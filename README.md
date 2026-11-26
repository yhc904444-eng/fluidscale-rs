# FluidScale (`fluidscale-rs`) v0.1.0

基于 **Rust + eframe/egui + wgpu (DX12 / Vulkan) + WGSL Compute Shader + Win32** 构建的净室原创（Clean-Room）Windows 屏幕空间超分与帧生成桌面应用。

无需注入游戏进程，兼容反作弊；通过 `WDA_EXCLUDEFROMCAPTURE` + 置顶穿透覆盖层对窗口化 / 无边框游戏进行实时画质放大与丝滑补帧。

## 核心架构与四大优化

1. **GPU 算力利用率优化**
   - **16×16 静态 Tile 跳过 + GPU IndirectDispatch**：仅对有变化的区块执行光流估算与插帧合成，天空/菜单等静止区域零光流开销。
   - **两级粗到细层次光流 + 亚像素 Lucas-Kanade**：覆盖 ±36 源像素/帧大位移高速镜头，无拖影跟丢。
   - **动态 FlowScale 闭环反馈**：当单帧耗时逼近帧预算时自动缩放光流分辨率保帧率，负载下降后平滑恢复。
2. **显存占用优化**
   - **RenderGraph 生命周期分析与瞬态显存池复用**：将生命周期不重叠的中间贴图（如 `ui_raw` 与 `synth_tex`）别名映射到同一物理显存槽。
3. **帧节奏平滑度优化**
   - **Huber / 中位数离群值滤波 + 1D Kalman** 过滤着色器编译尖峰。
   - **PI 锁相环（PLL）队列水位微调 + 内容感知重复帧去重** 防止锁帧重复画面破坏光流对。
   - **混合高精度定时器（WaitableTimerEx + 末段自旋）+ 高精度 QPC 采样**，以及瞬切场景自动直通防撕裂。
4. **HUD / 准星防果冻保护**
   - **运动门控时序滞回 UI 检测 + 形态学膨胀禁采区 + 单侧背景 Warp + 顶层清晰 UI 叠加**，支持手动拖拽框选最多 4 个固定 HUD 保护区。

## 桌面端 5 大功能页面

- **缩放与补帧**：一键快捷预设（竞技低延迟 / 3A 丝滑画质 / 核显省电 / 恢复默认）、固定倍率/锁定目标帧率、Bicubic+CAS / Lanczos-2 / 像素风 / 整数无损超分、16:9→16:10 保持原画比例自动黑边、纯着色器左上角帧率显示（小/中/大字号）。
- **实时效果对比**：内置 4 组纯 GPU 实时测试场景（高速夜行公路 / FPS 甩枪准星 / RPG 半静止省电 / 清晰度靶图），支持鼠标拖动左右拉帘实时对比“原生低帧 vs 补帧 / 无保护 vs 防果冻 / 省算力热力透视”。
- **HUD 保护框选**：在实时画面上直接鼠标拖拽框选血条、小地图、准星或字幕保护区，按归一化比例适配任意分辨率。
- **窗口捕获**：支持倒计时（3s / 5s / 8s）自动抓取前台窗口或手动锁定目标窗口；DXGI 桌面复制失败时自动回退单窗口 GDI 兼容模式；支持点击即改的全局快捷键（支持 `Ctrl`/`Alt` + 字母或 `F8`-`F10` 并实时热重载）；支持专属游戏配置档切窗自动触发（`auto_scale`）。
- **显卡与测速**：支持双卡/副卡补帧适配器选择、RenderGraph 显存槽位复用示意图与 1080p→2K / 2K→4K 实机 GPU 测速。

## 直接下载使用（Windows x64 便携版）

从 GitHub Releases 下载 `FluidScale-v0.1.0-windows-x64.zip`，解压到任意英文或中文目录后直接双击运行：

- `FluidScale.exe`：独立桌面图形界面（无控制台黑窗，配置自动保存在同目录 `config.toml`）。
- `fluidscale-rs.exe`：命令行 + 覆盖层工作进程（支持 `benchmark` / `capture` / `demo` / `list-gpus` / `list-windows`）。

> 环境要求：Windows 10 / 11 (x64)，支持 DirectX 12 或 Vulkan 的显卡驱动。

## 从源码编译与测试

```powershell
# 运行全部单元 + 实机 GPU 着色器 + 界面布局测试
cargo test -- --nocapture

# 编译 Release 发布版双产物 (target/release/FluidScale.exe + fluidscale-rs.exe)
cargo build --release --bins

# 命令行运行实机 1080p -> 2K GPU 基准测试
cargo run --release --bin fluidscale-rs -- benchmark --frames 40
```

## 开源协议

基于 [MIT License](LICENSE) 开源。
