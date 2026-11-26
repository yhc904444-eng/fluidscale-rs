// Pure GPU 4-Scene Real-Time Showcase Generator for FluidScale-RS
// Zero CPU pixel loop — renders crisp, high-contrast test scenes directly into the input frame texture:
//   Scene 0: High-Speed Highway Racing (3D perspective road, flying streetlights, parallax skyline + speed HUD)
//   Scene 1: FPS Fast Panning + Fine Crosshair (rapid horizontal buildings/neon/drone crossing 1px reticle + radar)
//   Scene 2: RPG Half-Static Town (calm night sky + static dialogue/skill bar -> >55% Tile-Skip savings showcase)
//   Scene 3: Sharpness & Resolution Chart (rotating starburst gear, diagonal gratings, fine grid -> Upscale/CAS test)

struct DemoSceneParams {
    width: u32,
    height: u32,
    scene_id: u32,     // 0=Highway, 1=FPS Pan, 2=RPG Static, 3=Sharpness Target
    scene_cut: u32,    // 1=inject sudden flash cut frame
    time_sec: f32,
    speed: f32,        // 0.5=slow, 1.0=normal, 1.8=fast
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var dst_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: DemoSceneParams;

fn hash21(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    let d = dot(q, q + vec2<f32>(45.32));
    return fract(q.x * q.y + d);
}

fn box_mask(uv: vec2<f32>, bmin: vec2<f32>, bmax: vec2<f32>) -> bool {
    return uv.x >= bmin.x && uv.x <= bmax.x && uv.y >= bmin.y && uv.y <= bmax.y;
}

// =========================================================================
// Scene 0: High-Speed Night Highway Racing (strong motion + translucent HUD)
// =========================================================================
fn render_highway(uv: vec2<f32>, t: f32) -> vec3<f32> {
    let horizon = 0.44;
    var col = vec3<f32>(0.04, 0.06, 0.11);

    if (uv.y < horizon) {
        // Dusk gradient sky + subtle parallax mountain ridge
        let sky_t = uv.y / horizon;
        col = mix(vec3<f32>(0.04, 0.06, 0.12), vec3<f32>(0.16, 0.21, 0.30), sky_t * sky_t);
        let m_x = uv.x * 5.0 + t * 0.25;
        let ridge = horizon - 0.07 - 0.045 * sin(m_x) - 0.025 * cos(m_x * 2.3);
        if (uv.y > ridge) {
            col = vec3<f32>(0.07, 0.09, 0.14);
        }
        // Fast-moving clouds / high-contrast light streaks in upper mid-sky
        let cx = fract(uv.x * 2.4 + t * 0.35);
        if (uv.y > 0.15 && uv.y < 0.21 && cx > 0.2 && cx < 0.55) {
            col = mix(col, vec3<f32>(0.34, 0.40, 0.52), 0.35);
        }
    } else {
        // Perspective 3D road
        let py = max(0.004, uv.y - horizon);
        let z = 0.28 / py; // depth
        let road_center = 0.5 + 0.14 * sin(t * 0.85) * (1.0 - min(py * 2.1, 1.0));
        let half_road = 0.055 + py * 0.72;
        let dx = abs(uv.x - road_center);

        let stripe_phase = fract(z * 1.4 - t * 3.8);
        let alt_band = select(0.0, 1.0, stripe_phase > 0.5);

        if (dx < half_road) {
            // Asphalt with depth-scaled stripes
            col = mix(vec3<f32>(0.10, 0.11, 0.14), vec3<f32>(0.14, 0.16, 0.20), alt_band * 0.6);
            // Center dashed white lane dividers
            if (dx < half_road * 0.032 && stripe_phase > 0.35) {
                col = vec3<f32>(0.92, 0.94, 0.98);
            }
            // Side lane bright markers
            if (abs(dx - half_road * 0.48) < half_road * 0.018 && stripe_phase > 0.45) {
                col = vec3<f32>(0.78, 0.84, 0.92);
            }
            // Racing red/white rumble curb
            if (dx > half_road * 0.91) {
                col = select(vec3<f32>(0.88, 0.90, 0.94), vec3<f32>(0.82, 0.24, 0.28), alt_band > 0.5);
            }
        } else {
            // Ground terrain scrolling quickly past
            col = mix(vec3<f32>(0.08, 0.11, 0.13), vec3<f32>(0.12, 0.16, 0.19), alt_band);
        }

        // Flying high-contrast roadside light poles (intense horizontal+radial optical flow!)
        let pole_z = fract(z * 0.55 - t * 1.5);
        if (pole_z < 0.09 && dx > half_road * 1.02 && dx < half_road * 1.18) {
            col = vec3<f32>(0.95, 0.92, 0.82);
        }
    }

    // Passing oncoming headlight / drone silhouette across middle horizon
    let car_x = fract(t * 0.34);
    let car_dist = length((uv - vec2<f32>(car_x, 0.54)) * vec2<f32>(1.77, 1.0));
    if (car_dist < 0.038) {
        col = vec3<f32>(0.96, 0.86, 0.55);
    }

    // Static translucent bottom-left telemetry HUD + speedometer bar (tests UI Anti-Jello!)
    if (box_mask(uv, vec2<f32>(0.03, 0.80), vec2<f32>(0.27, 0.94))) {
        let border = !box_mask(uv, vec2<f32>(0.033, 0.806), vec2<f32>(0.267, 0.934));
        if (border) {
            col = vec3<f32>(0.82, 0.86, 0.92);
        } else {
            col = col * 0.28 + vec3<f32>(0.05, 0.07, 0.10);
            // Crisp static gauge segments
            if (box_mask(uv, vec2<f32>(0.048, 0.845), vec2<f32>(0.245, 0.875))) {
                let seg = fract((uv.x - 0.048) * 55.0);
                if (seg > 0.22) {
                    col = vec3<f32>(0.32, 0.64, 0.96);
                }
            }
            if (box_mask(uv, vec2<f32>(0.048, 0.895), vec2<f32>(0.185, 0.915))) {
                col = vec3<f32>(0.86, 0.90, 0.96);
            }
        }
    }

    return col;
}

// =========================================================================
// Scene 1: FPS Fast Horizontal Panning + Crisp 1px Crosshair & Radar HUD
// =========================================================================
fn render_fps_pan(uv: vec2<f32>, t: f32) -> vec3<f32> {
    // Fast sinusoidal horizontal camera pan tuned cleanly inside +-36px optical-flow lock envelope
    let pan_x = t * 0.44 + 0.16 * sin(t * 1.35);
    let world_x = uv.x + pan_x;

    // Sky backdrop
    var col = mix(vec3<f32>(0.08, 0.10, 0.15), vec3<f32>(0.16, 0.20, 0.27), uv.y);

    // High-contrast vertical city buildings & bright window grids sweeping right behind crosshair!
    let b_idx = floor(world_x * 6.5);
    let b_local = fract(world_x * 6.5);
    let b_h = 0.22 + 0.42 * hash21(vec2<f32>(b_idx, 1.7));

    if (uv.y > (1.0 - b_h) && b_local > 0.08 && b_local < 0.92) {
        let shade = 0.14 + 0.12 * hash21(vec2<f32>(b_idx, 3.1));
        col = vec3<f32>(shade, shade * 1.08, shade * 1.22);

        // Crisp high-contrast windows (very hard for naive optical flow near static crosshair)
        let wx = fract(b_local * 5.0);
        let wy = fract(uv.y * 18.0);
        if (wx > 0.22 && wx < 0.78 && wy > 0.25 && wy < 0.75) {
            let lit = hash21(vec2<f32>(b_idx * 7.0 + floor(b_local * 5.0), floor(uv.y * 18.0)));
            if (lit > 0.42) {
                col = vec3<f32>(0.88, 0.91, 0.96);
            } else {
                col = vec3<f32>(0.07, 0.09, 0.13);
            }
        }
    }

    // Fast-moving target drone crossing center screen
    let drone_x = fract(t * 0.62);
    let drone_y = 0.48 + 0.12 * sin(t * 3.1);
    let d_drone = length((uv - vec2<f32>(drone_x, drone_y)) * vec2<f32>(1.77, 1.0));
    if (d_drone < 0.045) {
        col = select(vec3<f32>(0.94, 0.36, 0.32), vec3<f32>(0.98, 0.96, 0.92), d_drone < 0.018);
    }

    // Static center FPS tactical reticle (1px fine crosshair + rangefinder ticks)
    let px = uv * vec2<f32>(f32(params.width), f32(params.height));
    let center = vec2<f32>(f32(params.width) * 0.5, f32(params.height) * 0.5);
    let rel = abs(px - center);
    let on_h = rel.y < 1.1 && rel.x >= 5.0 && rel.x <= 26.0;
    let on_v = rel.x < 1.1 && rel.y >= 5.0 && rel.y <= 26.0;
    let on_ring = abs(length(px - center) - 34.0) < 1.1 && (rel.x > 16.0 || rel.y > 16.0);
    if (on_h || on_v || on_ring) {
        col = vec3<f32>(1.0, 1.0, 1.0);
    }

    // Static bottom-left HP/Ammo HUD card
    if (box_mask(uv, vec2<f32>(0.025, 0.81), vec2<f32>(0.26, 0.95))) {
        let border = !box_mask(uv, vec2<f32>(0.028, 0.815), vec2<f32>(0.257, 0.945));
        col = select(col * 0.25 + vec3<f32>(0.06, 0.07, 0.09), vec3<f32>(0.85, 0.88, 0.92), border);
        if (box_mask(uv, vec2<f32>(0.042, 0.855), vec2<f32>(0.215, 0.885))) {
            col = vec3<f32>(0.94, 0.96, 0.99);
        }
        if (box_mask(uv, vec2<f32>(0.042, 0.905), vec2<f32>(0.155, 0.925))) {
            col = vec3<f32>(0.34, 0.65, 0.96);
        }
    }

    // Static top-right circular radar minimap
    let radar_c = vec2<f32>(0.89, 0.16);
    let r_dist = length((uv - radar_c) * vec2<f32>(1.77, 1.0));
    if (r_dist < 0.115) {
        if (r_dist > 0.107) {
            col = vec3<f32>(0.85, 0.88, 0.94);
        } else {
            col = col * 0.22 + vec3<f32>(0.05, 0.07, 0.10);
            if (abs(uv.x - radar_c.x) < 0.002 || abs(uv.y - radar_c.y) < 0.003) {
                col = vec3<f32>(0.35, 0.42, 0.52);
            }
            if (r_dist < 0.012) {
                col = vec3<f32>(0.34, 0.68, 0.98);
            }
        }
    }

    return col;
}

// =========================================================================
// Scene 2: RPG Half-Static Town (55%+ static sky & bottom UI -> Tile-Skip demo!)
// =========================================================================
fn render_rpg_static(uv: vec2<f32>, t: f32) -> vec3<f32> {
    // Upper 42% completely calm night sky + static distant castle silhouette -> 0 GPU optical flow cost!
    if (uv.y < 0.42) {
        let g = uv.y / 0.42;
        var col = mix(vec3<f32>(0.04, 0.05, 0.09), vec3<f32>(0.10, 0.13, 0.20), g);
        // Static moon
        let md = length((uv - vec2<f32>(0.78, 0.16)) * vec2<f32>(1.77, 1.0));
        if (md < 0.055) {
            col = vec3<f32>(0.92, 0.93, 0.96);
        }
        // Static far roof silhouette
        let roof = 0.34 + 0.04 * sin(uv.x * 16.0);
        if (uv.y > roof) {
            col = vec3<f32>(0.06, 0.07, 0.10);
        }
        return col;
    }

    // Bottom 24% completely static RPG dialogue / skill hotbar -> also skipped by Tile-Skip!
    if (uv.y > 0.76) {
        var col = vec3<f32>(0.08, 0.09, 0.12);
        if (abs(uv.y - 0.763) < 0.004) {
            col = vec3<f32>(0.55, 0.60, 0.70);
        }
        // 6 neat static skill slots
        let slot_u = (uv.x - 0.26) / 0.48;
        if (slot_u >= 0.0 && slot_u <= 1.0 && uv.y > 0.80 && uv.y < 0.93) {
            let cell = fract(slot_u * 6.0);
            if (cell > 0.10 && cell < 0.90) {
                col = vec3<f32>(0.14, 0.17, 0.22);
                if (cell < 0.16 || cell > 0.84 || uv.y < 0.812 || uv.y > 0.918) {
                    col = vec3<f32>(0.42, 0.48, 0.58);
                }
            }
        }
        return col;
    }

    // Middle active corridor (0.42..0.76): moving water shimmer + patrolling glowing lantern orbs
    let water_x = fract(uv.x * 10.0 + t * 0.75);
    let wave = 0.5 + 0.5 * sin(uv.x * 38.0 + t * 4.0 + uv.y * 20.0);
    var col = mix(vec3<f32>(0.09, 0.13, 0.18), vec3<f32>(0.16, 0.22, 0.29), wave * 0.6);
    if (water_x > 0.45 && water_x < 0.52) {
        col = col + vec3<f32>(0.08, 0.10, 0.13);
    }

    // Two bright moving characters / lanterns patrolling left & right
    for (var i = 0; i < 2; i = i + 1) {
        let fi = f32(i);
        let ox = fract(t * (0.28 + fi * 0.16) + fi * 0.5);
        let oy = 0.54 + fi * 0.10 + 0.04 * sin(t * 2.5 + fi * 2.0);
        let d = length((uv - vec2<f32>(ox, oy)) * vec2<f32>(1.77, 1.0));
        if (d < 0.042) {
            col = mix(vec3<f32>(0.96, 0.78, 0.42), vec3<f32>(1.0, 0.97, 0.90), 1.0 - d / 0.042);
        }
    }
    return col;
}

// =========================================================================
// Scene 3: Sharpness, Upscaling & Rotating Precision Resolution Target
// =========================================================================
fn render_sharpness_target(uv: vec2<f32>, t: f32) -> vec3<f32> {
    var col = vec3<f32>(0.09, 0.10, 0.13);

    // Subtle moving diagonal line-pair grating on right half (reveals aliasing vs Bicubic+CAS!)
    let diag = sin((uv.x * 110.0 + uv.y * 65.0) - t * 6.0);
    if (uv.x > 0.52 && uv.y > 0.15 && uv.y < 0.85) {
        let v = select(0.14, 0.82, diag > 0.0);
        col = vec3<f32>(v * 0.94, v * 0.96, v);
    }

    // Rotating Siemens starburst resolution target on left-center (reveals motion smoothness + sharpness!)
    let c = vec2<f32>(0.30, 0.50);
    let rel = (uv - c) * vec2<f32>(1.77, 1.0);
    let r = length(rel);
    let ang = atan2(rel.y, rel.x) + t * 0.9;
    if (r < 0.34) {
        let spokes = sin(ang * 18.0);
        let rings = sin(r * 75.0);
        if (r < 0.05) {
            col = vec3<f32>(0.32, 0.64, 0.96);
        } else if (r > 0.315) {
            col = vec3<f32>(0.88, 0.91, 0.96);
        } else {
            let pat = select(0.10, 0.92, spokes > 0.0);
            let ring_mod = select(1.0, 0.72, rings > 0.75);
            col = vec3<f32>(pat * ring_mod);
        }
    }

    // Crisp static calibration crosshairs in corners
    let px = uv * vec2<f32>(f32(params.width), f32(params.height));
    let gx = fract(px.x / 40.0);
    let gy = fract(px.y / 40.0);
    if (uv.y < 0.11 || uv.y > 0.89) {
        col = vec3<f32>(0.07, 0.08, 0.11);
        if (gx < 0.04 || gy < 0.06) {
            col = vec3<f32>(0.24, 0.28, 0.36);
        }
    }

    return col;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
        / vec2<f32>(f32(params.width), f32(params.height));

    var rgb: vec3<f32>;
    if (params.scene_cut != 0u) {
        // Sudden bright high-contrast frame to test Scene-Cut zero-tear passthrough
        rgb = mix(vec3<f32>(0.88, 0.56, 0.24), vec3<f32>(0.22, 0.34, 0.52), uv.x);
    } else {
        let t = params.time_sec * params.speed;
        switch params.scene_id {
            case 0u: { rgb = render_highway(uv, t); }
            case 1u: { rgb = render_fps_pan(uv, t); }
            case 2u: { rgb = render_rpg_static(uv, t); }
            default: { rgb = render_sharpness_target(uv, t); }
        }
    }

    textureStore(dst_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb, 1.0));
}
