use anyhow::{bail, Context, Result};
use image::{io::Reader as ImageReader, DynamicImage, ImageBuffer, Rgb};
use log::{info, warn};
use palette::{FromColor, IntoColor, Lab, LinSrgb, Srgb};
use parking_lot::{Mutex, RwLock};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::f64::consts::E;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
    fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    fn norm(self) -> f64 { self.dot(self).sqrt() }
    fn normalize(self) -> Self { let n = self.norm(); if n == 0.0 { self } else { self.scale(1.0 / n) } }
}

fn color_diff(c1: Vec3, c2: Vec3) -> f64 { c1.sub(c2).norm() }

// Simple power iteration for dominant eigenvector of 3x3 covariance matrix
fn pca_first_component(data: &[Vec3]) -> Result<Vec3> {
    if data.is_empty() { bail!("no data for PCA") }
    let mut mean = Vec3::default();
    for v in data { mean = mean.add(*v); }
    mean = mean.scale(1.0 / data.len() as f64);
    let mut c00 = 0.0; let mut c01 = 0.0; let mut c02 = 0.0;
    let mut c11 = 0.0; let mut c12 = 0.0; let mut c22 = 0.0;
    for v in data {
        let d = v.sub(mean);
        c00 += d.x * d.x; c01 += d.x * d.y; c02 += d.x * d.z;
        c11 += d.y * d.y; c12 += d.y * d.z; c22 += d.z * d.z;
    }
    let n = data.len() as f64;
    c00 /= n; c01 /= n; c02 /= n; c11 /= n; c12 /= n; c22 /= n;
    let mut v = Vec3 { x: 1.0, y: 1.0, z: 1.0 }.normalize();
    for _ in 0..64 {
        let nv = Vec3 {
            x: c00 * v.x + c01 * v.y + c02 * v.z,
            y: c01 * v.x + c11 * v.y + c12 * v.z,
            z: c02 * v.x + c12 * v.y + c22 * v.z,
        };
        let nn = nv.norm(); if nn == 0.0 { break; }
        let v_next = nv.scale(1.0 / nn);
        if (v_next.sub(v)).norm() < 1e-12 { v = v_next; break; }
        v = v_next;
    }
    Ok(v)
}

#[derive(Debug, Clone)]
pub struct Config {
    pub stride_x: usize,
    pub stride_y: usize,
    pub t_final: f64,
    pub alpha: f64,
    pub epsilon_palette: f64,
    pub stag_eps: f64,
    pub stag_limit: usize,
    pub num_threads: usize,
    pub iter_timings: bool,
}

pub fn default_config(fast: bool) -> Config {
    if fast {
        Config { stride_x: 2, stride_y: 2, t_final: 2.0, alpha: 0.6, epsilon_palette: 2.0, stag_eps: 1e-4, stag_limit: 3, num_threads: num_cpus::get().max(1), iter_timings: false }
    } else {
        Config { stride_x: 1, stride_y: 1, t_final: 1.0, alpha: 0.7, epsilon_palette: 1.0, stag_eps: 1e-6, stag_limit: 5, num_threads: num_cpus::get().max(1), iter_timings: false }
    }
}

#[derive(Debug, Clone)]
pub struct Params {
    pub in_image_name: String,
    pub out_image_name: String,
    pub w_out: usize,
    pub h_out: usize,
    pub k_max: usize,
    pub config: Config,
}

#[derive(Clone)]
struct Coords {
    n: Arc<Mutex<usize>>,
    m: usize,
    w_in: usize,
    h_in: usize,
    w_out: usize,
    stride_x: usize,
    stride_y: usize,
}
impl Coords {
    fn new(w_in: usize, h_in: usize, w_out: usize, stride_x: usize, stride_y: usize) -> Self {
        let nx = (w_in + stride_x - 1) / stride_x;
        let ny = (h_in + stride_y - 1) / stride_y;
        let m = nx * ny;
        Self { n: Arc::new(Mutex::new(0)), m, w_in, h_in, w_out, stride_x, stride_y }
    }
    fn next(&self) -> Option<(usize, usize)> {
        let mut guard = self.n.lock();
        if *guard >= self.m { return None; }
        let idx = *guard; *guard += 1;
        let nx = (self.w_in + self.stride_x - 1) / self.stride_x;
        let sx = idx % nx;
        let sy = idx / nx;
        Some((sx * self.stride_x, sy * self.stride_y))
    }
}

#[derive(Clone)]
struct ColorEntry { color: Vec3, probability: f64 }
impl ColorEntry {
    fn condit_prob(&self, sp_color: Vec3, t: f64) -> f64 { self.probability * (-(color_diff(sp_color, self.color) / t)).exp() }
    fn perturb(&mut self, delta: Vec3) { self.color = self.color.add(delta); }
}

#[derive(Default)]
struct Accum { count: usize, sum_x: f64, sum_y: f64, sum_l: f64, sum_a: f64, sum_b: f64 }

#[derive(Clone)]
struct SuperPixel {
    x: f64,
    y: f64,
    palette_color: Vec3,
    p_s: f64,
    accum: Arc<Mutex<Accum>>,
    p_c: Vec<f64>,
    sp_color: Vec3,
    original_xy: (f64, f64),
    original_color: Vec3,
}

impl SuperPixel {
    fn new(x: f64, y: f64, c: Vec3, p_s: f64, original_color: Vec3) -> Self {
        Self { x, y, palette_color: c, p_s, accum: Arc::new(Mutex::new(Accum::default())), p_c: vec![0.5, 0.5], sp_color: Vec3::default(), original_xy: (x, y), original_color }
    }
    fn cost(&self, x0: usize, y0: usize, in_image: &[Vec<Vec3>], n: usize, m: usize) -> f64 {
        let in_color = in_image[x0][y0];
        let c_diff = color_diff(in_color, self.palette_color);
        let dx = self.x - x0 as f64; let dy = self.y - y0 as f64;
        c_diff + 45.0 * ((n as f64 / m as f64).sqrt()) * (dx * dx + dy * dy).sqrt()
    }
    fn add_pixel(&self, x0: usize, y0: usize, in_image: &[Vec<Vec3>]) {
        let mut a = self.accum.lock();
        a.count += 1;
        a.sum_x += x0 as f64; a.sum_y += y0 as f64;
        let p = in_image[x0][y0]; a.sum_l += p.x; a.sum_a += p.y; a.sum_b += p.z;
    }
    fn clear_pixels(&self) { *self.accum.lock() = Accum::default(); }
    fn normalize_probs(&mut self, palette: &[ColorEntry]) {
        let mut denom: f64 = self.p_c.iter().copied().sum(); if denom == 0.0 { denom = 1.0; }
        if let Some((imax, _)) = self.p_c.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal)) { self.palette_color = palette[imax].color; }
        for v in &mut self.p_c { *v /= denom; }
    }
    fn update_pos(&mut self) {
        let a = self.accum.lock();
        if a.count == 0 { self.x = self.original_xy.0; self.y = self.original_xy.1; return; }
        let n = a.count as f64; self.x = a.sum_x / n; self.y = a.sum_y / n;
    }
    fn update_sp_color(&mut self, _in_image: &[Vec<Vec3>]) {
        let a = self.accum.lock();
        if a.count == 0 { self.sp_color = self.original_color; return; }
        let n = a.count as f64; self.sp_color = Vec3 { x: a.sum_l / n, y: a.sum_a / n, z: a.sum_b / n };
    }
}

fn in_bounds(r: isize, c: isize, w_out: usize, h_out: usize) -> bool { r >= 0 && c >= 0 && (r as usize) < w_out && (c as usize) < h_out }

fn sp_refine(super_pixels: &Vec<Vec<Arc<RwLock<SuperPixel>>>>, in_image: &Vec<Vec<Vec3>>, num_threads: usize, w_in: usize, h_in: usize, w_out: usize, h_out: usize, stride_x: usize, stride_y: usize, m_full: usize) {
    let n = w_out * h_out;

    // Compute the sampling grid size (matches Coords::new logic)
    let nx = (w_in + stride_x - 1) / stride_x;
    let ny = (h_in + stride_y - 1) / stride_y;
    let m_samp = nx * ny;

    // Per-thread accumulators to avoid hot mutex contention
    #[derive(Default, Clone, Copy)]
    struct LocalAccum { count: usize, sum_x: f64, sum_y: f64, sum_l: f64, sum_a: f64, sum_b: f64 }

    let sp_arc = Arc::new(super_pixels.clone());
    let in_arc = Arc::new(in_image.clone());

    // Parallel assignment into thread-local accumulators (no locks on hot path)
    let mut handles = Vec::new();
    for t in 0..num_threads {
        let sp_c = sp_arc.clone();
        let in_c = in_arc.clone();
        let stride_x_c = stride_x; let stride_y_c = stride_y;
        let start = t * ((m_samp + num_threads - 1) / num_threads);
        let end = ((t + 1) * ((m_samp + num_threads - 1) / num_threads)).min(m_samp);
        let w_in_c = w_in; let h_in_c = h_in; let w_out_c = w_out; let h_out_c = h_out;
        let n_c = n; let m_full_c = m_full;
        let handle = thread::spawn(move || {
            let mut acc: Vec<Vec<LocalAccum>> = vec![vec![LocalAccum::default(); h_out_c]; w_out_c];
            let dx = [-1, -1, -1, 0, 0, 0, 1, 1, 1];
            let dy = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
            for s in start..end {
                let sx = s % nx; let sy = s / nx;
                let x = sx * stride_x_c; let y = sy * stride_y_c;
                let r = (x * w_out_c) / w_in_c; let c = (y * h_out_c) / h_in_c;
                let mut best_cost = f64::INFINITY; let mut best_pair = (r as isize, c as isize);
                for i in 0..9 {
                    let rr = r as isize + dx[i]; let cc = c as isize + dy[i];
                    if !in_bounds(rr, cc, w_out_c, h_out_c) { continue; }
                    let sp_r = sp_c[rr as usize][cc as usize].read();
                    let cost = sp_r.cost(x, y, &in_c, n_c, m_full_c);
                    if cost < best_cost { best_cost = cost; best_pair = (rr, cc); }
                }
                let rr = best_pair.0 as usize; let cc = best_pair.1 as usize;
                let p = in_c[x][y];
                let a = &mut acc[rr][cc];
                a.count += 1;
                a.sum_x += x as f64; a.sum_y += y as f64;
                a.sum_l += p.x; a.sum_a += p.y; a.sum_b += p.z;
            }
            acc
        });
        handles.push(handle);
    }
    let mut thread_accums: Vec<Vec<Vec<LocalAccum>>> = Vec::with_capacity(num_threads);
    for h in handles { thread_accums.push(h.join().unwrap_or_else(|_| vec![vec![LocalAccum::default(); h_out]; w_out])); }

    // Merge all thread-local accumulators once per cell
    for r in 0..w_out { for c in 0..h_out {
        let mut total = LocalAccum::default();
        for t in 0..num_threads { let a = thread_accums[t][r][c]; total.count += a.count; total.sum_x += a.sum_x; total.sum_y += a.sum_y; total.sum_l += a.sum_l; total.sum_a += a.sum_a; total.sum_b += a.sum_b; }
        let mut sp = super_pixels[r][c].write();
        { let mut acc = sp.accum.lock(); acc.count = total.count; acc.sum_x = total.sum_x; acc.sum_y = total.sum_y; acc.sum_l = total.sum_l; acc.sum_a = total.sum_a; acc.sum_b = total.sum_b; }
    }}

    // Update pos and color
    for r in 0..w_out { for c in 0..h_out { let mut sp = super_pixels[r][c].write(); sp.update_pos(); sp.update_sp_color(in_image); }}

    // Laplacian smoothing
    let mut new_coords = vec![vec![(0.0f64, 0.0f64); h_out]; w_out];
    for r in 0..w_out { for c in 0..h_out {
        let dx = [0, 0, -1, 1]; let dy = [1, -1, 0, 0]; let mut nnb = 0.0; let mut nx = 0.0; let mut ny = 0.0;
        for i in 0..4 { let rr = r as isize + dx[i]; let cc = c as isize + dy[i]; if in_bounds(rr, cc, w_out, h_out) { nnb += 1.0; let sp = super_pixels[rr as usize][cc as usize].read(); nx += sp.x; ny += sp.y; }}
        nx /= nnb; ny /= nnb; let sp = super_pixels[r][c].read(); new_coords[r][c] = (0.4 * nx + 0.6 * sp.x, 0.4 * ny + 0.6 * sp.y);
    }}

    // Bilateral-like smoothing on colors
    let mut new_colors = vec![vec![Vec3::default(); h_out]; w_out];
    for r in 0..w_out { for c in 0..h_out {
        let sp = super_pixels[r][c].read(); let dx = [-1, -1, -1, 0, 0, 1, 1, 1]; let dy = [-1, 0, 1, -1, 1, -1, 0, 1];
        let mut n = 0.0; let mut avg = Vec3::default();
        for i in 0..8 { let rr = r as isize + dx[i]; let cc = c as isize + dy[i]; if in_bounds(rr, cc, w_out, h_out) { let next = super_pixels[rr as usize][cc as usize].read().sp_color; let weight = (-(sp.sp_color.x - next.x).abs()).exp(); avg = avg.add(next.scale(weight)); n += weight; }}
        if n == 0.0 { new_colors[r][c] = sp.sp_color; } else { avg = avg.scale(1.0 / n); new_colors[r][c] = sp.sp_color.scale(0.5).add(avg.scale(0.5)); }
    }}
    for r in 0..w_out { for c in 0..h_out { let mut sp = super_pixels[r][c].write(); let (nx, ny) = new_coords[r][c]; sp.x = nx; sp.y = ny; sp.sp_color = new_colors[r][c]; }}
}

fn associate(super_pixels: &Vec<Vec<Arc<RwLock<SuperPixel>>>>, palette: &mut Vec<ColorEntry>, _clusters: &Vec<(usize, usize)>, t: f64) {
    for row in super_pixels.iter() { for sp in row.iter() { let mut spw = sp.write(); spw.p_c = vec![0.0; palette.len()]; for k in 0..palette.len() { spw.p_c[k] = palette[k].condit_prob(spw.sp_color, t); } spw.normalize_probs(&palette); }}
    for k in 0..palette.len() { palette[k].probability = 0.0; }
    for row in super_pixels.iter() { for sp in row.iter() { let sp = sp.read(); for k in 0..palette.len() { palette[k].probability += sp.p_c[k] * sp.p_s; } }}
}

fn palette_refine(super_pixels: &Vec<Vec<Arc<RwLock<SuperPixel>>>>, palette: &mut Vec<ColorEntry>) -> f64 {
    let mut total_change = 0.0; const EPS_PROB: f64 = 1e-12;
    for k in 0..palette.len() {
        let mut newc = Vec3::default(); let prob = palette[k].probability; if prob <= EPS_PROB { continue; }
        for row in super_pixels.iter() { for sp in row.iter() { let sp = sp.read(); newc = newc.add(sp.sp_color.scale((sp.p_c[k] * sp.p_s) / prob)); }}
        let old = palette[k].color; palette[k].color = newc; total_change += color_diff(old, newc);
    }
    total_change
}

fn expand(clusters: &mut Vec<(usize, usize)>, palette: &mut Vec<ColorEntry>, epsilon_cluster: f64, k: &mut usize, k_max: usize, delta: Vec3) {
    let mut i = 0; while i < *k && *k < k_max {
        let (i0, i1) = clusters[i]; let (c1, c2) = (palette[i0].color, palette[i1].color);
        if color_diff(c1, c2) > epsilon_cluster { *k += 1; let p0 = palette[i0].probability / 2.0; let p1 = palette[i1].probability / 2.0; palette[i0].probability = p0; palette[i1].probability = p1; palette.push(ColorEntry { color: c1, probability: p0 }); palette.push(ColorEntry { color: c2, probability: p1 }); let len = palette.len(); clusters.push((i1, len - 1)); clusters[i] = (i0, len - 2); }
        i += 1;
    }
    if *k >= k_max {
        let mut new_palette = Vec::new(); let mut new_clusters = Vec::new(); for idx in 0..*k { let (i0, i1) = clusters[idx]; let c0 = palette[i0].color; let p0 = palette[i0].probability; let c1 = palette[i1].color; let p1 = palette[i1].probability; let mix = Vec3 { x: (c0.x + c1.x) / 2.0, y: (c0.y + c1.y) / 2.0, z: (c0.z + c1.z) / 2.0 }; new_palette.push(ColorEntry { color: mix, probability: p0 + p1 }); new_clusters.push((idx, idx)); }
        *palette = new_palette; *clusters = new_clusters;
    } else { for i in 0..*k { let (_, i1) = clusters[i]; let c = &mut palette[i1]; c.perturb(delta); }}
}

fn saturate(out_lab: &mut [Vec<Vec3>], w_out: usize, h_out: usize) { for r in 0..w_out { for c in 0..h_out { out_lab[r][c].y *= 1.1; out_lab[r][c].z *= 1.1; }}}

fn to_lab_image(rgb: &image::DynamicImage) -> Result<(Vec<Vec<Vec3>>, usize, usize)> {
    let rgb = rgb.to_rgb8(); let (w, h) = (rgb.width() as usize, rgb.height() as usize); let mut out = vec![vec![Vec3::default(); h]; w];
    for x in 0..w { for y in 0..h { let p = rgb.get_pixel(x as u32, y as u32); let srgb: Srgb<f32> = Srgb::new(p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0); let lin: LinSrgb<f32> = srgb.into_linear(); let lab: Lab = Lab::from_color(lin); out[x][y] = Vec3 { x: lab.l as f64, y: lab.a as f64, z: lab.b as f64 }; }}
    Ok((out, w, h))
}

fn lab_to_rgb_image(lab: &[Vec<Vec3>], w: usize, h: usize) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w as u32, h as u32);
    for x in 0..w { for y in 0..h { let v = lab[x][y]; let lab = Lab::new(v.x as f32, v.y as f32, v.z as f32); let lin: LinSrgb<f32> = lab.into_color(); let srgb: Srgb<f32> = Srgb::from_linear(lin); let r = (srgb.red.clamp(0.0, 1.0) * 255.0).round() as u8; let g = (srgb.green.clamp(0.0, 1.0) * 255.0).round() as u8; let b = (srgb.blue.clamp(0.0, 1.0) * 255.0).round() as u8; img.put_pixel(x as u32, y as u32, Rgb([r, g, b])); }}
    img
}

fn tmp_progress_path(out_image_name: &str, idx: usize) -> String {
    let p = Path::new(out_image_name); let parent: PathBuf = p.parent().unwrap_or_else(|| Path::new("")).to_path_buf(); let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output"); let filename = format!("{}_{}.png", stem, idx); parent.join(filename).to_string_lossy().into_owned()
}

/// Core algorithm operating on a provided image in memory and returning the output image buffer.
pub fn process_dynamic(dyn_img: &DynamicImage, w_out: usize, h_out: usize, k_max: usize, config: Config) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    info!("Starting Pixel-convert (Rust) in-memory"); info!("Output dimensions: {}x{}", w_out, h_out); info!("Max clusters (K_max): {}", k_max);

    let (in_lab, w_in, h_in) = to_lab_image(dyn_img)?;
    let mut pca_data = Vec::with_capacity(w_in * h_in); for x in 0..w_in { for y in 0..h_in { pca_data.push(in_lab[x][y]); }}
    let pc1 = pca_first_component(&pca_data)?; let delta = pc1.scale(1.5); info!("PCA first component: [{:.3}, {:.3}, {:.3}]", pc1.x, pc1.y, pc1.z);

    let mut t = 35.0f64; let t_final = config.t_final; let mut k = 1usize; let alpha = config.alpha; let epsilon_palette = config.epsilon_palette; let epsilon_cluster = 0.25f64; let num_threads = config.num_threads; let m_full = w_in * h_in; let n = w_out * h_out;

    let avg_color = { let mut s = Vec3::default(); for v in &pca_data { s = s.add(*v); } s.scale(1.0 / (pca_data.len() as f64)) };
    let xs: Vec<usize> = (0..w_out).map(|r| (r * w_in) / w_out).collect(); let ys: Vec<usize> = (0..h_out).map(|c| (c * h_in) / h_out).collect();
    let mut super_pixels: Vec<Vec<Arc<RwLock<SuperPixel>>>> = vec![vec![]; w_out];
    for r in 0..w_out { for c in 0..h_out { let xi = xs[r]; let yi = ys[c]; let sp = SuperPixel::new(xi as f64, yi as f64, avg_color, 1.0 / (n as f64), in_lab[xi][yi]); super_pixels[r].push(Arc::new(RwLock::new(sp))); }}

    let mut palette = vec![ ColorEntry { color: avg_color, probability: 0.5 }, ColorEntry { color: avg_color.add(delta), probability: 0.5 } ]; let mut clusters: Vec<(usize, usize)> = vec![(0, 1)];

    let mut iterations = 0usize; let mut stagnant = 0usize;
    while t > t_final {
        let iter_start = if config.iter_timings { Some(Instant::now()) } else { None };
        iterations += 1; info!("K={}, T={:.3}, iter={}", k, t, iterations);
        sp_refine(&super_pixels, &in_lab, num_threads, w_in, h_in, w_out, h_out, config.stride_x, config.stride_y, m_full);
        associate(&super_pixels, &mut palette, &clusters, t);
        let total_change = palette_refine(&super_pixels, &mut palette);
        if total_change < epsilon_palette { t *= alpha; if k < k_max { expand(&mut clusters, &mut palette, epsilon_cluster, &mut k, k_max, delta); }}
        if total_change < config.stag_eps { stagnant += 1; } else { stagnant = 0; }
        if stagnant >= config.stag_limit { info!("Early stop: converged (stagnation)"); break; }
        if iterations >= 1010 { warn!("Max iterations reached"); break; }
        if let Some(start) = iter_start { info!("iter_time_ms={}", start.elapsed().as_millis()); }
    }

    let mut out_lab = vec![vec![Vec3::default(); h_out]; w_out]; for r in 0..w_out { for c in 0..h_out { out_lab[r][c] = super_pixels[r][c].read().palette_color; }} saturate(&mut out_lab, w_out, h_out); let out_img = lab_to_rgb_image(&out_lab, w_out, h_out);
    Ok(out_img)
}

pub fn process(params: Params) -> Result<()> {
    let Params { in_image_name, out_image_name, w_out, h_out, k_max, config } = params;
    info!("Starting Pixel-convert (Rust)"); info!("Input image: {}", in_image_name); info!("Output image: {}", out_image_name); info!("Output dimensions: {}x{}", w_out, h_out); info!("Max clusters (K_max): {}", k_max);

    let dyn_img = ImageReader::open(&in_image_name)?.decode()?; let (in_lab, w_in, h_in) = to_lab_image(&dyn_img)?;
    let mut pca_data = Vec::with_capacity(w_in * h_in); for x in 0..w_in { for y in 0..h_in { pca_data.push(in_lab[x][y]); }}
    let pc1 = pca_first_component(&pca_data)?; let delta = pc1.scale(1.5); info!("PCA first component: [{:.3}, {:.3}, {:.3}]", pc1.x, pc1.y, pc1.z);

    let mut t = 35.0f64; let t_final = config.t_final; let mut k = 1usize; let alpha = config.alpha; let epsilon_palette = config.epsilon_palette; let epsilon_cluster = 0.25f64; let num_threads = config.num_threads; let m_full = w_in * h_in; let n = w_out * h_out;

    let avg_color = { let mut s = Vec3::default(); for v in &pca_data { s = s.add(*v); } s.scale(1.0 / (pca_data.len() as f64)) };
    let xs: Vec<usize> = (0..w_out).map(|r| (r * w_in) / w_out).collect(); let ys: Vec<usize> = (0..h_out).map(|c| (c * h_in) / h_out).collect();
    let mut super_pixels: Vec<Vec<Arc<RwLock<SuperPixel>>>> = vec![vec![]; w_out];
    for r in 0..w_out { for c in 0..h_out { let xi = xs[r]; let yi = ys[c]; let sp = SuperPixel::new(xi as f64, yi as f64, avg_color, 1.0 / (n as f64), in_lab[xi][yi]); super_pixels[r].push(Arc::new(RwLock::new(sp))); }}

    let mut palette = vec![ ColorEntry { color: avg_color, probability: 0.5 }, ColorEntry { color: avg_color.add(delta), probability: 0.5 } ]; let mut clusters: Vec<(usize, usize)> = vec![(0, 1)];

    let mut iterations = 0usize; let mut stagnant = 0usize;
    while t > t_final {
        let iter_start = if config.iter_timings { Some(Instant::now()) } else { None };
        iterations += 1; info!("K={}, T={:.3}, iter={}", k, t, iterations);
        sp_refine(&super_pixels, &in_lab, num_threads, w_in, h_in, w_out, h_out, config.stride_x, config.stride_y, m_full);
        associate(&super_pixels, &mut palette, &clusters, t);
        let total_change = palette_refine(&super_pixels, &mut palette);
        if total_change < epsilon_palette { t *= alpha; if k < k_max { expand(&mut clusters, &mut palette, epsilon_cluster, &mut k, k_max, delta); }}
        if total_change < config.stag_eps { stagnant += 1; } else { stagnant = 0; }
        if stagnant >= config.stag_limit { info!("Early stop: converged (stagnation)"); break; }
        if iterations >= 1010 { warn!("Max iterations reached"); break; }
        if iterations % 100 == 0 && iterations > 1 { let mut out_lab = vec![vec![Vec3::default(); h_out]; w_out]; for r in 0..w_out { for c in 0..h_out { out_lab[r][c] = super_pixels[r][c].read().palette_color; }} saturate(&mut out_lab, w_out, h_out); let out_img = lab_to_rgb_image(&out_lab, w_out, h_out); let tmp_path = tmp_progress_path(&out_image_name, iterations/100); out_img.save(&tmp_path)?; info!("Intermediate output saved: {}", tmp_path); }
        if let Some(start) = iter_start { info!("iter_time_ms={}", start.elapsed().as_millis()); }
    }

    let mut out_lab = vec![vec![Vec3::default(); h_out]; w_out]; for r in 0..w_out { for c in 0..h_out { out_lab[r][c] = super_pixels[r][c].read().palette_color; }} saturate(&mut out_lab, w_out, h_out); let out_img = lab_to_rgb_image(&out_lab, w_out, h_out); out_img.save(out_image_name)?; info!("Final output saved");
    Ok(())
}

// Python bindings live in a separate module to keep lib.rs focused.
// Python bindings are moved to a separate crate to avoid linking issues during `cargo test`.
