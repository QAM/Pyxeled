use anyhow::{bail, Context, Result};
use image::{io::Reader as ImageReader, ImageBuffer, Rgb};
use log::{info, warn};
use palette::{Lab, Srgb, LinSrgb, FromColor, IntoColor};
use parking_lot::{Mutex, RwLock};
use num_cpus;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::f64::consts::E;
use std::io::{self, Read};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn from_slice(s: &[f64; 3]) -> Self {
        Self { x: s[0], y: s[1], z: s[2] }
    }
    fn to_arr(self) -> [f64; 3] { [self.x, self.y, self.z] }
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
    // center
    let mut mean = Vec3::default();
    for v in data { mean = mean.add(*v); }
    mean = mean.scale(1.0 / data.len() as f64);
    // covariance matrix (symmetric)
    let mut c00 = 0.0; let mut c01 = 0.0; let mut c02 = 0.0;
    let mut c11 = 0.0; let mut c12 = 0.0; let mut c22 = 0.0;
    for v in data {
        let d = v.sub(mean);
        c00 += d.x * d.x;
        c01 += d.x * d.y;
        c02 += d.x * d.z;
        c11 += d.y * d.y;
        c12 += d.y * d.z;
        c22 += d.z * d.z;
    }
    let n = data.len() as f64;
    c00 /= n; c01 /= n; c02 /= n; c11 /= n; c12 /= n; c22 /= n;
    // power iteration
    let mut v = Vec3 { x: 1.0, y: 1.0, z: 1.0 }.normalize();
    for _ in 0..64 {
        let nv = Vec3 {
            x: c00 * v.x + c01 * v.y + c02 * v.z,
            y: c01 * v.x + c11 * v.y + c12 * v.z,
            z: c02 * v.x + c12 * v.y + c22 * v.z,
        };
        let nn = nv.norm();
        if nn == 0.0 { break; }
        let v_next = nv.scale(1.0 / nn);
        if (v_next.sub(v)).norm() < 1e-12 { v = v_next; break; }
        v = v_next;
    }
    Ok(v)
}

#[derive(Clone)]
struct Coords { n: Arc<Mutex<usize>>, m: usize, w_in: usize, w_out: usize }
impl Coords {
    fn new(m: usize, w_in: usize, w_out: usize) -> Self { Self { n: Arc::new(Mutex::new(0)), m, w_in, w_out } }
    fn next(&self) -> Option<(usize, usize)> {
        let mut guard = self.n.lock();
        if *guard >= self.m { return None; }
        let idx = *guard; *guard += 1;
        Some((idx % self.w_in, idx / self.w_in))
    }
    fn reset(&self) { *self.n.lock() = 0; }
}

#[derive(Clone)]
struct ColorEntry { color: Vec3, probability: f64 }

impl ColorEntry {
    fn condit_prob(&self, sp_color: Vec3, t: f64) -> f64 { self.probability * (-(color_diff(sp_color, self.color) / t)).exp() }
    fn perturb(&mut self, delta: Vec3) { self.color = self.color.add(delta); }
}

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
        Self {
            x, y,
            palette_color: c,
            p_s,
            accum: Arc::new(Mutex::new(Accum::default())),
            p_c: vec![0.5, 0.5],
            sp_color: Vec3::default(),
            original_xy: (x, y),
            original_color,
        }
    }

    fn cost(&self, x0: usize, y0: usize, in_image: &[Vec<Vec3>], n: usize, m: usize) -> f64 {
        let in_color = in_image[x0][y0];
        let c_diff = color_diff(in_color, self.palette_color);
        let dx = self.x - x0 as f64; let dy = self.y - y0 as f64;
        let spatial = (dx * dx + dy * dy).sqrt();
        c_diff + 45.0 * ((n as f64 / m as f64).sqrt()) * spatial
    }

    fn add_pixel(&self, x0: usize, y0: usize, in_image: &[Vec<Vec3>]) {
        let mut a = self.accum.lock();
        a.count += 1;
        a.sum_x += x0 as f64;
        a.sum_y += y0 as f64;
        let p = in_image[x0][y0];
        a.sum_l += p.x; a.sum_a += p.y; a.sum_b += p.z;
    }
    fn clear_pixels(&self) { *self.accum.lock() = Accum::default(); }

    fn normalize_probs(&mut self, palette: &[ColorEntry]) {
        let mut denom: f64 = self.p_c.iter().copied().sum();
        if denom == 0.0 { denom = 1.0; }
        // pick palette color of max p_c
        if let Some((imax, _)) = self.p_c
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        {
            self.palette_color = palette[imax].color;
        }
        for v in &mut self.p_c { *v /= denom; }
    }

    fn update_pos(&mut self) {
        let a = self.accum.lock();
        if a.count == 0 {
            self.x = self.original_xy.0;
            self.y = self.original_xy.1;
            return;
        }
        let n = a.count as f64;
        self.x = a.sum_x / n; self.y = a.sum_y / n;
    }

    fn update_sp_color(&mut self, in_image: &[Vec<Vec3>]) {
        let a = self.accum.lock();
        if a.count == 0 { self.sp_color = self.original_color; return; }
        let n = a.count as f64;
        self.sp_color = Vec3 { x: a.sum_l / n, y: a.sum_a / n, z: a.sum_b / n };
    }
}

#[derive(Default)]
struct Accum { count: usize, sum_x: f64, sum_y: f64, sum_l: f64, sum_a: f64, sum_b: f64 }

fn in_bounds(r: isize, c: isize, w_out: usize, h_out: usize) -> bool {
    r >= 0 && c >= 0 && (r as usize) < w_out && (c as usize) < h_out
}

fn sp_thread(coords: Coords, super_pixels: Arc<Vec<Vec<Arc<RwLock<SuperPixel>>>>>, in_image: Arc<Vec<Vec<Vec3>>>, n: usize, m: usize, w_in: usize, h_in: usize, w_out: usize, h_out: usize) {
    let dx = [-1, -1, -1, 0, 0, 0, 1, 1, 1];
    let dy = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
    loop {
        let cur = coords.next();
        let (x, y) = match cur { Some(v) => v, None => break };
        let r = (x * w_out) / w_in;
        let c = (y * h_out) / h_in;
        let mut best_cost = f64::INFINITY;
        let mut best_pair = (r as isize, c as isize);
        for i in 0..9 {
            let rr = r as isize + dx[i];
            let cc = c as isize + dy[i];
            if !in_bounds(rr, cc, w_out, h_out) { continue; }
            let sp_r = super_pixels[rr as usize][cc as usize].read();
            let cost = sp_r.cost(x, y, &in_image, n, m);
            if cost < best_cost { best_cost = cost; best_pair = (rr, cc); }
        }
        let sp = &super_pixels[best_pair.0 as usize][best_pair.1 as usize];
        // Use read lock to access &SuperPixel, then lock its internal accum mutex
        let sp_read = sp.read();
        sp_read.add_pixel(x, y, &in_image);
    }
}

fn sp_refine(super_pixels: &Vec<Vec<Arc<RwLock<SuperPixel>>>>, in_image: &Vec<Vec<Vec3>>, num_threads: usize, w_in: usize, h_in: usize, w_out: usize, h_out: usize) {
    let m = w_in * h_in; let n = w_out * h_out;
    for row in super_pixels.iter() {
        for sp in row.iter() { sp.write().clear_pixels(); }
    }
    let coords = Coords::new(m, w_in, w_out);
    let sp_arc = Arc::new(super_pixels.clone());
    let in_arc = Arc::new(in_image.clone());
    let mut handles = Vec::new();
    for _ in 0..num_threads { 
        let coords_c = coords.clone();
        let sp_c = sp_arc.clone();
        let in_c = in_arc.clone();
        let h = thread::spawn(move || {
            sp_thread(coords_c, sp_c, in_c, n, m, w_in, h_in, w_out, h_out);
        });
        handles.push(h);
    }
    for h in handles { let _ = h.join(); }

    // Update pos and color
    for r in 0..w_out {
        for c in 0..h_out {
            let mut sp = super_pixels[r][c].write();
            sp.update_pos();
            sp.update_sp_color(in_image);
        }
    }

    // Laplacian smoothing
    let mut new_coords = vec![vec![(0.0f64, 0.0f64); h_out]; w_out];
    for r in 0..w_out {
        for c in 0..h_out {
            let dx = [0, 0, -1, 1];
            let dy = [1, -1, 0, 0];
            let mut nnb = 0.0; let mut nx = 0.0; let mut ny = 0.0;
            for i in 0..4 {
                let rr = r as isize + dx[i]; let cc = c as isize + dy[i];
                if in_bounds(rr, cc, w_out, h_out) {
                    nnb += 1.0;
                    let sp = super_pixels[rr as usize][cc as usize].read();
                    nx += sp.x; ny += sp.y;
                }
            }
            nx /= nnb; ny /= nnb;
            let sp = super_pixels[r][c].read();
            new_coords[r][c] = (0.4 * nx + 0.6 * sp.x, 0.4 * ny + 0.6 * sp.y);
        }
    }

    // Bilateral-like smoothing on colors
    let mut new_colors = vec![vec![Vec3::default(); h_out]; w_out];
    for r in 0..w_out {
        for c in 0..h_out {
            let sp = super_pixels[r][c].read();
            let dx = [-1, -1, -1, 0, 0, 1, 1, 1];
            let dy = [-1, 0, 1, -1, 1, -1, 0, 1];
            let mut n = 0.0; let mut avg = Vec3::default();
            for i in 0..8 {
                let rr = r as isize + dx[i]; let cc = c as isize + dy[i];
                if in_bounds(rr, cc, w_out, h_out) {
                    let next = super_pixels[rr as usize][cc as usize].read().sp_color;
                    let weight = (-(sp.sp_color.x - next.x).abs()).exp();
                    avg = avg.add(next.scale(weight));
                    n += weight;
                }
            }
            if n == 0.0 { new_colors[r][c] = sp.sp_color; }
            else {
                avg = avg.scale(1.0 / n);
                let mixed = sp.sp_color.scale(0.5).add(avg.scale(0.5));
                new_colors[r][c] = mixed;
            }
        }
    }

    for r in 0..w_out { for c in 0..h_out {
        let mut sp = super_pixels[r][c].write();
        let (nx, ny) = new_coords[r][c];
        sp.x = nx; sp.y = ny; sp.sp_color = new_colors[r][c];
    }}
}

fn associate(super_pixels: &Vec<Vec<Arc<RwLock<SuperPixel>>>>, palette: &mut Vec<ColorEntry>, _clusters: &Vec<(usize, usize)>, t: f64) {
    for row in super_pixels.iter() {
        for sp in row.iter() {
            let mut spw = sp.write();
            spw.p_c = vec![0.0; palette.len()];
            for k in 0..palette.len() { spw.p_c[k] = palette[k].condit_prob(spw.sp_color, t); }
            // normalize and update palette color (argmax only, like Python)
            spw.normalize_probs(&palette);
        }
    }

    for k in 0..palette.len() { palette[k].probability = 0.0; }
    for row in super_pixels.iter() { for sp in row.iter() {
        let sp = sp.read();
        for k in 0..palette.len() { palette[k].probability += sp.p_c[k] * sp.p_s; }
    }}
}

fn palette_refine(super_pixels: &Vec<Vec<Arc<RwLock<SuperPixel>>>>, palette: &mut Vec<ColorEntry>) -> f64 {
    let mut total_change = 0.0;
    const EPS_PROB: f64 = 1e-12;
    for k in 0..palette.len() {
        let mut newc = Vec3::default();
        let prob = palette[k].probability;
        // Guard against zero/near-zero probability to avoid division by zero and NaNs.
        if prob <= EPS_PROB {
            // Skip update and keep previous color to maintain stability.
            continue;
        }
        for row in super_pixels.iter() { for sp in row.iter() {
            let sp = sp.read();
            newc = newc.add(sp.sp_color.scale((sp.p_c[k] * sp.p_s) / prob));
        }}
        let old = palette[k].color;
        palette[k].color = newc;
        total_change += color_diff(old, newc);
    }
    total_change
}

fn expand(clusters: &mut Vec<(usize, usize)>, palette: &mut Vec<ColorEntry>, epsilon_cluster: f64, k: &mut usize, k_max: usize, delta: Vec3) {
    let mut i = 0;
    while i < *k && *k < k_max {
        let (i0, i1) = clusters[i];
        let (c1, c2) = (palette[i0].color, palette[i1].color);
        if color_diff(c1, c2) > epsilon_cluster {
            *k += 1;
            let p0 = palette[i0].probability / 2.0;
            let p1 = palette[i1].probability / 2.0;
            palette[i0].probability = p0; palette[i1].probability = p1;
            palette.push(ColorEntry { color: c1, probability: p0 });
            palette.push(ColorEntry { color: c2, probability: p1 });
            let len = palette.len();
            clusters.push((i1, len - 1));
            clusters[i] = (i0, len - 2);
        }
        i += 1;
    }

    if *k >= k_max {
        let mut new_palette = Vec::new();
        let mut new_clusters = Vec::new();
        for idx in 0..*k {
            let (i0, i1) = clusters[idx];
            let c0 = palette[i0].color; let p0 = palette[i0].probability;
            let c1 = palette[i1].color; let p1 = palette[i1].probability;
            let mix = Vec3 { x: (c0.x + c1.x) / 2.0, y: (c0.y + c1.y) / 2.0, z: (c0.z + c1.z) / 2.0 };
            new_palette.push(ColorEntry { color: mix, probability: p0 + p1 });
            new_clusters.push((idx, idx));
        }
        *palette = new_palette;
        *clusters = new_clusters;
    } else {
        // allow sub-clusters to separate
        for i in 0..*k { let (_, i1) = clusters[i]; let c = &mut palette[i1]; c.perturb(delta); }
    }
}

fn saturate(out_lab: &mut [Vec<Vec3>], w_out: usize, h_out: usize) {
    for r in 0..w_out { for c in 0..h_out {
        out_lab[r][c].y *= 1.1; // a
        out_lab[r][c].z *= 1.1; // b
    }}
}

fn to_lab_image(rgb: &image::DynamicImage) -> Result<(Vec<Vec<Vec3>>, usize, usize)> {
    let rgb = rgb.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut out = vec![vec![Vec3::default(); h]; w];
    for x in 0..w { for y in 0..h {
        let p = rgb.get_pixel(x as u32, y as u32);
        let srgb: Srgb<f32> = Srgb::new(p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);
        let lin: LinSrgb<f32> = srgb.into_linear();
        let lab: Lab = Lab::from_color(lin);
        out[x][y] = Vec3 { x: lab.l as f64, y: lab.a as f64, z: lab.b as f64 };
    }}
    Ok((out, w, h))
}

fn lab_to_rgb_image(lab: &[Vec<Vec3>], w: usize, h: usize) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w as u32, h as u32);
    for x in 0..w { for y in 0..h {
        let v = lab[x][y];
        let lab = Lab::new(v.x as f32, v.y as f32, v.z as f32);
        let lin: LinSrgb<f32> = lab.into_color();
        let srgb: Srgb<f32> = Srgb::from_linear(lin);
        let r = (srgb.red.clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (srgb.green.clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (srgb.blue.clamp(0.0, 1.0) * 255.0).round() as u8;
        img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
    }}
    img
}

fn read_params() -> Result<(String, String, usize, usize, usize)> {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 6 {
        let in_image_name = args[1].clone();
        let out_image_name = args[2].clone();
        let w_out: usize = args[3].parse().context("invalid width")?;
        let h_out: usize = args[4].parse().context("invalid height")?;
        let k_max: usize = args[5].parse().context("invalid K_max")?;
        return Ok((in_image_name, out_image_name, w_out, h_out, k_max));
    }
    // Fallback to stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let mut lines = input.lines();
    let in_image_name = lines.next().context("missing input image path")?.to_string();
    let out_image_name = lines.next().context("missing output image path")?.to_string();
    let dims = lines.next().context("missing output dimensions")?;
    let k_max_s = lines.next().context("missing K_max")?;
    let mut dims_it = dims.split_whitespace();
    let w_out: usize = dims_it.next().context("missing width")?.parse()?;
    let h_out: usize = dims_it.next().context("missing height")?.parse()?;
    let k_max: usize = k_max_s.parse()?;
    Ok((in_image_name, out_image_name, w_out, h_out, k_max))
}

fn tmp_progress_path(out_image_name: &str, idx: usize) -> String {
    let p = Path::new(out_image_name);
    let parent: PathBuf = p.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let filename = format!("{}_{}.png", stem, idx);
    parent.join(filename).to_string_lossy().into_owned()
}

fn main() -> Result<()> {
    env_logger::init();
    let (in_image_name, out_image_name, w_out, h_out, k_max) = read_params()?;

    info!("Starting Pyxeled (Rust)");
    info!("Input image: {}", in_image_name);
    info!("Output image: {}", out_image_name);
    info!("Output dimensions: {}x{}", w_out, h_out);
    info!("Max clusters (K_max): {}", k_max);

    // Load image and convert to Lab
    let dyn_img = ImageReader::open(&in_image_name)?.decode()?;
    let (in_lab, w_in, h_in) = to_lab_image(&dyn_img)?;

    // PCA prep
    let mut pca_data = Vec::with_capacity(w_in * h_in);
    for x in 0..w_in { for y in 0..h_in { pca_data.push(in_lab[x][y]); }}
    let pc1 = pca_first_component(&pca_data)?;
    let delta = pc1.scale(1.5);
    info!("PCA first component: [{:.3}, {:.3}, {:.3}]", pc1.x, pc1.y, pc1.z);

    // Constants and init
    let mut t = 35.0f64; // same as Python
    let t_final = 1.0f64;
    let mut k = 1usize;
    let alpha = 0.7f64;
    let epsilon_palette = 1.0f64;
    let epsilon_cluster = 0.25f64;
    let num_threads = num_cpus::get().max(1);
    let m = w_in * h_in; let n = w_out * h_out;

    // Initialize super pixels grid
    let avg_color = {
        let mut s = Vec3::default();
        for v in &pca_data { s = s.add(*v); }
        s.scale(1.0 / (pca_data.len() as f64))
    };
    let xs: Vec<usize> = (0..w_out).map(|r| (r * w_in) / w_out).collect();
    let ys: Vec<usize> = (0..h_out).map(|c| (c * h_in) / h_out).collect();

    let mut super_pixels: Vec<Vec<Arc<RwLock<SuperPixel>>>> = vec![vec![]; w_out];
    for r in 0..w_out {
        for c in 0..h_out {
            let xi = xs[r]; let yi = ys[c];
            let sp = SuperPixel::new(xi as f64, yi as f64, avg_color, 1.0 / (n as f64), in_lab[xi][yi]);
            super_pixels[r].push(Arc::new(RwLock::new(sp)));
        }
    }

    // Palette and clusters
    let mut palette = vec![
        ColorEntry { color: avg_color, probability: 0.5 },
        ColorEntry { color: avg_color.add(delta), probability: 0.5 },
    ];
    let mut clusters: Vec<(usize, usize)> = vec![(0, 1)];

    // Main loop
    let mut iterations = 0usize;
    let mut stagnant = 0usize;
    const STAG_EPS: f64 = 1e-6;
    while t > t_final {
        iterations += 1;
        info!("K={}, T={:.3}, iter={}", k, t, iterations);

        sp_refine(&super_pixels, &in_lab, num_threads, w_in, h_in, w_out, h_out);
        associate(&super_pixels, &mut palette, &clusters, t);
        let total_change = palette_refine(&super_pixels, &mut palette);

        if total_change < epsilon_palette {
            t *= alpha;
            if k < k_max { expand(&mut clusters, &mut palette, epsilon_cluster, &mut k, k_max, delta); }
        }

        // Early stop if changes are negligible for a while
        if total_change < STAG_EPS { stagnant += 1; } else { stagnant = 0; }
        if stagnant >= 5 { info!("Early stop: converged (stagnation)"); break; }

        if iterations >= 1010 { warn!("Max iterations reached"); break; }

        if iterations % 100 == 0 && iterations > 1 {
            let mut out_lab = vec![vec![Vec3::default(); h_out]; w_out];
            for r in 0..w_out { for c in 0..h_out {
                out_lab[r][c] = super_pixels[r][c].read().palette_color;
            }}
            saturate(&mut out_lab, w_out, h_out);
            let out_img = lab_to_rgb_image(&out_lab, w_out, h_out);
            let tmp_path = tmp_progress_path(&out_image_name, iterations/100);
            out_img.save(&tmp_path)?;
            info!("Intermediate output saved: {}", tmp_path);
        }
    }

    // Final output
    let mut out_lab = vec![vec![Vec3::default(); h_out]; w_out];
    for r in 0..w_out { for c in 0..h_out { out_lab[r][c] = super_pixels[r][c].read().palette_color; }}
    saturate(&mut out_lab, w_out, h_out);
    let out_img = lab_to_rgb_image(&out_lab, w_out, h_out);
    out_img.save(out_image_name)?;
    info!("Final output saved");

    Ok(())
}
