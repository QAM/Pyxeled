use anyhow::{bail, Result};
use image::{ImageBuffer, Rgb, RgbImage};
use palette::{FromColor, Lab, LinSrgb, Srgb};

#[derive(Clone, Copy, Debug)]
pub enum ColorDistanceAlgorithm {
    Rgb,
    Lab,
    Ciede2000,
}

impl ColorDistanceAlgorithm {
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "rgb" | "rgb_euclidean" => Self::Rgb,
            "lab" | "deltae76" | "lab_euclidean" => Self::Lab,
            "ciede2000" | "de2000" | "deltae2000" | "de00" => Self::Ciede2000,
            _ => Self::Rgb,
        }
    }
}

#[inline]
fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let srgb: Srgb<f32> = Srgb::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let lin: LinSrgb<f32> = srgb.into_linear();
    let lab: Lab = Lab::from_color(lin);
    (lab.l, lab.a, lab.b)
}

// CIEDE2000 distance implementation (returns squared distance for consistency with comparisons)
// Based on Sharma et al. 2005. Uses f32 for performance which is sufficient for comparisons.
fn ciede2000_sq(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    // Convert to CIEDE2000 components
    let k_l = 1.0f32;
    let k_c = 1.0f32;
    let k_h = 1.0f32;

    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1 + c2) * 0.5;
    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25f32.powi(7))).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();
    let h1p = b1.atan2(a1p).to_degrees().rem_euclid(360.0);
    let h2p = b2.atan2(a2p).to_degrees().rem_euclid(360.0);

    let dl = l2 - l1;
    let dc = c2p - c1p;
    let dh = if c1p * c2p == 0.0 {
        0.0
    } else {
        let mut dhp = h2p - h1p;
        if dhp > 180.0 {
            dhp -= 360.0;
        }
        if dhp < -180.0 {
            dhp += 360.0;
        }
        dhp.to_radians() * 2.0 * (c1p * c2p).sqrt()
    };

    let l_bar = (l1 + l2) * 0.5;
    let c_bar_p = (c1p + c2p) * 0.5;
    let h_bar_p = if c1p * c2p == 0.0 {
        h1p + h2p
    } else {
        let mut hsum = h1p + h2p;
        if (h1p - h2p).abs() > 180.0 {
            if hsum < 360.0 {
                hsum += 360.0;
            } else {
                hsum -= 360.0;
            }
        }
        hsum * 0.5
    };

    let t = 1.0 - 0.17 * (h_bar_p - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar_p).to_radians().cos()
        + 0.32 * (3.0 * h_bar_p + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar_p - 63.0).to_radians().cos();

    let s_l = 1.0 + (0.015 * (l_bar - 50.0).powi(2)) / (20.0 + (l_bar - 50.0).powi(2)).sqrt();
    let s_c = 1.0 + 0.045 * c_bar_p;
    let s_h = 1.0 + 0.015 * c_bar_p * t;

    let r_t = {
        let delta_theta = 30.0 * (-(((h_bar_p - 275.0) / 25.0).powi(2))).exp();
        let r_c = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + 25f32.powi(7))).sqrt();
        -r_c * delta_theta.to_radians().sin()
    };

    let d_l = dl / (k_l * s_l);
    let d_c = dc / (k_c * s_c);
    let d_h = dh / (k_h * s_h);
    let de = (d_l * d_l) + (d_c * d_c) + (d_h * d_h) + r_t * d_c * d_h;
    de
}

pub fn map_image_to_palette(
    rgb: &RgbImage,
    palette_rgb: &[(u8, u8, u8)],
    algorithm: ColorDistanceAlgorithm,
) -> RgbImage {
    let (w, h) = rgb.dimensions();
    let mut out: RgbImage = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w, h);
    if palette_rgb.is_empty() {
        return out; // empty image; but callers should validate
    }

    match algorithm {
        ColorDistanceAlgorithm::Rgb => {
            let pal: Vec<(i32, i32, i32)> =
                palette_rgb.iter().map(|&(r, g, b)| (r as i32, g as i32, b as i32)).collect();
            for y in 0..h {
                for x in 0..w {
                    let p = rgb.get_pixel(x, y);
                    let (pr, pg, pb) = (p[0] as i32, p[1] as i32, p[2] as i32);
                    let mut best = 0usize;
                    let mut bestd = u32::MAX;
                    for (i, (r, g, b)) in pal.iter().enumerate() {
                        let dr = pr - *r;
                        let dg = pg - *g;
                        let db = pb - *b;
                        let d = (dr * dr + dg * dg + db * db) as u32;
                        if d < bestd {
                            bestd = d;
                            best = i;
                        }
                    }
                    let (r, g, b) = pal[best];
                    out.put_pixel(x, y, Rgb([r as u8, g as u8, b as u8]));
                }
            }
        }
        ColorDistanceAlgorithm::Lab => {
            let pal_lab: Vec<(f32, f32, f32)> = palette_rgb.iter().map(|&(r, g, b)| rgb_to_lab(r, g, b)).collect();
            for y in 0..h {
                for x in 0..w {
                    let p = rgb.get_pixel(x, y);
                    let (l, a, b) = rgb_to_lab(p[0], p[1], p[2]);
                    let mut best = 0usize;
                    let mut bestd = f32::MAX;
                    for (i, &(pl, pa, pb)) in pal_lab.iter().enumerate() {
                        let dl = l - pl;
                        let da = a - pa;
                        let db = b - pb;
                        let d = dl * dl + da * da + db * db;
                        if d < bestd {
                            bestd = d;
                            best = i;
                        }
                    }
                    let (r, g, b) = palette_rgb[best];
                    out.put_pixel(x, y, Rgb([r, g, b]));
                }
            }
        }
        ColorDistanceAlgorithm::Ciede2000 => {
            let pal_lab: Vec<(f32, f32, f32)> = palette_rgb.iter().map(|&(r, g, b)| rgb_to_lab(r, g, b)).collect();
            for y in 0..h {
                for x in 0..w {
                    let p = rgb.get_pixel(x, y);
                    let (l, a, b) = rgb_to_lab(p[0], p[1], p[2]);
                    let mut best = 0usize;
                    let mut bestd = f32::MAX;
                    for (i, &(pl, pa, pb)) in pal_lab.iter().enumerate() {
                        let d = ciede2000_sq(l, a, b, pl, pa, pb);
                        if d < bestd {
                            bestd = d;
                            best = i;
                        }
                    }
                    let (r, g, b) = palette_rgb[best];
                    out.put_pixel(x, y, Rgb([r, g, b]));
                }
            }
        }
    }
    out
}

pub fn map_file_to_palette(
    input_path: &str,
    output_path: &str,
    palette_rgb: &[(u8, u8, u8)],
    algorithm: ColorDistanceAlgorithm,
) -> Result<()> {
    if palette_rgb.is_empty() {
        bail!("palette colors list must not be empty");
    }
    let dyn_img = image::io::Reader::open(input_path)?.decode()?;
    let rgb = dyn_img.to_rgb8();
    let out = map_image_to_palette(&rgb, palette_rgb, algorithm);
    out.save(output_path)?;
    Ok(())
}
