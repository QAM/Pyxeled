use anyhow::{bail, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::OnceLock;

/// Compile-time embedded DMC palette loaded from `src/data/dmc_palette.txt`.
/// Lines formatted as `R,G,B`.
static DMC_PALETTE: OnceLock<Vec<(u8, u8, u8)>> = OnceLock::new();

fn get_dmc_palette() -> &'static [(u8, u8, u8)] {
    let v = DMC_PALETTE.get_or_init(|| {
        let mut out = Vec::new();
        let s = include_str!("data/dmc_palette.txt");
        for line in s.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() == 3 {
                if let (Ok(r), Ok(g), Ok(b)) =
                    (parts[0].trim().parse::<u8>(), parts[1].trim().parse::<u8>(), parts[2].trim().parse::<u8>())
                {
                    out.push((r, g, b));
                }
            }
        }
        out
    });
    v.as_slice()
}

pub fn named_palette(name: &str) -> Option<&'static [(u8, u8, u8)]> {
    match name.to_ascii_lowercase().as_str() {
        "dmc" | "dmc_color" | "dmc_colors" | "dmc_palette" => Some(get_dmc_palette()),
        _ => None,
    }
}

pub fn named_palette_vec(name: &str) -> Option<Vec<(u8, u8, u8)>> {
    named_palette(name).map(|s| s.to_vec())
}

/// Load a palette from a simple text file.
/// Accepted formats per line: `R,G,B` or `#RRGGBB`. Ignores blank lines and comments starting with `#`.
pub fn load_palette_file(path: &str) -> Result<Vec<(u8, u8, u8)>> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for line in r.lines() {
        let line = line?;
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if s.starts_with('#') && s.len() != 7 {
            continue;
        }
        if let Some(hex) = s.strip_prefix('#') {
            if hex.len() == 6 {
                let r = u8::from_str_radix(&hex[0..2], 16)?;
                let g = u8::from_str_radix(&hex[2..4], 16)?;
                let b = u8::from_str_radix(&hex[4..6], 16)?;
                out.push((r, g, b));
                continue;
            }
        }
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() == 3 {
            let r: u8 = parts[0].trim().parse()?;
            let g: u8 = parts[1].trim().parse()?;
            let b: u8 = parts[2].trim().parse()?;
            out.push((r, g, b));
        }
    }
    if out.is_empty() {
        bail!("no colors parsed from palette file: {}", path);
    }
    Ok(out)
}
