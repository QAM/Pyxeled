// CLI entry for pixel_convert
use anyhow::{bail, Result};
use clap::{ArgAction, Args, Parser, ValueHint};
use pixel_convert_rust::{
    default_config, load_palette_file, map_file_to_palette, named_palette_vec, process, ColorDistanceAlgorithm, Config,
    Params,
};

#[derive(Parser, Debug)]
#[command(name = "pixel_convert", version, about = "Pixel Convert image algorithm (Rust core)")]
struct Cli {
    /// Fast preset (subsampling + aggressive cooling)
    #[arg(short = 'f', long = "fast", action = ArgAction::SetTrue)]
    fast: bool,

    /// Subsampling stride for both axes
    #[arg(long = "stride")]
    stride: Option<usize>,
    /// Subsampling stride (x axis)
    #[arg(long = "stride-x")]
    stride_x: Option<usize>,
    /// Subsampling stride (y axis)
    #[arg(long = "stride-y")]
    stride_y: Option<usize>,

    /// Annealing alpha (cooling factor)
    #[arg(long = "alpha")]
    alpha: Option<f64>,
    /// Palette-change threshold for cooling
    #[arg(long = "epsilon-palette")]
    epsilon_palette: Option<f64>,
    /// Final temperature
    #[arg(long = "t-final")]
    t_final: Option<f64>,
    /// Stagnation epsilon for early stop
    #[arg(long = "stag-eps")]
    stag_eps: Option<f64>,
    /// Stagnation iteration limit
    #[arg(long = "stag-limit")]
    stag_limit: Option<usize>,
    /// Number of threads
    #[arg(long = "threads")]
    threads: Option<usize>,

    /// Print per-iteration timings (ms)
    #[arg(long = "iter-timings", action = ArgAction::SetTrue)]
    iter_timings: bool,

    /// Input image path
    #[arg(value_hint = ValueHint::FilePath)]
    input: String,
    /// Output image path
    #[arg(value_hint = ValueHint::FilePath)]
    output: String,
    /// Output width
    width: usize,
    /// Output height
    height: usize,
    /// Maximum clusters
    kmax: usize,
}

fn build_config(cli: &Cli) -> Config {
    let mut cfg = default_config(cli.fast);
    if let Some(v) = cli.stride {
        cfg.stride_x = v;
        cfg.stride_y = v;
    }
    if let Some(v) = cli.stride_x {
        cfg.stride_x = v;
    }
    if let Some(v) = cli.stride_y {
        cfg.stride_y = v;
    }
    if let Some(v) = cli.alpha {
        cfg.alpha = v;
    }
    if let Some(v) = cli.epsilon_palette {
        cfg.epsilon_palette = v;
    }
    if let Some(v) = cli.t_final {
        cfg.t_final = v;
    }
    if let Some(v) = cli.stag_eps {
        cfg.stag_eps = v;
    }
    if let Some(v) = cli.stag_limit {
        cfg.stag_limit = v;
    }
    if let Some(v) = cli.threads {
        cfg.num_threads = v.max(1);
    }
    if cli.iter_timings {
        cfg.iter_timings = true;
    }
    cfg
}

#[derive(Parser, Debug)]
#[command(name = "pixel_convert map", about = "Map pixels to a given RGB palette")]
struct MapCli {
    /// Algorithm: rgb | lab | ciede2000
    #[arg(long = "algorithm", default_value = "rgb")]
    algorithm: String,

    /// Palette color, repeatable. Formats: R,G,B or #RRGGBB
    #[arg(long = "color")]
    color: Vec<String>,

    /// Named palette to use (e.g., dmc). Ignored if --color is provided.
    #[arg(long = "palette")]
    palette: Option<String>,

    /// Load palette from a file (CSV or hex list). Ignored if --color is provided.
    #[arg(long = "palette-file", value_hint = ValueHint::FilePath)]
    palette_file: Option<String>,

    /// Input image path
    #[arg(value_hint = ValueHint::FilePath)]
    input: String,
    /// Output image path
    #[arg(value_hint = ValueHint::FilePath)]
    output: String,
}

fn parse_color(s: &str) -> Result<(u8, u8, u8)> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix('#') {
        if hex.len() == 6 {
            let r = u8::from_str_radix(&hex[0..2], 16)?;
            let g = u8::from_str_radix(&hex[2..4], 16)?;
            let b = u8::from_str_radix(&hex[4..6], 16)?;
            return Ok((r, g, b));
        }
    }
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() == 3 {
        let r: u8 = parts[0].trim().parse()?;
        let g: u8 = parts[1].trim().parse()?;
        let b: u8 = parts[2].trim().parse()?;
        return Ok((r, g, b));
    }
    bail!("invalid color format: {}", s)
}

fn main() -> Result<()> {
    env_logger::init();
    // Simple manual dispatch to avoid breaking existing positional CLI
    if std::env::args().nth(1).as_deref() == Some("map") {
        // Parse map-specific CLI (skip the first arg which is the subcommand name)
        let args: Vec<String> = std::env::args().skip(2).collect();
        let map_cli = MapCli::parse_from(std::iter::once("pixel_convert").map(String::from).chain(args.into_iter()));
        let colors: Vec<(u8, u8, u8)> = if !map_cli.color.is_empty() {
            map_cli.color.iter().map(|c| parse_color(c)).collect::<Result<_>>()?
        } else if let Some(name) = &map_cli.palette {
            named_palette_vec(name).ok_or_else(|| anyhow::anyhow!("unknown named palette: {}", name))?
        } else if let Some(path) = &map_cli.palette_file {
            load_palette_file(path)?
        } else {
            bail!("provide --color, or --palette <name>, or --palette-file <path>");
        };
        let algo = ColorDistanceAlgorithm::from_str(&map_cli.algorithm);
        map_file_to_palette(&map_cli.input, &map_cli.output, &colors, algo)
    } else {
        let cli = Cli::parse();
        let cfg = build_config(&cli);
        let params = Params {
            in_image_name: cli.input,
            out_image_name: cli.output,
            w_out: cli.width,
            h_out: cli.height,
            k_max: cli.kmax,
            config: cfg,
        };
        process(params)
    }
}
