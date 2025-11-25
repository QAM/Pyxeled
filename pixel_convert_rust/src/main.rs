// CLI entry for pixel_convert
use anyhow::Result;
use clap::{ArgAction, Parser, ValueHint};
use pixel_convert_rust::{default_config, process, Config, Params};

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
    if let Some(v) = cli.stride { cfg.stride_x = v; cfg.stride_y = v; }
    if let Some(v) = cli.stride_x { cfg.stride_x = v; }
    if let Some(v) = cli.stride_y { cfg.stride_y = v; }
    if let Some(v) = cli.alpha { cfg.alpha = v; }
    if let Some(v) = cli.epsilon_palette { cfg.epsilon_palette = v; }
    if let Some(v) = cli.t_final { cfg.t_final = v; }
    if let Some(v) = cli.stag_eps { cfg.stag_eps = v; }
    if let Some(v) = cli.stag_limit { cfg.stag_limit = v; }
    if let Some(v) = cli.threads { cfg.num_threads = v.max(1); }
    if cli.iter_timings { cfg.iter_timings = true; }
    cfg
}

fn main() -> Result<()> {
    env_logger::init();
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
