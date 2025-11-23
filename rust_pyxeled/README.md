rust_pyxeled
=============

A Rust port of the Pyxeled image processing algorithm with a clean CLI (clap) and a reusable library API.

Usage
-----

- Build: `cargo build --release`

- Run (CLI args):
  `./target/release/rust_pyxeled input_images/dog3.jpg output_images/dog3_rust.png 100 100 4`

- Help and flags:
  `./target/release/rust_pyxeled --help`

- Fast mode (~quality tradeoffs, much faster):
  `./target/release/rust_pyxeled --fast input_images/dog3.jpg output_images/dog3_rust.png 100 100 4`
  - Uses 2x2 subsampling for assignment
  - More aggressive cooling and convergence thresholds
  - Slightly higher palette-change threshold

- Advanced overrides (after flags, before positional args):
  `--stride <n>` (both axes) | `--stride-x <n>` | `--stride-y <n>`
  `--alpha <f64>` | `--epsilon-palette <f64>` | `--t-final <f64>`
  `--stag-eps <f64>` | `--stag-limit <usize>` | `--threads <usize>`
  Example:
  `./target/release/rust_pyxeled --fast --stride 3 --alpha 0.55 --threads 8 input.jpg out.png 160 160 8`

Library API
-----------

Use the core algorithm in your own Rust code:

```
use rust_pyxeled::{process, default_config, Params};

fn run() -> anyhow::Result<()> {
    let mut cfg = default_config(false); // or true for fast preset
    cfg.stride_x = 1;
    cfg.stride_y = 1;
    let params = Params {
        in_image_name: "input_images/dog3.jpg".into(),
        out_image_name: "output_images/dog3_rust.png".into(),
        w_out: 100,
        h_out: 100,
        k_max: 120,
        config: cfg,
    };
    process(params)
}
```

Notes
-----

- Color conversions use the `palette` crate (sRGB <-> CIE Lab, D65).
- 3x3 neighborhood search for assignments + edge-aware smoothing for stability.
- PCA's first component via power iteration on the 3x3 covariance.
- Logging via `env_logger`; set `RUST_LOG=info` for progress.
- See `algorithm_description.md` for a full algorithm walk-through and references.
