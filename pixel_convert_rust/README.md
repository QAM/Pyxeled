pixel_convert
=============

A Rust image processing core with a clean CLI (clap) and a reusable library API.

Usage
-----

- Build: `cargo build --release`

- Run (CLI args):
  `./target/release/pixel_convert examples/input_images/dog3.jpg examples/output_images/dog3_rust.png 100 100 4`

- Help and flags:
  `./target/release/pixel_convert --help`

- Fast mode (~quality tradeoffs, much faster):
  `./target/release/pixel_convert --fast examples/input_images/dog3.jpg examples/output_images/dog3_rust.png 100 100 4`
  - Uses 2x2 subsampling for assignment
  - More aggressive cooling and convergence thresholds
  - Slightly higher palette-change threshold

- Per‑iteration timings (log at info):
  `RUST_LOG=info ./target/release/pixel_convert --iter-timings examples/input_images/dog3.jpg examples/output_images/dog3_160x160_8.png 160 160 8`

- Advanced overrides (after flags, before positional args):
  `--stride <n>` (both axes) | `--stride-x <n>` | `--stride-y <n>`
  `--alpha <f64>` | `--epsilon-palette <f64>` | `--t-final <f64>`
  `--stag-eps <f64>` | `--stag-limit <usize>` | `--threads <usize>`
  `--iter-timings` (print per-iteration timing)
  Example:
  `./target/release/pixel_convert --fast --stride 3 --alpha 0.55 --threads 8 examples/input_images/dog3.jpg examples/output_images/dog3_160x160_8.png 160 160 8`

Palette mapping subcommand
--------------------------

Map each pixel to the nearest color from a provided palette. Algorithms: `rgb`, `lab`, `ciede2000`.

```bash
# Built-in DMC palette (no need to pass colors)
./target/release/pixel_convert map \
  --palette dmc --algorithm ciede2000 \
  examples/input_images/dog3.jpg examples/output_images/dog3_map_dmc.png

# RGB, colors as R,G,B tuples
./target/release/pixel_convert map \
  --algorithm rgb \
  --color 0,0,0 --color 255,255,255 --color 220,20,60 \
  examples/input_images/dog3.jpg examples/output_images/dog3_map_rgb.png

# Lab Euclidean (DeltaE76), hex colors
./target/release/pixel_convert map \
  --algorithm lab \
  --color #000000 --color #FFFFFF --color #DC143C \
  examples/input_images/dog3.jpg examples/output_images/dog3_map_lab.png

# CIEDE2000 (perceptual)
./target/release/pixel_convert map \
  --algorithm ciede2000 \
  --color #000000 --color #FFFFFF --color #DC143C \
  examples/input_images/dog3.jpg examples/output_images/dog3_map_ciede2000.png
```

Library API
-----------

Use the core algorithm in your own Rust code:

```
use pixel_convert_rust::{process, default_config, Params};

fn run() -> anyhow::Result<()> {
    let mut cfg = default_config(false); // or true for fast preset
    cfg.stride_x = 1;
    cfg.stride_y = 1;
    let params = Params {
        in_image_name: "examples/input_images/dog3.jpg".into(),
        out_image_name: "examples/output_images/dog3_rust.png".into(),
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
