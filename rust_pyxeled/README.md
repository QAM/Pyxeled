rust_pyxeled
=============

A Rust port of the Pyxeled image processing algorithm. It mirrors the Python script's stdin-driven interface for easy drop-in use.

Usage
-----

- Build: `cargo build --release`

- Run (CLI args):
  `./target/release/rust_pyxeled input_images/dog3.jpg output_images/dog3_rust.png 100 100 4`

- Or run (stdin parameters, compatible with the Python script):
  `printf 'input_images/dog3.jpg\noutput_images/dog3_rust.png\n100 100\n4\n' | ./target/release/rust_pyxeled`

Notes
-----

- Color conversions use the `palette` crate (sRGB <-> CIE Lab, D65).
- The algorithm uses a 3x3 neighborhood search and approximate bilateral smoothing, matching the Python.
- PCA's first component is obtained via power iteration on the 3x3 covariance matrix.
- Logging uses `env_logger`; set `RUST_LOG=info` for progress.
