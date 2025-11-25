# Pixel-convert

Pixelation and palette abstraction based on the paper “Pixelated Image Abstraction” (Gerstner et al.).

This repo contains:
- A Rust core (`pixel_convert`) with a CLI
- A Python package (PyO3 bindings) exposing the algorithm as `pixel_convert.transform`
- A small Python utility (`combine_image.py`) to stitch step images into a single “progression” image

If you want a deeper dive into the algorithm and examples, see the algorithm description in the Rust crate (algorithm_description.md).

## Requirements
- Python 3.12+
- Pillow (`PIL`) for image IO
- Rust toolchain (for building the Python extension)
- Optional: `maturin` for building/installing the Python bindings

Quick setup
- Make sure you have Rust (rustup) and uv installed, then run:
  - `make setup` (creates `.venv`, installs dev deps via uv or pip fallback, and ensures Rust stable toolchain)
  - `make build-py` (installs the Python extension in release into the venv)
  - Optional native-tuned CLI build: `make build-cli-native` (uses `-C target-cpu=native`)

The top-level `pyproject.toml` also lists Python dependencies used in examples and tooling.

## Python API (recommended)

The Python API is provided by the native extension module `pixel_convert` (built from the PyO3 crate).

Functions:
```
pixel_convert.transform(
  image: PIL.Image.Image,
  width: int,
  height: int,
  kmax: int,
  *,
  fast: bool = False,
  stride: int | None = None,
  stride_x: int | None = None,
  stride_y: int | None = None,
  alpha: float | None = None,
  epsilon_palette: float | None = None,
  t_final: float | None = None,
  stag_eps: float | None = None,
  stag_limit: int | None = None,
  threads: int | None = None,
  iter_timings: bool = False,
) -> PIL.Image.Image

pixel_convert.transform_file(
  input_path: str,
  output_path: str,
  width: int,
  height: int,
  kmax: int,
  *,
  fast: bool = False,
  stride: int | None = None,
  stride_x: int | None = None,
  stride_y: int | None = None,
  alpha: float | None = None,
  epsilon_palette: float | None = None,
  t_final: float | None = None,
  stag_eps: float | None = None,
  stag_limit: int | None = None,
  threads: int | None = None,
  iter_timings: bool = False,
) -> None
```

Parameters:
- `width`, `height`: output pixel dimensions
- `kmax`: maximum palette size (distinct colors)
- `fast`: enables a faster preset (minor quality trade-offs)
- `stride`, `stride_x`, `stride_y`: optional pixel subsampling for speed
- `alpha`, `epsilon_palette`, `t_final`, `stag_eps`, `stag_limit`: advanced convergence controls
- `threads`: number of threads (>= 1)
- `iter_timings`: print per-iteration timing to logs (info)

Tip: If Pillow image conversion/tobytes is a bottleneck for you, prefer `transform_file(...)` so Rust performs file I/O directly.

### Build and install the Python extension

Option A: develop install (easy for local iteration)
```
pip install maturin
maturin develop -m pixel_convert_py/Cargo.toml
```

Option B: build a wheel, then install
```
pip install maturin
maturin build -m pixel_convert_py/Cargo.toml --release
pip install target/wheels/*.whl
```

After installation, you can `import pixel_convert` in Python.

### Quickstart example
```python
from PIL import Image
import pixel_convert as rx

# Load any RGB image
img = Image.open("examples/input_images/dog3.jpg")

# 100x100 output with a 30-color palette
out = rx.transform(img, 100, 100, 30, fast=True, threads=4)
out.save("examples/output_images/dog3_100x100_30.png")
```

### File-to-file example (fastest path)
```python
import pixel_convert as rx

rx.transform_file(
    "examples/input_images/dog3.jpg",
    "examples/output_images/dog3_100x100_30.png",
    100,
    100,
    30,
    fast=True,
    threads=4,
)
```

### Batch example (multiple sizes/palettes)
```python
from PIL import Image
import pixel_convert as rx
from pathlib import Path

inp = Path("examples/input_images/dog3.jpg")
out_dir = Path("examples/output_images")
out_dir.mkdir(exist_ok=True)

cfgs = [
    (64,  64,  16),
    (96,  96,  24),
    (128, 128, 30),
]

img = Image.open(inp)
for w, h, k in cfgs:
    out = rx.transform(img, w, h, k, fast=True, threads=4)
    out.save(out_dir / f"dog3_{w}x{h}_{k}colors.png")
```

## Combine progression images

`combine_image.py` scans an input directory for files like `name_1.png`, `name_2.png`, …, `name.png` (final) and creates a single combined progression image per group.

Usage:
```
# Default: reads from output_images/, writes to combined_images/, horizontal layout
python combine_image.py

# Change layout
python combine_image.py --layout vertical
python combine_image.py --layout grid

# Only combine a specific group (by base name)
python combine_image.py --group dog3

# Custom directories
python combine_image.py --input-dir examples/output_images --output-dir examples/combined_images
```

The script creates files like `combined_images/dog3_progression.png` that show step images followed by the final image.

## Rust CLI (optional)

If you prefer a CLI, the Rust implementation provides one:
```
cargo build --release
./target/release/pixel_convert --help
./target/release/pixel_convert examples/input_images/dog3.jpg examples/output_images/dog3_rust.png 100 100 30
# Fast mode
./target/release/pixel_convert --fast examples/input_images/dog3.jpg examples/output_images/dog3_rust_fast.png 100 100 30
```

## Testing

There is a basic Python test that exercises the extension:
```
python -m pytest -q
```
Note: Build/install the `pixel_convert` extension first (see above), otherwise the test will be skipped.

## Notes
- Example input/output folders in this repo: `examples/input_images/`, `examples/output_images/`, `examples/combined_images/`.

## Benchmark Guide

- General
  - Use identical inputs and parameters across runs (`fast`, `threads`, `width`, `height`, `kmax`).
  - Warm the cache: run twice and time the second run.
  - Enable per‑iteration logs for insight: set `RUST_LOG=info` and `iter_timings`/`--iter-timings`.

- Rust CLI (release)
  - Build: `cargo build --release`
  - Single run with timings: 
    `RUST_LOG=info ./target/release/pixel_convert --iter-timings --threads 1 examples/input_images/dog3.jpg examples/output_images/dog3_iter_t1.png 100 100 30`
  - Thread sweep (try 1,2,3,4,6,8):
    `RUST_LOG=info ./target/release/pixel_convert --iter-timings --threads 4 examples/input_images/dog3.jpg examples/output_images/dog3_iter_t4.png 100 100 30`
  - Optional: `--fast` for speed/quality tradeoff.

- Python API (file→file, release wheel)
  - Ensure the extension is built/installed in release.
  - Run with timings:
    `RUST_LOG=info python -c 'import pixel_convert as rx; rx.transform_file("examples/input_images/dog3.jpg","examples/output_images/dog3_py_t1.png",100,100,30, threads=1, fast=True, iter_timings=True)'
    `
  - Sweep threads as above (1,2,3,4,6,8) and compare wall time and `iter_time_ms`.


- Interpreting results
  - Prefer the smallest wall time; in many cases small thread counts perform best.
  - If Python is consistently slower than the CLI with identical params, check that both are the same architecture (e.g., arm64) and that the Python wheel is release‑optimized.
