# Pyxeled

Pixelation and palette abstraction based on the paper “Pixelated Image Abstraction” (Gerstner et al.).

This repo contains:
- A Rust core (`rust_pyxeled`) with a CLI
- A Python package (PyO3 bindings) exposing the algorithm as `rust_pyxeled.transform`
- A small Python utility (`combine_image.py`) to stitch step images into a single “progression” image

If you want a deeper dive into the algorithm and examples, see the Rust docs in `rust_pyxeled/algorithm_description.md`.

## Requirements
- Python 3.12+
- Pillow (`PIL`) for image IO
- Rust toolchain (for building the Python extension)
- Optional: `maturin` for building/installing the Python bindings

The top-level `pyproject.toml` also lists Python dependencies used in examples and tooling.

## Python API (recommended)

The Python API is provided by the native extension module `rust_pyxeled`, built from `rust_pyxeled_py/`.

Functions:
```
rust_pyxeled.transform(
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
) -> PIL.Image.Image

rust_pyxeled.transform_file(
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
) -> None
```

Parameters:
- `width`, `height`: output pixel dimensions
- `kmax`: maximum palette size (distinct colors)
- `fast`: enables a faster preset (minor quality trade-offs)
- `stride`, `stride_x`, `stride_y`: optional pixel subsampling for speed
- `alpha`, `epsilon_palette`, `t_final`, `stag_eps`, `stag_limit`: advanced convergence controls
- `threads`: number of threads (>= 1)

Tip: If Pillow image conversion/tobytes is a bottleneck for you, prefer `transform_file(...)` so Rust performs file I/O directly.

### Build and install the Python extension

Option A: develop install (easy for local iteration)
```
pip install maturin
maturin develop -m rust_pyxeled_py/Cargo.toml
```

Option B: build a wheel, then install
```
pip install maturin
maturin build -m rust_pyxeled_py/Cargo.toml --release
pip install target/wheels/*.whl
```

After installation, you can `import rust_pyxeled` in Python.

### Quickstart example
```python
from PIL import Image
import rust_pyxeled as rx

# Load any RGB image
img = Image.open("input_images/dog3.jpg")

# 100x100 output with a 30-color palette
out = rx.transform(img, 100, 100, 30, fast=True, threads=4)
out.save("output_images/dog3_100x100_30.png")
```

### File-to-file example (fastest path)
```python
import rust_pyxeled as rx

rx.transform_file(
    "input_images/dog3.jpg",
    "output_images/dog3_100x100_30.png",
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
import rust_pyxeled as rx
from pathlib import Path

inp = Path("input_images/dog3.jpg")
out_dir = Path("output_images")
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
python combine_image.py --input-dir output_images --output-dir combined_images
```

The script creates files like `combined_images/dog3_progression.png` that show step images followed by the final image.

## Rust CLI (optional)

If you prefer a CLI, the Rust implementation provides one in `rust_pyxeled/`:
```
cargo build --release
./target/release/rust_pyxeled --help
./target/release/rust_pyxeled input_images/dog3.jpg output_images/dog3_rust.png 100 100 30
# Fast mode
./target/release/rust_pyxeled --fast input.jpg out.png 100 100 30
```

## Testing

There is a basic Python test that exercises the extension:
```
python -m pytest -q
```
Note: Build/install the `rust_pyxeled` extension first (see above), otherwise the test will be skipped.

## Notes
- The old `pyxeled.py`/stdin configuration flow is no longer used. Prefer the Python API or the Rust CLI above.
- Example input/output folders in this repo: `input_images/`, `output_images/`, `combined_images/`.
