rust_pyxeled (Python package)

This package exposes a Python API backed by the Rust `rust_pyxeled` core via PyO3.

API
- `rust_pyxeled.transform(image: PIL.Image.Image, width: int, height: int, kmax: int, *, fast=False, stride=None, stride_x=None, stride_y=None, alpha=None, epsilon_palette=None, t_final=None, stag_eps=None, stag_limit=None, threads=None) -> PIL.Image.Image`
- `rust_pyxeled.transform_file(input_path: str, output_path: str, width: int, height: int, kmax: int, *, fast=False, stride=None, stride_x=None, stride_y=None, alpha=None, epsilon_palette=None, t_final=None, stag_eps=None, stag_limit=None, threads=None) -> None`

Build from source
- `pip install maturin`
- `maturin build -m rust_pyxeled_py/Cargo.toml --release`
- `pip install target/wheels/*.whl`

Develop install for local testing
- `maturin develop -m rust_pyxeled_py/Cargo.toml`

Example
```
from PIL import Image
import rust_pyxeled as rx

img = Image.open("input.jpg")
out = rx.transform(img, 128, 128, 30, fast=True)
out.save("out.png")
```

File-to-file example (faster, Rust handles I/O)
```
import rust_pyxeled as rx

rx.transform_file(
    "input.jpg",
    "out.png",
    128,
    128,
    30,
    fast=True,
    threads=4,
)
```
