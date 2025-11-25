pixel_convert_py (Python package)

This package exposes a Python API backed by the Rust `pixel_convert` core via PyO3.

API
- `pixel_convert.transform(image: PIL.Image.Image, width: int, height: int, kmax: int, *, fast=False, stride=None, stride_x=None, stride_y=None, alpha=None, epsilon_palette=None, t_final=None, stag_eps=None, stag_limit=None, threads=None, iter_timings=False) -> PIL.Image.Image`
- `pixel_convert.transform_file(input_path: str, output_path: str, width: int, height: int, kmax: int, *, fast=False, stride=None, stride_x=None, stride_y=None, alpha=None, epsilon_palette=None, t_final=None, stag_eps=None, stag_limit=None, threads=None, iter_timings=False) -> None`

Build from source
- `pip install maturin`
- `maturin build -m pixel_convert_py/Cargo.toml --release`
- `pip install target/wheels/*.whl`

Develop install for local testing
- `maturin develop -m pixel_convert_py/Cargo.toml`

Example
```
from PIL import Image
import pixel_convert as rx

img = Image.open("input.jpg")
out = rx.transform(img, 128, 128, 30, fast=True)
out.save("out.png")
```

File-to-file example (faster, Rust handles I/O)
```
import pixel_convert as rx

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

Timing per iteration (logs)
```
import pixel_convert as rx, os
os.environ["RUST_LOG"] = "info"
rx.transform_file("input.jpg", "out.png", 128, 128, 30, iter_timings=True)
```
