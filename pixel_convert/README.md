pixel_convert (Python package)

This package exposes a Python API backed by the Rust `pixel_convert_rust` core via PyO3.

API
- `pixel_convert.transform(image: PIL.Image.Image, width: int, height: int, kmax: int, *, fast=False, stride=None, stride_x=None, stride_y=None, alpha=None, epsilon_palette=None, t_final=None, stag_eps=None, stag_limit=None, threads=None, iter_timings=False) -> PIL.Image.Image`
- `pixel_convert.transform_file(input_path: str, output_path: str, width: int, height: int, kmax: int, *, fast=False, stride=None, stride_x=None, stride_y=None, alpha=None, epsilon_palette=None, t_final=None, stag_eps=None, stag_limit=None, threads=None, iter_timings=False) -> None`
 - `pixel_convert.map_to_colors_image(image: PIL.Image.Image, colors: list[tuple[int,int,int]], algorithm: str = "rgb") -> PIL.Image.Image`
 - `pixel_convert.map_to_colors_file(input_path: str, output_path: str, colors: list[tuple[int,int,int]], algorithm: str = "rgb") -> None`
 - `pixel_convert.map_to_named_palette_image(image: PIL.Image.Image, palette_name: str, algorithm: str = "rgb") -> PIL.Image.Image`
 - `pixel_convert.map_to_named_palette_file(input_path: str, output_path: str, palette_name: str, algorithm: str = "rgb") -> None`

Color mapping algorithms
- `"rgb"` / `"rgb_euclidean"`: Euclidean distance in RGB.
- `"lab"` / `"deltae76"` / `"lab_euclidean"`: Euclidean distance in CIE Lab (D65).
- `"ciede2000"` / `"de2000"` / `"deltae2000"` / `"de00"`: CIEDE2000 perceptual distance.

Build from source
- `pip install maturin`
- `maturin build -m pixel_convert/Cargo.toml --release`
- `pip install target/wheels/*.whl`

Develop install for local testing
- `maturin develop -m pixel_convert/Cargo.toml`

Example
```
from PIL import Image
import pixel_convert as rx

img = Image.open("examples/input_images/dog3.jpg")
out = rx.transform(img, 128, 128, 30, fast=True)
out.save("examples/output_images/dog3_py_128x128_30.png")
```

Palette mapping (PIL image)
```
from PIL import Image
import pixel_convert as rx

img = Image.open("examples/input_images/dog3.jpg")
palette = [(0,0,0), (255,255,255), (220,20,60)]
for algo in ["rgb", "lab", "ciede2000"]:
    out = rx.map_to_colors_image(img, palette, algorithm=algo)
    out.save(f"examples/output_images/dog3_map_{algo}.png")
```

Palette mapping with named palette (DMC)
```
from PIL import Image
import pixel_convert as rx

img = Image.open("examples/input_images/dog3.jpg")
for algo in ["rgb", "lab", "ciede2000"]:
    out = rx.map_to_named_palette_image(img, "dmc", algorithm=algo)
    out.save(f"examples/output_images/dog3_map_dmc_{algo}.png")
```

File-to-file example (faster, Rust handles I/O)
```
import pixel_convert as rx

rx.transform_file(
    "examples/input_images/dog3.jpg",
    "examples/output_images/dog3_py_128x128_30.png",
    128,
    128,
    30,
    fast=True,
    threads=4,
)
```

Palette mapping (file-to-file)
```
import pixel_convert as rx
palette = [(0,0,0), (255,255,255), (220,20,60)]
rx.map_to_colors_file(
    "examples/input_images/dog3.jpg",
    "examples/output_images/dog3_map_ciede.png",
    palette,
    algorithm="ciede2000",
)
```

Palette mapping (file-to-file) with named palette
```
import pixel_convert as rx
rx.map_to_named_palette_file(
    "examples/input_images/dog3.jpg",
    "examples/output_images/dog3_map_dmc_lab.png",
    "dmc",
    algorithm="lab",
)
```

Timing per iteration (logs)
```
import pixel_convert as rx, os
os.environ["RUST_LOG"] = "info"
rx.transform_file("examples/input_images/dog3.jpg", "examples/output_images/dog3_py_128x128_30.png", 128, 128, 30, iter_timings=True)
```
