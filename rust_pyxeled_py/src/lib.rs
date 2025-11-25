use pyo3::prelude::*;
use pyo3::types::PyBytes;

use ::rust_pyxeled::{default_config, process_dynamic, Config};

fn build_config_from_kwargs(
    fast: bool,
    stride: Option<usize>,
    stride_x: Option<usize>,
    stride_y: Option<usize>,
    alpha: Option<f64>,
    epsilon_palette: Option<f64>,
    t_final: Option<f64>,
    stag_eps: Option<f64>,
    stag_limit: Option<usize>,
    threads: Option<usize>,
) -> Config {
    let mut cfg = default_config(fast);
    if let Some(v) = stride { cfg.stride_x = v; cfg.stride_y = v; }
    if let Some(v) = stride_x { cfg.stride_x = v; }
    if let Some(v) = stride_y { cfg.stride_y = v; }
    if let Some(v) = alpha { cfg.alpha = v; }
    if let Some(v) = epsilon_palette { cfg.epsilon_palette = v; }
    if let Some(v) = t_final { cfg.t_final = v; }
    if let Some(v) = stag_eps { cfg.stag_eps = v; }
    if let Some(v) = stag_limit { cfg.stag_limit = v; }
    if let Some(v) = threads { cfg.num_threads = v.max(1); }
    cfg
}

#[pyfunction]
#[pyo3(signature = (
    image,
    width,
    height,
    kmax,
    *,
    fast = false,
    stride = None,
    stride_x = None,
    stride_y = None,
    alpha = None,
    epsilon_palette = None,
    t_final = None,
    stag_eps = None,
    stag_limit = None,
    threads = None,
))]
pub fn transform(
    py: Python<'_>,
    image: &PyAny,
    width: usize,
    height: usize,
    kmax: usize,
    fast: bool,
    stride: Option<usize>,
    stride_x: Option<usize>,
    stride_y: Option<usize>,
    alpha: Option<f64>,
    epsilon_palette: Option<f64>,
    t_final: Option<f64>,
    stag_eps: Option<f64>,
    stag_limit: Option<usize>,
    threads: Option<usize>,
) -> PyResult<PyObject> {
    // Ensure RGB and pull pixel data
    let img_rgb = image.call_method1("convert", ("RGB",))?;
    let (w, h): (u32, u32) = img_rgb.getattr("size")?.extract()?;
    let data: Vec<u8> = img_rgb.call_method0("tobytes")?.extract()?;

    // Build DynamicImage
    let buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(w, h, data)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid image buffer size"))?;
    let dyn_img = image::DynamicImage::ImageRgb8(buf);

    // Build config
    let cfg = build_config_from_kwargs(
        fast, stride, stride_x, stride_y, alpha, epsilon_palette, t_final, stag_eps, stag_limit, threads,
    );

    // Run algorithm
    let out = process_dynamic(&dyn_img, width, height, kmax, cfg)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let bytes = out.into_raw();

    // Return a PIL.Image.Image via Image.frombytes("RGB", (w,h), data)
    let pil = PyModule::import(py, "PIL.Image")?;
    let py_img = pil
        .call_method1("frombytes", ("RGB", (width, height), PyBytes::new(py, &bytes)))?;
    Ok(py_img.into())
}

#[pymodule]
fn rust_pyxeled(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transform, m)?)?;
    Ok(())
}
