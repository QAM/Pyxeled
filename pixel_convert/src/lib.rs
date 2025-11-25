use pyo3::prelude::*;
use pyo3::types::PyBytes;

use ::pixel_convert_rust::{
    default_config, process, process_dynamic, Config, Params,
    map_file_to_palette, map_image_to_palette, ColorDistanceAlgorithm,
    named_palette, named_palette_vec,
};

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
    iter_timings: bool,
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
    cfg.iter_timings = iter_timings;
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
    iter_timings = false,
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
    iter_timings: bool,
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
        fast, stride, stride_x, stride_y, alpha, epsilon_palette, t_final, stag_eps, stag_limit, threads, iter_timings,
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
fn pixel_convert(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transform, m)?)?;
    m.add_function(wrap_pyfunction!(transform_file, m)?)?;
    m.add_function(wrap_pyfunction!(map_to_colors_file, m)?)?;
    m.add_function(wrap_pyfunction!(map_to_colors_image, m)?)?;
    m.add_function(wrap_pyfunction!(map_to_named_palette_image, m)?)?;
    m.add_function(wrap_pyfunction!(map_to_named_palette_file, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    input_path,
    output_path,
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
    iter_timings = false,
))]
pub fn transform_file(
    _py: Python<'_>,
    input_path: &str,
    output_path: &str,
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
    iter_timings: bool,
) -> PyResult<()> {
    let cfg = build_config_from_kwargs(
        fast, stride, stride_x, stride_y, alpha, epsilon_palette, t_final, stag_eps, stag_limit, threads, iter_timings,
    );
    let params = Params {
        in_image_name: input_path.to_string(),
        out_image_name: output_path.to_string(),
        w_out: width,
        h_out: height,
        k_max: kmax,
        config: cfg,
    };
    process(params)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}


/// Map each pixel of a PIL image to the nearest color in `colors` using the chosen `algorithm`.
/// Returns a new PIL Image in RGB mode.
#[pyfunction]
#[pyo3(signature = (image, colors, algorithm = "rgb"))]
pub fn map_to_colors_image(
    py: Python<'_>,
    image: &PyAny,
    colors: Vec<(u8, u8, u8)>,
    algorithm: &str,
) -> PyResult<PyObject> {
    if colors.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("colors list must not be empty"));
    }

    let img_rgb = image.call_method1("convert", ("RGB",))?;
    let (w, h): (u32, u32) = img_rgb.getattr("size")?.extract()?;
    let data: Vec<u8> = img_rgb.call_method0("tobytes")?.extract()?;

    let buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(w, h, data)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid image buffer size"))?;

    let algo = ColorDistanceAlgorithm::from_str(algorithm);
    let out = map_image_to_palette(&buf, &colors, algo);
    let bytes = out.into_raw();
    let pil = PyModule::import(py, "PIL.Image")?;
    let py_img = pil
        .call_method1("frombytes", ("RGB", (w, h), PyBytes::new(py, &bytes)))?;
    Ok(py_img.into())
}

/// Map each pixel of an input file to the nearest color in `colors` using the chosen `algorithm`.
#[pyfunction]
#[pyo3(signature = (input_path, output_path, colors, algorithm = "rgb"))]
pub fn map_to_colors_file(
    _py: Python<'_>,
    input_path: &str,
    output_path: &str,
    colors: Vec<(u8, u8, u8)>,
    algorithm: &str,
) -> PyResult<()> {
    if colors.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("colors list must not be empty"));
    }
    let algo = ColorDistanceAlgorithm::from_str(algorithm);
    map_file_to_palette(input_path, output_path, &colors, algo)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(())
}

/// Map to a named, built-in palette (e.g., "dmc"). Returns a new PIL Image.
#[pyfunction]
#[pyo3(signature = (image, palette_name, algorithm = "rgb"))]
pub fn map_to_named_palette_image(
    py: Python<'_>,
    image: &PyAny,
    palette_name: &str,
    algorithm: &str,
) -> PyResult<PyObject> {
    let colors = named_palette_vec(palette_name)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unknown palette: {}", palette_name)))?;
    if colors.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("named palette is empty (not embedded)"));
    }
    let img_rgb = image.call_method1("convert", ("RGB",))?;
    let (w, h): (u32, u32) = img_rgb.getattr("size")?.extract()?;
    let data: Vec<u8> = img_rgb.call_method0("tobytes")?.extract()?;
    let buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(w, h, data)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid image buffer size"))?;
    let algo = ColorDistanceAlgorithm::from_str(algorithm);
    let out = map_image_to_palette(&buf, &colors, algo);
    let bytes = out.into_raw();
    let pil = PyModule::import(py, "PIL.Image")?;
    let py_img = pil
        .call_method1("frombytes", ("RGB", (w, h), PyBytes::new(py, &bytes)))?;
    Ok(py_img.into())
}

/// Map file-to-file using a named, built-in palette (e.g., "dmc").
#[pyfunction]
#[pyo3(signature = (input_path, output_path, palette_name, algorithm = "rgb"))]
pub fn map_to_named_palette_file(
    _py: Python<'_>,
    input_path: &str,
    output_path: &str,
    palette_name: &str,
    algorithm: &str,
) -> PyResult<()> {
    let colors = named_palette_vec(palette_name)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unknown palette: {}", palette_name)))?;
    if colors.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("named palette is empty (not embedded)"));
    }
    let algo = ColorDistanceAlgorithm::from_str(algorithm);
    map_file_to_palette(input_path, output_path, &colors, algo)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(())
}
