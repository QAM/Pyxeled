use image::{DynamicImage, ImageBuffer, Rgb};
use pixel_convert::{default_config, process_dynamic};

#[test]
fn outputs_expected_dimensions() {
    // Create a tiny RGB image (solid color)
    let w_in = 8u32;
    let h_in = 6u32;
    let buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w_in, h_in, |_x, _y| Rgb([200, 120, 80]));
    let dyn_img = DynamicImage::ImageRgb8(buf);

    // Target output size
    let w_out = 5usize;
    let h_out = 7usize;
    let kmax = 4usize;
    let cfg = default_config(true);

    let out = process_dynamic(&dyn_img, w_out, h_out, kmax, cfg).expect("process_dynamic should succeed");
    assert_eq!(out.width(), w_out as u32);
    assert_eq!(out.height(), h_out as u32);
}
