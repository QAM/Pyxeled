use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use pixel_convert_rust::{map_image_to_palette, named_palette, ColorDistanceAlgorithm};

fn tiny_test_image() -> RgbImage {
    // Two pixels matching two DMC entries exactly, so mapping should preserve them.
    let c1 = Rgb([159, 56, 69]); // from DMC list
    let c2 = Rgb([242, 188, 197]);
    ImageBuffer::from_fn(2, 1, |x, _y| if x == 0 { c1 } else { c2 })
}

#[test]
fn map_with_dmc_rgb_preserves_exact_matches() {
    let img = tiny_test_image();
    let dmc = named_palette("dmc").expect("dmc palette should be available");
    assert!(dmc.len() > 0);
    let out = map_image_to_palette(&img, dmc, ColorDistanceAlgorithm::Rgb);
    assert_eq!(out.get_pixel(0, 0).0, [159, 56, 69]);
    assert_eq!(out.get_pixel(1, 0).0, [242, 188, 197]);
}

#[test]
fn map_with_dmc_lab_preserves_exact_matches() {
    let img = tiny_test_image();
    let dmc = named_palette("dmc").expect("dmc palette should be available");
    let out = map_image_to_palette(&img, dmc, ColorDistanceAlgorithm::Lab);
    assert_eq!(out.get_pixel(0, 0).0, [159, 56, 69]);
    assert_eq!(out.get_pixel(1, 0).0, [242, 188, 197]);
}

#[test]
fn map_with_dmc_ciede_preserves_exact_matches() {
    let img = tiny_test_image();
    let dmc = named_palette("dmc").expect("dmc palette should be available");
    let out = map_image_to_palette(&img, dmc, ColorDistanceAlgorithm::Ciede2000);
    assert_eq!(out.get_pixel(0, 0).0, [159, 56, 69]);
    assert_eq!(out.get_pixel(1, 0).0, [242, 188, 197]);
}
