import unittest
from PIL import Image

try:
    import rust_pyxeled as rx
except Exception as e:  # pragma: no cover
    rx = None
    _import_error = e


class TestPythonTransform(unittest.TestCase):
    def setUp(self):
        if rx is None:
            self.skipTest(f"rust_pyxeled not built: {_import_error}")

    def test_returns_pil_image_with_expected_size(self):
        # Create a small RGB image in memory
        img = Image.new("RGB", (10, 8), color=(100, 150, 200))
        out = rx.transform(img, 6, 5, 3, fast=True, threads=2)
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (6, 5))


if __name__ == "__main__":
    unittest.main()
