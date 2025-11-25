#!/usr/bin/env python3
"""
Image Combiner for Pixel-convert Output
Combines images with pattern name_1.png, name_2.png, ..., name.png
to show the progression from step 1 to final image.
"""

import os
import re
from PIL import Image
from collections import defaultdict
import argparse


def get_image_groups(output_dir):
    """Group images by base name and sort by step number."""
    groups = defaultdict(list)

    # Pattern to match: name_number.png or name.png
    pattern = re.compile(r"^(.+?)(?:_(\d+))?\.png$")

    for filename in os.listdir(output_dir):
        if filename.endswith(".png"):
            match = pattern.match(filename)
            if match:
                base_name = match.group(1)
                step_num = match.group(2)

                if step_num:
                    # This is a step image (e.g., dog_1.png)
                    groups[base_name].append((int(step_num), filename))
                else:
                    # This is the final image (e.g., dog.png)
                    groups[base_name].append(
                        (float("inf"), filename)
                    )  # Final image gets highest priority

    # Sort each group by step number
    for base_name in groups:
        groups[base_name].sort(key=lambda x: x[0])

    return groups


def combine_images(image_paths, output_path, layout="horizontal"):
    """
    Combine multiple images into one showing progression.

    Args:
        image_paths: List of image file paths
        output_path: Output file path
        layout: 'horizontal', 'vertical', or 'grid'
    """
    if not image_paths:
        print("No images to combine!")
        return

    # Load all images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
            print(f"Loaded: {os.path.basename(path)} ({img.size})")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    if not images:
        print("No valid images loaded!")
        return

    if layout == "horizontal":
        # Arrange horizontally
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        combined = Image.new("RGB", (total_width, max_height), "white")
        x_offset = 0

        for i, img in enumerate(images):
            # Center vertically if needed
            y_offset = (max_height - img.height) // 2
            combined.paste(img, (x_offset, y_offset))
            x_offset += img.width

    elif layout == "vertical":
        # Arrange vertically
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)

        combined = Image.new("RGB", (max_width, total_height), "white")
        y_offset = 0

        for i, img in enumerate(images):
            # Center horizontally if needed
            x_offset = (max_width - img.width) // 2
            combined.paste(img, (x_offset, y_offset))
            y_offset += img.height

    elif layout == "grid":
        # Arrange in a grid (try to make it roughly square)
        num_images = len(images)
        cols = int(num_images**0.5) + (1 if num_images**0.5 % 1 > 0 else 0)
        rows = (num_images + cols - 1) // cols

        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        combined = Image.new("RGB", (cols * max_width, rows * max_height), "white")

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols

            x_offset = col * max_width + (max_width - img.width) // 2
            y_offset = row * max_height + (max_height - img.height) // 2

            combined.paste(img, (x_offset, y_offset))

    # Save the combined image
    combined.save(output_path)
    print(f"Combined image saved: {output_path}")
    print(f"Final size: {combined.size}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine Pixel-convert output images to show progression"
    )
    parser.add_argument(
        "--input-dir",
        default="output_images",
        help="Input directory containing images (default: output_images)",
    )
    parser.add_argument(
        "--output-dir",
        default="combined_images",
        help="Output directory for combined images (default: combined_images)",
    )
    parser.add_argument(
        "--layout",
        choices=["horizontal", "vertical", "grid"],
        default="horizontal",
        help="Layout for combining images",
    )
    parser.add_argument("--group", help='Process only specific group (e.g., "dog")')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get image groups
    groups = get_image_groups(args.input_dir)

    if not groups:
        print(f"No image groups found in {args.input_dir}")
        return

    print(f"Found {len(groups)} image groups:")
    for base_name, images in groups.items():
        step_images = [img for step, img in images if step != float("inf")]
        final_image = [img for step, img in images if step == float("inf")]
        print(
            f"  {base_name}: {len(step_images)} steps + {'final' if final_image else 'no final'}"
        )

    # Process groups
    for base_name, images in groups.items():
        if args.group and base_name != args.group:
            continue

        print(f"\nProcessing group: {base_name}")

        # Get full paths
        image_paths = [
            os.path.join(args.input_dir, filename) for step, filename in images
        ]

        # Create output filename
        output_filename = f"{base_name}_progression.png"
        output_path = os.path.join(args.output_dir, output_filename)

        # Combine images
        combine_images(image_paths, output_path, args.layout)

    print(f"\nAll combined images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
