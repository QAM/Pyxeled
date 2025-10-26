# Image Combiner Usage Examples

## Basic Usage
```bash
# Combine all image groups with horizontal layout (default)
uv run combine_image.py

# Use different layouts
uv run combine_image.py --layout vertical
uv run combine_image.py --layout grid

# Process only specific group
uv run combine_image.py --group dog
uv run combine_image.py --group mountains

# Custom input/output directories
uv run combine_image.py --input-dir output_images --output-dir my_combined_images
```

## What it does:
- Automatically detects image patterns: `name_1.png`, `name_2.png`, ..., `name.png`
- Groups images by base name (dog, dog3, mountains)
- Creates two versions for each group:
  - `name_progression.png`: Shows all steps + final image
  - `name_steps_only.png`: Shows only the progression steps (no final)

## Layout Options:
- **horizontal**: Images arranged side by side (default)
- **vertical**: Images stacked vertically  
- **grid**: Images arranged in a grid pattern

## Output:
All combined images are saved to `combined_images/` directory by default.
