# Release Process

This project publishes a Rust crate (`pixel_convert`) and a Python wheel (`pixel_convert_py`). Releases are driven by Git tags and GitHub Releases with Trusted Publishing to PyPI.

## Overview
- Bump versions in both crates (Rust + Python) and commit.
- Tag the version (e.g., `v0.2.0`) and push the tag.
- CI builds wheels for macOS/Windows/Linux in release mode and publishes via PyPI Trusted Publishing.

## Versioning
- Rust (Cargo): semantic versioning in `pixel_convert/Cargo.toml` under `[package].version`.
- Python (PEP 440): version in `pixel_convert_py/Cargo.toml` under `[package].version` (maturin reads this value).
- Keep both versions aligned (recommended).

## Steps
1) Set versions
- Edit files:
  - `pixel_convert/Cargo.toml` → `[package].version = "X.Y.Z"`
  - `pixel_convert_py/Cargo.toml` → `[package].version = "X.Y.Z"`
- Optional (using cargo-edit):
  - `cargo set-version -p pixel_convert X.Y.Z`
  - `cargo set-version -p pixel_convert_py X.Y.Z`

2) Commit
```
git add pixel_convert/Cargo.toml pixel_convert_py/Cargo.toml
git commit -m "chore(release): bump to X.Y.Z"
```

3) Tag
```
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```
Alternatively, create a GitHub Release for `vX.Y.Z`.

## CI: Build and Publish
- Workflow: `.github/workflows/wheels.yml`
- Matrix builds release wheels with:
  - ThinLTO + `codegen-units=1` for speed
  - Per-arch macOS wheels (`aarch64-apple-darwin`, `x86_64-apple-darwin`)
  - Manylinux2014 on Linux; Windows wheels
- Publish job uses PyPI Trusted Publishing (OIDC). No API token is required.
  - Ensure your PyPI project is configured with a Trusted Publisher for this repo.

## Pre-releases (RCs)
- Use PEP 440 pre-release versions in Python crate (and mirror for Rust):
  - `pixel_convert_py/Cargo.toml` → `version = "X.Y.Zrc1"`
  - `pixel_convert/Cargo.toml` → `version = "X.Y.Z-rc.1"` (Rust conventions) or align exactly if desired.
- Tag with the same pre-release tag: `git tag -a vX.Y.Zrc1 -m "vX.Y.Zrc1"`

## Sanity Checklist
- Versions set correctly in both:
  - `pixel_convert/Cargo.toml`
  - `pixel_convert_py/Cargo.toml`
- Tag name matches version (e.g., `v0.2.0`).
- CI green on main.
- PyPI Trusted Publisher configured for this repository.

## Local Validation (Optional)
- Rust CLI: `make build-cli` then `./pixel_convert/target/release/pixel_convert --help`
- Python wheel dev install: `make build-py` and `python -c "import pixel_convert as rx; print(rx)"`
- Benchmark: see the "Benchmark Guide" in the root README.

## Troubleshooting
- CI publish failed with auth error: Ensure PyPI Trusted Publishing is enabled for this repo and `id-token: write` permission exists in `wheels.yml`.
- Wrong version on PyPI: Verify `pixel_convert_py/Cargo.toml` version; maturin uses that for Python wheels.
- Mixed arch issues on macOS: Use the per-arch wheels provided by CI; avoid universal2 for performance-sensitive use.

