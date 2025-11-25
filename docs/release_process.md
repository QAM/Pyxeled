# Release Process

This project publishes a Rust crate (`pixel_convert_rust`) and a Python wheel (`pixel_convert`). Releases are driven by Git tags and GitHub Releases with Trusted Publishing to PyPI.

## Overview
- Bump versions in both crates (Rust + Python) and commit.
- Tag the version (e.g., `v0.2.0`) and push the tag.
- CI builds wheels for macOS/Windows/Linux in release mode and publishes via PyPI Trusted Publishing.

## Versioning
- Rust (Cargo): semantic versioning in `pixel_convert/Cargo.toml` under `[package].version`.
- Python (PEP 440): version in `pixel_convert/Cargo.toml` under `[package].version` (maturin reads this value).
- Keep both versions aligned (recommended).

## Steps
1) Set versions
- Edit files:
  - `pixel_convert/Cargo.toml` → `[package].version = "X.Y.Z"`
  - `pixel_convert/Cargo.toml` → `[package].version = "X.Y.Z"`
- Optional (using cargo-edit):
  - `cargo set-version -p pixel_convert X.Y.Z`
  - `cargo set-version -p pixel_convert X.Y.Z`

2) Commit
```
git add pixel_convert_rust/Cargo.toml pixel_convert/Cargo.toml
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
  - `pixel_convert/Cargo.toml` → `version = "X.Y.Zrc1"`
  - `pixel_convert/Cargo.toml` → `version = "X.Y.Z-rc.1"` (Rust conventions) or align exactly if desired.
- Tag with the same pre-release tag: `git tag -a vX.Y.Zrc1 -m "vX.Y.Zrc1"`

## Sanity Checklist
- Versions set correctly in both:
  - `pixel_convert/Cargo.toml`
  - `pixel_convert/Cargo.toml`
- Tag name matches version (e.g., `v0.2.0`).
- CI green on main.
- PyPI Trusted Publisher configured for this repository.

## Local Validation (Optional)
- Rust CLI: `make build-cli` then `./pixel_convert_rust/target/release/pixel_convert --help`
- Python wheel dev install: `make build-py` and `python -c "import pixel_convert as rx; print(rx)"`
- Benchmark: see the "Benchmark Guide" in the root README.

## Troubleshooting
- CI publish failed with auth error: Ensure PyPI Trusted Publishing is enabled for this repo and `id-token: write` permission exists in `wheels.yml`.
- Wrong version on PyPI: Verify `pixel_convert/Cargo.toml` version; maturin uses that for Python wheels.
- Mixed arch issues on macOS: Use the per-arch wheels provided by CI; avoid universal2 for performance-sensitive use.

## What to do when a release fails

- Re-run the workflow for the same tag (recommended)
  - In GitHub Actions → the failed run → “Re-run all jobs” (or “Re-run failed jobs”). This uses the same commit and tag; good for transient errors or fixed external config (e.g., PyPI Trusted Publishing).

- Re‑publish the GitHub Release
  - If your workflow triggers on “release published”, deleting and re‑publishing the Release for the same tag will trigger again.

- Manually run the workflow
  - The `wheels.yml` workflow supports `workflow_dispatch`. Use “Run workflow” and select the branch/commit with the fix. This builds from that commit, not the old tag. Only do this if you intentionally want to rebuild from the latest commit with the same version.

Important constraints (PyPI)

- You cannot overwrite files for the same version on PyPI.
  - The publish step uses `skip-existing: true`, so re-runs will skip already-uploaded wheels and only upload missing ones.
  - If you need to change the artifacts for that version, you must bump the version (e.g., `0.2.1`) and retag.
- If nothing was uploaded (auth/config error), re-running is fine — the same tag/version will publish once the issue is fixed.

Using the same tag to retrigger “push tag”

- If you must retrigger the “push tag” event, you can force-move the tag (not recommended):
  - `git tag -d vX.Y.Z`
  - `git push origin :refs/tags/vX.Y.Z`
  - `git tag -a vX.Y.Z -m "vX.Y.Z"`  # on the new commit
  - `git push origin vX.Y.Z`
- Prefer re-run or release re‑publish to avoid rewriting tags.

Recommended flow

- Fix the issue (e.g., PyPI Trusted Publisher config).
- Re-run the failed workflow for the same tag. If some wheels already exist on PyPI, they will be skipped; any missing ones will upload.
- Only bump versions if you truly need new artifacts for that release number.
