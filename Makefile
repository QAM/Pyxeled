.PHONY: build-cli build-cli-native bench-cli bench-py build-py help setup setup-py setup-rust wheel-py

# Defaults (override on the command line)
IMG ?= examples/input_images/dog3.jpg
OUT ?= examples/output_images/bench_out.png
W   ?= 100
H   ?= 100
K   ?= 30
THREADS ?= 1 2 3 4 6 8
PYTHON ?= python3
FAST ?= True
CLI_BIN ?= pixel_convert_rust/target/release/pixel_convert

help:
	@echo "Targets:"
	@echo "  build-cli         - Build Rust CLI in release"
	@echo "  bench-cli         - Sweep threads with Rust CLI"
	@echo "  bench-py          - Sweep threads with Python transform_file()"
	@echo "  build-py          - Install Python extension (maturin develop --release)"
	@echo "  setup             - Run setup-py and setup-rust"
	@echo "  setup-py          - Create .venv and install deps via uv (or pip fallback)"
	@echo "  setup-rust        - Ensure Rust toolchain (rustup stable)"
	@echo "  wheel-py          - Build a release wheel and install"
	@echo "Variables (override): IMG OUT W H K THREADS PYTHON FAST"

build-cli:
	@echo "Building Rust CLI (pixel_convert) in release..."
	cd pixel_convert_rust && cargo build --release

build-cli-native:
	@echo "Building Rust CLI (native CPU tuning) in release..."
	cd pixel_convert_rust && RUSTFLAGS="-C target-cpu=native" cargo build --release

bench-cli: build-cli
	@echo "Benchmarking Rust CLI on $(IMG) -> $(OUT) at $(W)x$(H), K=$(K)"
	@for t in $(THREADS); do \
	  echo "-- threads=$$t --"; \
	  RUST_LOG=info $(CLI_BIN) --iter-timings --threads $$t $(IMG) $(OUT) $(W) $(H) $(K); \
	done

bench-py:
	@echo "Benchmarking Python transform_file on $(IMG) -> $(OUT) at $(W)x$(H), K=$(K)"
	@for t in $(THREADS); do \
	  echo "-- threads=$$t --"; \
	  RUST_LOG=info $(PYTHON) -c "import pixel_convert as rx; rx.transform_file('$(IMG)','$(OUT)',$(W),$(H),$(K), threads=$$t, fast=$(FAST), iter_timings=True)"; \
	done

build-py:
	@echo "Building/Installing Python extension (maturin develop --release)..."
	maturin develop -m pixel_convert/Cargo.toml --release

setup: setup-py setup-rust

setup-py:
	@echo "Setting up Python environment (.venv) with uv if available..."
	@if command -v uv >/dev/null 2>&1; then \
	  echo "uv found"; \
	  uv venv --python 3.12 || uv venv; \
	  uv sync --group dev || uv sync; \
	else \
	  echo "uv not found; falling back to venv + pip"; \
	  python3 -m venv .venv; \
	  . .venv/bin/activate; \
	  python -m pip install -U pip; \
	  python -m pip install maturin pillow loguru scikit-image scikit-learn pytest ruff; \
	fi

setup-rust:
	@echo "Ensuring Rust toolchain (stable) via rustup if available..."
	@if command -v rustup >/dev/null 2>&1; then \
	  rustup toolchain install stable; \
	  rustup default stable; \
	else \
	  echo "rustup not found. Install from https://rustup.rs then re-run 'make setup-rust'"; \
	fi

wheel-py:
	@echo "Building and installing release wheel via maturin..."
	@if ! command -v maturin >/dev/null 2>&1; then \
	  echo "maturin not found. Run 'make setup-py' first"; \
	  exit 1; \
	fi
	maturin build -m pixel_convert/Cargo.toml --release --strip
	pip install target/wheels/*.whl --force-reinstall
