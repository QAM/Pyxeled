Algorithm Description (pixel_convert)
====================================

Overview
--------

This is a Rust viersion of Pixelated Image Abstraction algorithm, which generates a pixelated/stylized rendition of an input image by optimizing a grid of “super‑pixels” and a global color palette. The method alternates between assigning input pixels to nearby super‑pixels, estimating local average colors, inferring palette color responsibilities, and refining the palette. A simulated annealing temperature T controls how strongly color distances influence assignments. The palette may expand (split) up to K_max colors.

Relationship to the Paper
-------------------------

- Reference: Gerstner, T., DeCarlo, D., Finkelstein, A., Gingold, Y., Neubert, B., “Pixelated Image Abstraction,” NPAR 2012. (Princeton Graphics Group)
- Our implementation follows the core ideas:
  - Represent the output as a coarse grid of cells (super‑pixels) with positions and colors.
  - Use an energy with a data/color term and a spatial regularization term, optimized iteratively with annealing.
  - Maintain a small global color palette and assign each cell to a palette color with temperature‑dependent soft responsibilities; refine palette by weighted averaging.
  - Apply edge‑aware smoothing (bilateral‑like) and Laplacian regularization on the grid to prevent artifacts and promote coherence.
  - Grow the palette by splitting if colors become sufficiently separated (bounded by K_max).

Compared to the paper, our code adopts simple, efficient approximations suited to a fast CPU implementation: 3×3 local search for cell selection, Lab L2 distances with a Gibbs/softmax form, and lightweight smoothing passes.

Inputs and Outputs
------------------

- Inputs:
  - `input_path`, `output_path`
  - Output resolution: `w_out × h_out`
  - Maximum number of palette clusters: `K_max`
  - Optional `--fast` flag (performance mode)
- Output: Final image written to `output_path`. Intermediate progress images are emitted every 100 iterations as `<stem>_N.png` in the same directory.

Color Space
-----------

- The algorithm operates mainly in CIE Lab for perceptual color distances.
- Loading uses `image` crate, conversion uses `palette` crate (sRGB → linear → Lab; and back for output).

Initialization
--------------

1) Convert the input image to Lab and flatten into a list of 3D vectors.
2) Compute the first principal component (PC1) via 3×3 covariance + power iteration.
3) Set `delta = 1.5 × PC1`. This direction is used for initial palette perturbation.
4) Initialize a grid of `w_out × h_out` super‑pixels positioned by sampling input coordinates proportional to output grid (nearest neighbor mapping).
5) Initialize the palette with two colors (same average color), with the second perturbed by `delta` and equal probabilities 0.5/0.5.
6) Start with a single cluster `[(0,1)]`, i.e., two palette entries in one cluster.

Core Data Structures
--------------------

- SuperPixel
  - Position `(x, y)` in input coordinates (float).
  - `sp_color`: locally averaged color (Lab) computed from assigned input pixels.
  - `palette_color`: the palette color chosen for rendering/association (argmax of responsibilities).
  - `p_s`: super‑pixel prior probability (uniform `1/N`, `N = w_out × h_out`).
  - Accumulators for fast updates: `count`, `sum_x`, `sum_y`, `sum_L`, `sum_a`, `sum_b`.

- ColorEntry (palette entry)
  - `color`: Lab color.
  - `probability`: inferred mixture weight.

- Clusters: pairs of palette indices `(i0, i1)` that may split/expand.

Main Loop (Refine → Associate → Palette Refine → Possibly Expand)
-----------------------------------------------------------------

The loop runs while `T > T_final`, with a cap of 1010 iterations and an early‑stopping criterion.

1) Super‑pixel refine (assignment + local smoothing)
   - For each input pixel `(x, y)`, search the 3×3 neighborhood of the corresponding output cell for the best super‑pixel by minimizing:
     `cost = ||Lab(x, y) − palette_color||_2 + λ · distance((x, y), (sp.x, sp.y))`, with `λ = 45·sqrt(N/M)`.
   - Assign the input pixel to the best super‑pixel by incrementally updating its accumulators.
   - After all assignments: update each super‑pixel’s `x, y` (mean of assigned coordinates) and `sp_color` (mean Lab of assigned pixels).
   - Laplacian smoothing on positions: mix each `(x, y)` with the average of its 4 neighbors (weighted 0.6 self, 0.4 neighbors).
   - Bilateral‑like color smoothing: mix each `sp_color` with a weighted average of 8 neighbors using weights `exp(−|L_i − L_j|)` (then 0.5/0.5 mix).

2) Associate (palette responsibilities)
   - For each super‑pixel, compute conditional probabilities over palette entries:
     `p_c[k] = P(color_k | sp) ∝ P(color_k) · exp(−||sp_color − color_k||_2 / T)`.
   - Normalize `p_c` and set `palette_color` to the argmax palette entry. This mirrors the effective behavior of the Python version (the cluster‑averaging path is unused there).
   - Update each palette probability as the sum over super‑pixels: `P_k = Σ_sp p_c[k] · p_s`.

   Interpretation relative to the paper:
   - This step corresponds to the E‑step of an EM‑like update on a palette mixture: responsibilities depend on color distances modulated by the current temperature T. As T cools, responsibilities harden, producing a crisp, low‑entropy assignment in the final stages, as described by the annealing strategy in the paper.

3) Palette refine
   - For each palette entry k with probability `P_k > ε` (ε = 1e−12), update its color as the responsibility‑weighted average of super‑pixel colors:
     `color_k ← Σ_sp sp_color · (p_c[k] · p_s / P_k)`.
   - Accumulate a `total_change` as the sum of Lab distances between old and new palette colors.

   Interpretation relative to the paper:
   - This is akin to the M‑step where palette colors are updated to minimize the expected color error under the current responsibilities.

4) Cooling and expansion
   - If `total_change < ε_palette` (default 1.0), cool: `T ← α·T` (default α = 0.7).
   - If `K < K_max`, attempt to expand clusters: when the two entries in a cluster are far (`||c1−c2|| > ε_cluster`, default 0.25), split by duplicating entries and halving probabilities; perturb one side by `delta`.
   - If `K ≥ K_max`, collapse each cluster pair into a single averaged entry.

   Interpretation relative to the paper:
   - Palette splitting mirrors the paper’s strategy of increasing palette capacity when justified by the data, guided by separation in color space. The temperature schedule controls when expansions occur (early when colors move quickly; later, palette stabilizes).

Stopping Criteria
-----------------

- Continue while `T > T_final` (default `T_final = 1`).
- Hard cap: 1010 iterations.
- Early stop: if `total_change < 1e−6` for 5 consecutive iterations (stagnation), stop.

Performance Optimizations (Rust)
--------------------------------

1) O(1) accumulators instead of per‑pixel sets
   - Original Python aggregated assigned pixels by storing all `(x, y)` and then averaging. The Rust port keeps only counts and sums, reducing memory and lock pressure dramatically.

2) Reduced locking and cloning
   - The hot path uses a read lock to access a super‑pixel, then locks a small mutex for its accumulators, avoiding full write locks and clones per assignment.

3) Math micro‑optimizations
   - Use `exp()` instead of `powf(E, …)`; minor but consistent speedups.

4) Parallelism
   - Assignment is multi‑threaded across all CPU cores (`num_cpus`).

5) Avoid repeated conversions
   - Input is converted to Lab once and reused.

6) Data structures and locality
   - We keep per‑cell accumulators (count and sums), not the full pixel sets, dramatically improving cache locality and reducing synchronization costs.

“Fast” Mode (`--fast`)
----------------------

The `--fast` flag trades some fidelity for speed to target ~10s processing on typical multi‑core machines.

Changes in fast mode:

- Subsampled assignment
  - Process every 2nd pixel in both X and Y during assignment (2×2 stride). This reduces the per‑iteration workload by ~4× while preserving global structure.

- More aggressive annealing and thresholds
  - `T_final = 2.0` (stop earlier in temperature schedule).
  - `α = 0.6` (cool faster), `ε_palette = 2.0` (trigger cooling earlier).
  - Stagnation early‑stop if `total_change < 1e−4` for 3 iterations.

Expected effects:

- Pros: Much faster convergence with a small quality tradeoff. Effective for previewing or generating smaller outputs.
- Cons: Slightly less smooth palette evolution and local color accuracy compared to default mode.

Complexity (High‑Level)
-----------------------

- Let `M = w_in × h_in`, `N = w_out × h_out`.
- Assignment per iteration is roughly `O(M)` with small constant due to 3×3 neighborhood checks.
- Updates (positions/colors) are `O(N)`.
- Association and palette refine are `O(N × K)`.
- With `--fast` subsampling, assignment cost drops by ~4×.

Notes and Limitations
---------------------

- Lab conversion differs slightly across libraries; results may not exactly match scikit‑image.
- The spatial weight `λ = 45·sqrt(N/M)` mirrors the Python code; tuning this constant can change the “tightness” of super‑pixel clustering.
- Further speed/quality tradeoffs are possible:
  - Expose stride, α, ε parameters as CLI flags.
  - Use `f32` instead of `f64` in core math.
  - SIMD/vectorization for distance computations.
  - Parallelize additional stages (e.g., smoothing and association) via rayon.

Additional References
---------------------

- Gerstner, T., DeCarlo, D., Finkelstein, A., Gingold, Y., Neubert, B. Pixelated Image Abstraction. Proceedings of NPAR 2012. Princeton Graphics Group.
- Bilateral filtering: Tomasi, C., Manduchi, R. Bilateral Filtering for Gray and Color Images. ICCV 1998. (Our implementation uses a lightweight 8‑neighbor approximation in Lab space.)
