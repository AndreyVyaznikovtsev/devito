# Implementation plan: acoustic topography solver (IBM)

Reference: Li et al. (2020), J. Geophys. Eng. 17, 643–660.
Target path: `examples/seismic/acoustic_topo/`

---

## Step 0 — understand existing infrastructure

- [ ] Read `examples/seismic/model.py`: `GenericModel.__init__`, `_initialize_damp`,
      `critical_dt`, `spacing_map`. Understand how `damp` is built and stored.
- [ ] Read `examples/seismic/acoustic/operators.py`: how `m`, `b` enter the stencil,
      how source injection and receiver interpolation are wired, operator signature.
- [ ] Read `examples/seismic/acoustic/wavesolver.py`: `@memoized_meth` pattern,
      `forward()` argument passing, operator call signature.
- [ ] Confirm variable-density acoustic stencil compiles cleanly:
      ```python
      pde = m * p.dt2 - (b * p.dx).dx - (b * p.dz).dz
      ```
      Quick standalone test in a notebook or script before touching model files.

---

## Step 1 — ModelTopo

File: `examples/seismic/acoustic_topo/model.py`

- [ ] Define `ModelTopo(GenericModel)`:
  - Constructor args: `shape`, `origin`, `spacing`, `v`, `rho`, `topo` (1D array
    `z0[ix]` giving surface depth at each x column), `nbl`, `space_order`, `bcs="damp"`.
  - Build `Grid` (same as `GenericModel`).
  - Initialize `damp` sponge at bottom and lateral edges (call `_initialize_damp` or
    equivalent). Top boundary intentionally left open (IBM handles it).
  - Create `Function` objects: `b = 1/rho` (buoyancy), `m = 1/v²` (slowness squared).
  - Store `topo` as a plain NumPy array (not a Devito Function — only needed at setup).
  - Implement `critical_dt`: standard acoustic CFL, `dt = cfl * min(spacing) / max(v)`.
    CFL coefficient 0.45 for 2nd order in time.

- [ ] Add a `_build_surface_mask(self)` helper that returns a boolean array
  `is_ghost[iz, ix]` — True for all grid points strictly above the topography.
  Use: `iz * dz + origin_z < topo[ix]` (or `<=` depending on convention — match
  Li et al.: ghost points are those on integer grid **above** the surface).

---

## Step 2 — ghost/mirror precomputation

File: `examples/seismic/acoustic_topo/ibm_setup.py` (pure NumPy, no Devito)

This module is called once during `ModelTopo.__init__` and produces arrays that
are stored on the model and consumed by the IBM operator each time step.

- [ ] `find_ghost_points(topo, origin, spacing, space_order)` →
  `ghost_coords`: array of shape `(N_ghost, 2)` — integer `(iz, ix)` indices of
  all ghost points (all layers, not just the first).
  Number of ghost layers = `space_order // 2`.

- [ ] `find_intercept_points(ghost_coords, topo, origin, spacing)` →
  `intercept_coords`: array `(N_ghost, 2)` — physical `(z, x)` coordinates of the
  closest point on the surface for each ghost point. Dense-sample the surface curve
  and take the nearest sample as the intercept (one-time cost, accuracy sufficient
  at grid scale).

- [ ] `find_mirror_points(ghost_coords, intercept_coords, origin, spacing)` →
  `mirror_coords`: array `(N_ghost, 2)` — physical `(z, x)` of each mirror point.
  Formula: `mirror = 2 * intercept − ghost` (in physical coordinates).

- [ ] `lagrange_coefficients(mirror_coords, origin, spacing, stencil_half)` →
  `coeff_indices`: `(N_ghost, M, 2)` integer indices of the M stencil neighbours,
  `coeff_weights`: `(N_ghost, M)` float Lagrange weights.
  Use a 2D tensor-product Lagrange stencil of width `stencil_half` on each side
  (same order as the FD scheme). M = `(2*stencil_half)²`.

- [ ] `copy_model_params_to_ghosts(b_data, m_data, ghost_coords, mirror_coords, ...)`:
  For each ghost point, set `b[iz_ghost, ix_ghost]` and `m[iz_ghost, ix_ghost]` to
  the bilinearly interpolated value at the mirror point. Called once after `b`/`m`
  are initialized.

---

## Step 3 — IBM correction operator

File: `examples/seismic/acoustic_topo/operators.py`

The IBM correction updates ghost point values from the current pressure field.
It is called in a Python loop (up to 20 iterations) inside the time loop.

- [ ] Store `coeff_indices` and `coeff_weights` as `SparseFunction` or plain NumPy
  arrays attached to a custom Devito `Function`. Simplest approach: store them as
  NumPy arrays on the model and build an explicit index-based update expression.

- [ ] `build_ibm_operator(model, p)` → `Operator`:
  For each ghost point `g`:
  ```
  p_mirror = sum_k( coeff_weights[g,k] * p[coeff_indices[g,k]] )
  p[ghost_coords[g]] = -p_mirror
  ```
  Options for expressing this in Devito:
  - **Option A (preferred):** Use `SparseFunction` injection/interpolation machinery.
    Define a `SparseFunction` with coordinates at ghost points; use custom
    `interpolate`/`inject` expressions to read the weighted sum and write the
    negated value back.
  - **Option B (fallback):** Express as a plain Python loop over ghost points calling
    NumPy indexing on `p.data`. Simple, zero Devito involvement, acceptable if
    N_ghost is small relative to the full grid (it always is — ghost points are only
    near the surface).
  Start with Option B to validate correctness, then profile before optimizing.

- [ ] `ibm_step(p, model, n_iter=20)`: Python function that calls the IBM operator
  (or NumPy fallback) `n_iter` times in a loop. Checks for convergence optionally
  (L∞ norm change < tol) to exit early.

---

## Step 4 — forward operator

File: `examples/seismic/acoustic_topo/operators.py`

- [ ] `ForwardOperator(model, geometry, space_order, save)`:
  Builds the main pressure update `Operator` using the variable-density stencil:
  ```python
  pde = model.m * p.dt2 - (model.b * p.dx).dx - (model.b * p.dz).dz
  stencil = Eq(p.forward, model.damp * solve(pde, p.forward))
  ```
  Source injection and receiver interpolation follow the same pattern as
  `examples/seismic/acoustic/operators.py`.

- [ ] Confirm `model.damp` is 1.0 everywhere in the interior and tapers smoothly
  at lateral/bottom edges. The ghost region at the top must have `damp = 1` (not
  attenuated) so IBM sets the correct values each step without being overridden.

---

## Step 5 — wavesolver

File: `examples/seismic/acoustic_topo/wavesolver.py`

- [ ] `AcousticTopoSolver`:
  - `__init__(self, model, geometry, space_order=4)`: store model + geometry.
  - `@memoized_meth forward(self, src, rec, u, vp, rho, save, **kwargs)`:
    1. Call `self._op_fwd` (cached `ForwardOperator`).
    2. The time loop is **manual** (unlike standard Devito operators where time
       is internal): iterate over time steps, call `op_fwd.apply(time_M=1, ...)` for
       one step, then call `ibm_step(p, model)` for the IBM correction.

    **Important:** Devito's `Operator.apply()` with `time_M=1` advances one step.
    The IBM correction must happen after each FD step, before the next. This requires
    exposing the per-step time loop at the Python level rather than letting Devito
    run the full time loop internally.

    Alternative: inject the IBM correction directly into the `Operator` equation list
    using conditional `Eq` expressions guarded by the ghost-point mask. Evaluate
    feasibility — if ghost count is small and Devito can express the interpolation
    stencil, this avoids the Python time loop overhead. Otherwise use the manual loop.

---

## Step 6 — tests

File: `tests/test_acoustic_topo.py`

- [ ] **Flat surface test:** Set `topo` to a constant depth (flat surface at z=0).
  IBM should reduce exactly to the standard acoustic solver with a reflecting top
  boundary. Compare pressure snapshots between IBM solver and standard acoustic
  solver with a zero-BC top — should match to machine precision (or FD order accuracy).

- [ ] **Gaussian hill — homogeneous medium:** Reproduce Li et al. Fig. 7c.
  `z0(x) = z_flat - A * exp(-(x - x0)²/σ²)`. Use a homogeneous velocity and density.
  Verify no staircase diffractions (compare against coarse vacuum method result).

- [ ] **Sine topography — stability test:** Reproduce Li et al. Fig. 12 series.
  Run 4 models with increasing surface dip (a=100,150,200,300). Confirm stable for
  a=100 (73° max dip) and that instability appears for very steep a values.

- [ ] **Variable density:** Two-layer model with density contrast. Verify that the
  variable-density stencil produces the correct reflection amplitude at the interface
  (compare against analytical reflection coefficient `R = (ρ2*v2 - ρ1*v1)/(ρ2*v2 + ρ1*v1)`).

- [ ] **IBM convergence:** For a single time step, plot ghost-point values vs. iteration
  number (reproduce Li et al. Fig. 6). Confirm convergence within 20 iterations.

---

## Step 7 — example notebook

File: `examples/seismic/tutorials/19_acoustic_topo_constant.ipynb`

- [ ] Homogeneous medium, Gaussian hill topography.
- [ ] Show: model setup, source/receiver geometry on the surface, pressure snapshots,
  shot record, comparison with vacuum method to illustrate absence of diffractions.

---

## Notes and open questions

- **Ghost point mask and `damp`:** Ensure the `damp` field is constructed with
  `nbl` padding only at bottom/lateral edges, not the top. Check `_initialize_damp`
  in `GenericModel` — may need a `no_top` flag or manual zeroing of the top sponge.

- **Time loop exposure:** Devito operators normally run the full time loop internally.
  Splitting into per-step calls (`time_M=1` repeatedly) adds Python overhead. Profile
  on a medium-sized model (500×500, 1000 steps) to decide if the IBM inner loop is
  the bottleneck or if the per-step Python call is. If Python overhead dominates,
  explore inlining IBM as a `Conditional` expression in the operator.

- **Adjoint / FWI:** The IBM operator is not self-adjoint as written. For gradient
  computation the adjoint of the ghost-correction step needs to be derived. Out of
  scope for the forward-modeling phase.
