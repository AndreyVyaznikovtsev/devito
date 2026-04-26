# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Devito Is

Devito is a Python DSL and JIT-compiler for finite-difference stencil computations. Users write equations symbolically (using SymPy), and Devito generates, compiles, and runs optimized C/OpenMP/OpenACC/CUDA code at runtime. The primary user-facing abstraction is `Operator`, which takes a list of symbolic `Eq` objects and produces a compiled stencil kernel.

## Commands

**Install (editable, all extras):**
```bash
uv pip install -e ".[extras,mpi,nvidia,tests]"
```

All commands use `uv`. Run notebooks and scripts via `uv run` or inside the uv-managed environment.

**Run all tests:**
```bash
uv run pytest tests/
```

**Run a single test file or test:**
```bash
uv run pytest tests/test_operator.py
uv run pytest tests/test_operator.py::TestOperator::test_create_explicit_dimension
```

**Lint:**
```bash
uv run ruff check devito/ examples/
```

**Format check (isort):**
```bash
uv run isort --check devito/ examples/
```

Line length is 90. The ruff config is in `pyproject.toml`.

## High-Level Architecture

### Core package (`devito/`)

- **`types/`** — Symbolic objects: `Grid`, `Dimension`, `Function`, `TimeFunction`, `SparseFunction`, `Constant`, etc. These are SymPy subclasses. `TimeFunction` with `time_order` and `space_order` is the workhorse for wave fields.
- **`finite_differences/`** — FD derivative operators (`.dx`, `.dz`, `.dt`, `.laplace`, etc.) built on top of the symbolic types. Staggered grids are expressed via `staggered=` kwarg or `NODE`.
- **`operator/`** — `Operator` class: takes a list of `Eq`s, runs the compiler pipeline (lowering → IET → C code generation → JIT compile → `.so` load).
- **`ir/`** — Intermediate Representation (IET = Iteration/Expression Tree). After symbolic lowering, equations become loop nests here. Passes in `passes/` transform the IET for parallelism, blocking, etc.
- **`passes/`** — Optimization passes over the IET (loop blocking, SIMD, OpenMP/OpenACC parallelism, etc.).
- **`arch/`** — Platform detection (CPU, GPU, ARM), compiler wrappers (gcc, icc, nvcc, etc.).
- **`symbolics/`** — SymPy extensions and simplification utilities used across the compiler.
- **`builtins/`** — Built-in operators: `initialize_function`, `gaussian_smooth`, `mmax`, `mmin`.

### Seismic examples (`examples/seismic/`)

Physics solvers live here, structured as `model.py` + per-physics directory:

- **`model.py`** — `GenericModel` base class + `SeismicModel` (acoustic/elastic/VTI/TTI). All models create a `Grid` with PML padding (`nbl` layers), initialize a `damp` field for absorbing boundaries, and store physical parameters as Devito `Function` objects.
- **`source.py`** — `RickerSource`, `AcquisitionGeometry`, receiver utilities.
- **`acoustic/`**, **`elastic/`**, **`tti/`**, **`vti/`**, **`acoustic_topo/`**, … — Each contains `operators.py` (builds `Operator` from symbolic stencil equations) and `wavesolver.py` (thin wrapper with `forward()`/`adjoint()` methods and operator caching via `@memoized_meth`).

### Acoustic topography solver (`examples/seismic/acoustic_topo/`)

Variable-density acoustic FDTD modeling with irregular free surface topography.
Method: Immersed Boundary Method (IBM) with iterative symmetric interpolation
(Li et al. 2020, J. Geophys. Eng. 17, 643–660).

**Governing equation — 2nd order, non-staggered, variable density:**

```
(1/v²) ∂²p/∂t² = ρ ∇·(1/ρ ∇p) + src
               = (b·∂p/∂x)_x + (b·∂p/∂z)_z + src
```

where `b = 1/ρ` is buoyancy. Both `v(x,z)` and `ρ(x,z)` vary spatially.
Single pressure field `p` — no staggering, no velocity components.

In Devito symbolic form:

```python
# m = 1/v² (slowness squared), b = 1/rho (buoyancy) — both Function objects
pde = m * p.dt2 - (b * p.dx).dx - (b * p.dz).dz
stencil = Eq(p.forward, solve(pde, p.forward))
```

**Why 2nd order non-staggered (not 1st order staggered):** IBM requires antisymmetric
ghost-point mirroring on a single scalar field (`p = 0` at the surface). A staggered
velocity-pressure system would need ghost conditions on three fields (`p`, `vx`, `vz`)
at different half-integer grid positions, with no clean antisymmetric BC for the velocity
components. The 2nd order pressure formulation keeps IBM straightforward.

**IBM free surface — setup (done once before the time loop):**

1. Number of ghost layers = `space_order // 2` (half the FD stencil length).
2. Identify all ghost points: integer grid points above the surface.
3. For each ghost point, find its intercept point — the closest point on the
   surface boundary (nearest-point search on a dense surface sample).
4. Mirror point = reflection of ghost point through the intercept (lands on a
   fractional grid position below the surface).
5. Compute Lagrange interpolation coefficients mapping surrounding integer-grid
   values to each mirror point. Store as precomputed arrays (done once, O(N_ghost)).
6. Copy model parameters (`b`, `m`) at ghost points from their mirror points.

**IBM free surface — per time step (interleaved with the FD update):**

1. Run the standard FD pressure update (covers the full grid including ghost points).
2. Zero out ghost point values.
3. Iterate up to 20 times (typically converges in ~5):
   a. Interpolate `p` at each mirror point using precomputed Lagrange coefficients
      applied to surrounding integer-grid values (ghost points participate as
      neighbours — this is what makes the interpolation "symmetric").
   b. Set `p_ghost = −p_mirror` (antisymmetric enforcement of `p = 0`).
4. After convergence the ghost wavefield is consistent and the next FD step proceeds.

The inner iteration is a Python loop calling a small Devito `Operator` each pass,
not a symbolic inner loop. Convergence is fast so overhead is small.

**Absorbing boundaries:** Standard `damp` sponge (`bcs="damp"`) at the bottom and
lateral edges via `SeismicModel`. The IBM ghost region (top) and the sponge (other
three sides) are geometrically disjoint and do not interact.

**Key constraints and failure modes:**
- Stable for surface dip angles up to ~73°. Very steep slopes cause multiple ghost
  points in one cell to share mirror points — leads to instability.
- Topography must not reach the lateral/bottom sponge layers.
- Use Lagrange interpolation (not bilinear — produces high-frequency artifacts;
  not extrapolation — unstable).
- Ghost point model parameters (`b`, `m`) must be set to mirror-point values
  before the time loop and do not need updating each step (model is static).

**File structure:**
- `model.py` — `ModelTopo`: inherits `GenericModel`; accepts velocity and density
  arrays plus a topography array `z0(x)`; builds `Grid`, `damp` sponge, `b`/`m`
  `Function` objects, and precomputes ghost/mirror/interpolation data structures.
- `operators.py` — `IBMOperator` (ghost correction step) + `ForwardOperator`
  (composes FD pressure update with IBM correction).
- `wavesolver.py` — `AcousticTopoSolver` with `forward()` and `@memoized_meth`
  operator caching.

### Key patterns

**Staggered derivative**: `f.dx` on a field staggered at `(x+h/2)` produces a derivative at `x`; on a field at `x` produces a derivative at `x+h/2`.

**`solve(eq, target)`**: SymPy solve used to isolate `target.forward` from the PDE residual `eq`.

**`model.spacing_map`**: passed as `subs=` to `Operator` to replace symbolic spacing with numeric values.

**`@memoized_meth`**: caches compiled operators on the solver object so repeated calls don't recompile.

**Absorbing boundaries**: `damp` field multiplied onto the forward field update (`model.damp * solve(...)`). With `bcs="mask"`, `damp=1` in the interior and tapers to 0 in the PML.
