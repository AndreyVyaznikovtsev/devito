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

- **`model.py`** — `GenericModel` base class + `SeismicModel` (acoustic/elastic/VTI/TTI) + `ModelSH`. All models create a `Grid` with PML padding (`nbl` layers), initialize a `damp` (or `mask`) field for absorbing boundaries, and store physical parameters as Devito `Function` objects.
- **`source.py`** — `RickerSource`, `AcquisitionGeometry`, receiver utilities.
- **`acoustic/`**, **`elastic/`**, **`tti/`**, **`vti/`**, **`sh/`**, … — Each contains `operators.py` (builds `Operator` from symbolic stencil equations) and `wavesolver.py` (thin wrapper with `forward()`/`adjoint()` methods and operator caching via `@memoized_meth`).

### SH wave solver (`examples/seismic/sh/`)

Velocity-stress staggered-grid formulation (Virieux 1984):

```
v^{n+1/2}    = v^{n-1/2}  + dt * b  * (d/dx tau_xy + d/dz tau_zy)
tau_xy^{n+1} = tau_xy^{n} + dt * mu * d/dx v^{n+1/2}
tau_zy^{n+1} = tau_zy^{n} + dt * mu * d/dz v^{n+1/2}
```

Grid staggering:
- `v`, `b` — at `NODE` (integer grid points)
- `tau_xy` — staggered in x (`staggered=(x,)`)
- `tau_zy` — staggered in z (`staggered=(z,)`)
- `mu` — at `NODE`; Devito harmonic-averages it to half-point positions automatically

`ModelSH` inherits `GenericModel` (not `SeismicModel`). Its `critical_dt` uses the elastic CFL coefficient. Boundary damping uses `bcs="mask"` (multiplied onto the update, so `damp=1` inside domain).

Working example notebooks:
- `examples/seismic/tutorials/18_sh_constant.ipynb` — homogeneous medium
- `examples/seismic/tutorials/18_sh_varying.ipynb` — heterogeneous medium
- `examples/seismic/tutorials/18_sh_snaps.ipynb` — wavefield snapshots

### Key patterns

**Staggered derivative**: `f.dx` on a field staggered at `(x+h/2)` produces a derivative at `x`; on a field at `x` produces a derivative at `x+h/2`.

**`solve(eq, target)`**: SymPy solve used to isolate `target.forward` from the PDE residual `eq`.

**`model.spacing_map`**: passed as `subs=` to `Operator` to replace symbolic spacing with numeric values.

**`@memoized_meth`**: caches compiled operators on the solver object so repeated calls don't recompile.

**Absorbing boundaries**: `damp` field multiplied onto the forward field update (`model.damp * solve(...)`). With `bcs="mask"`, `damp=1` in the interior and tapers to 0 in the PML.
