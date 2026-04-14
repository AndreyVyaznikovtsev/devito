# Notes: Standalone SH-wave Application Without Devito

## Goal

Use Devito-generated C code for SH-wave modeling in a standalone Python GUI
application that works without Devito as a runtime dependency.

---

## Core Idea

The generated C code (e.g. `ForwardSH`) is self-contained. Devito is only needed
to generate and JIT-compile it. At runtime, the application just needs to:

1. Load the compiled `.so` / `.dll` via `ctypes.CDLL`
2. Wrap numpy arrays into `dataobj` structs
3. Call the C function

The Python wrapper for this is ~80-100 lines — no Devito needed at runtime.

---

## Minimal Python Runtime Wrapper

- Define the `dataobj` ctypes struct (~20 lines)
- Define the `profiler` ctypes struct (~5 lines, can be zeroed if timing not needed)
- Write a helper to pack a numpy array into a `dataobj` (pointer + shape + metadata)
- Load the `.so`/`.dll` with `ctypes.CDLL`, set `argtypes`, call the function

---

## Things to Handle Without Devito

| Component | Notes |
|-----------|-------|
| `dataobj` struct wrapping | Reimplement with ctypes (~20 lines) |
| `profiler` struct | Zero it out — only needed for timing |
| `damp` array | Compute once with Devito, save as `.npy`; or reimplement cosine taper (~10 lines numpy) |
| Ricker source wavelet | Pure numpy, trivial to reimplement |
| Source/receiver coordinates | Pass directly as float arrays — interpolation is inside the C code |
| `mu` / `b` arrays | For heterogeneous models these are `dataobj*` (spatially varying fields) |

---

## Model-Specific Constraints

The generated C code is tied to:

- **Space order** — FD coefficients are hardcoded (e.g. `1.38888896e-3F` for `space_order=4`)
- **`mu`/`b` signature** — scalar (`const float`) for homogeneous, `struct dataobj*` for heterogeneous
- **Grid size** — flexible at runtime via iteration bound arguments (`x_m`, `x_M`, etc.)

To change space order or switch between homogeneous/heterogeneous: regenerate C code with Devito.

---

## Architecture Recommendation

> **Devito as a build-time code generator, not a runtime dependency.**

- Generate the C code on Linux (or in CI)
- Commit the `.c` file to the repository
- Compile it for each target platform in a build/CI step
- Ship the compiled binary with the application

---

## Windows Compatibility

The generated C code is **not MSVC-compatible** out of the box. Issues:

| Problem | Detail |
|---------|--------|
| `#pragma omp simd aligned(...)` | GCC-specific pragma |
| `__attribute__ ((aligned (64)))` | GCC/Clang attribute, not MSVC |
| `#include "sys/time.h"` | POSIX header, not available on Windows |

The `sys/time.h` issue is the most immediate blocker — it comes from the
`START`/`STOP` profiling macros, which can simply be stripped out since the
profiler is optional.

### Options for Windows

| Option | Description |
|--------|-------------|
| **Pre-compiled `.dll`** | Compile once using MinGW-w64 or Clang on Windows; bundle with app. Recommended for GUI. |
| **MinGW at runtime** | App compiles C on startup if MinGW is present. Requires user toolchain. |
| **Python extension wheel** | Wrap in a `.pyd` via `setuptools`/`scikit-build`; distribute as a wheel. |
| **WSL2** | Run solver in WSL, GUI on Windows, communicate over socket/file. Avoids cross-compilation entirely. |

**Recommended for a GUI app:** pre-compiled `.dll` built on a Windows CI runner
(e.g. GitHub Actions), shipped with the installer.
