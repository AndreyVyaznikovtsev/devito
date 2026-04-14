# Devito Operator Call Mechanism: Python → C

## Overview

When you call `op.apply(...)`, Devito loads a JIT-compiled C shared library and
calls its entry function via **ctypes**. This document traces the full execution
path and describes everything needed to call the generated C code from other software.

---

## 1. Compilation (`Operator.__init__`)

When you create `op = ForwardOperator(...)`, Devito:

- Generates C source code symbolically from the stencil expressions
- JIT-compiles it to a `.so` via `gcc`
- Caches the result in `/tmp/devito-jitcache-uid1000/`

---

## 2. Loading the `.so` (`operator.py:826` — `cfunction` property)

```python
self._lib = numpy.ctypeslib.load_library(soname, '.')  # dlopen the .so
self._cfunction = getattr(self._lib, self.name)        # get function pointer
self._cfunction.argtypes = [p._C_ctype for p in self.parameters]
```

It is plain **ctypes** — the `.so` is loaded with `numpy.ctypeslib.load_library`
(which wraps `ctypes.CDLL`).

---

## 3. The `dataobj` struct (`dense.py:653`)

Every array argument (`v`, `tau_xy`, `damp`, etc.) is wrapped in this struct:

```c
struct dataobj {
    void *restrict data;   // pointer to the numpy array's raw memory
    int  *size;            // shape array [dim0, dim1, ...]
    unsigned long  nbytes; // total bytes
    unsigned long *npsize; // unpadded (no halo) sizes
    unsigned long *dsize;  // domain sizes
    int  *hsize;           // halo extents per side
    int  *hofs;            // halo offsets
    int  *oofs;            // owned-region offsets (MPI, NULL for serial)
    void *dmap;            // MPI distribution map (NULL for serial)
};
```

Python fills this in `_C_make_dataobj()` (`dense.py:675`): it copies the numpy
array's `ctypes.data_as()` pointer and shape into the struct fields via `byref()`.

**Important:** The `data` pointer must point to a **padded buffer** — the numpy
array includes `nbl` absorbing-boundary layers on each side, so its shape is
`(nx + 2*nbl, ny + 2*nbl)`, not the physical domain size. On top of that, the
generated C code adds offsets of `space_order/2` (e.g. `x + 4` for
`space_order=4`) to account for ghost cells.

---

## 4. The `profiler` struct

The `struct profiler` in the generated C (with fields `section0`, `section1`,
etc.) **is** a real ctypes struct created on the Python side. The `START()`/`STOP()`
macros in the C code write wall-clock timings into it. You can zero it out if you
do not need timings.

---

## 5. Actual call (`operator.py:1001`)

```python
arg_values = [args[p.name] for p in self.parameters]
retval = cfunction(*arg_values)
```

ctypes marshals everything:
- `dataobj*` pointers pass through as-is
- Python `float`/`int` scalars become `c_float`/`c_int`

---

## 6. Generated C function signature (`ForwardSH`)

```c
int ForwardSH(
    const float             b,
    struct dataobj *restrict damp_vec,
    const float             mu,
    struct dataobj *restrict rec_vec,
    struct dataobj *restrict rec_coords_vec,
    struct dataobj *restrict src_vec,
    struct dataobj *restrict src_coords_vec,
    struct dataobj *restrict tau_xy_vec,
    struct dataobj *restrict tau_zy_vec,
    struct dataobj *restrict v_vec,
    const int  x_M,  const int  x_m,
    const int  y_M,  const int  y_m,
    const float dt,
    const float o_x, const float o_y,
    const int  p_rec_M, const int  p_rec_m,
    const int  p_src_M, const int  p_src_m,
    const int  time_M,  const int  time_m,
    struct profiler *timers
);
```

---

## 7. Key files

| File | Purpose | Key lines |
|------|---------|-----------|
| `devito/operator/operator.py` | `apply()` and `cfunction` property | 925, 826 |
| `devito/arch/compiler.py` | `.so` loading via `numpy.ctypeslib` | 278 |
| `devito/types/dense.py` | `dataobj` struct definition and construction | 653, 675 |
| `devito/types/basic.py` | Scalar `_C_ctype` mapping | 495 |
| `devito/types/dimension.py` | Dimension argument (iteration bounds) processing | 266 |
| `devito/tools/dtypes_lowering.py` | `dtype → ctypes` conversion | 138 |
| `devito/ir/iet/visitors.py` | C struct/signature code generation | 253, 379 |
| `devito/operator/profiling.py` | Python-side profiler | 30+ |

---

## 8. Porting the C code to another software

To call the generated `.c` file from another codebase:

1. **Compile** the `.c` file — it is self-contained (needs `math.h`, `stdlib.h`,
   optionally OpenMP via `-fopenmp`).

2. **Allocate padded float buffers** of shape `(nx + 2*nbl + ghost, ny + 2*nbl + ghost)`
   for each field (`v`, `tau_xy`, `tau_zy`, `damp`, etc.).

3. **Fill a `dataobj` struct** for each array: set `data` to your buffer pointer
   and `size[]` to the padded dimensions.

4. **Pass scalars** (`dt`, `mu`, `b`) directly as `float`.

5. **Pass iteration bounds** (`x_m`, `x_M`, `y_m`, `y_M`, `time_m`, `time_M`)
   as `int` — these exclude the ghost/halo padding and determine the active loop
   range.

6. **Pass origin offsets** `o_x`, `o_y` — these are the physical coordinates of
   grid point (0, 0), used to convert source/receiver coordinates to grid indices.

7. **Pass source/receiver arrays** (`src_vec`, `rec_vec`) as `dataobj` structs
   wrapping 2-D float arrays of shape `(nt, n_src)` / `(nt, n_rec)`, and
   coordinate arrays of shape `(n_src, 2)` / `(n_rec, 2)`.

8. **Pass a zeroed `profiler` struct** if you do not need timing information.
