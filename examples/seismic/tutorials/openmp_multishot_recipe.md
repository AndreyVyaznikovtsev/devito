# Recipe: OpenMP multi-shot seismic modelling with Devito-generated C

## Concept

Devito generates a highly-optimised C kernel for a **single shot** (one source, one time loop).
Multi-shot survey parallelism is added manually as an outer OpenMP loop in a thin C wrapper.

```
Devito  →  ForwardSH.c      (stencil + source injection + OMP inner loops)
You     →  multishot.c      (shot loop, memory, collect results)
gcc     →  survey           (ready-to-run binary, no Python at runtime)
```

Division of work:

| Devito generates | You write |
|-----------------|-----------|
| time loop | outer shot loop |
| `#pragma omp` stencil loops | `struct dataobj` initialisation |
| SIMD vectorisation | alloc / free per shot |
| source injection + receiver interpolation | collect output |
| absorbing boundary conditions | compile command |
| all finite-difference math | |

---

## Step 1 — Generate the C kernel with Devito

```python
# generate.py
from devito import configuration
configuration['language'] = 'openmp'

import numpy as np
from examples.seismic import demo_model, AcquisitionGeometry
from examples.seismic.sh.operators import ForwardOperator

# Build a representative model (shape / spacing / space_order must match production)
model = demo_model('layers-sh',
                   shape=(400, 200), spacing=(10., 10.),
                   nbl=50, space_order=4)

src_pos = np.array([[0., 0.]], dtype=np.float32)   # position does not matter here
rec_pos = np.array([[0., 0.]], dtype=np.float32)
geo = AcquisitionGeometry(model, rec_pos, src_pos,
                          t0=0., tn=1000., src_type='Ricker', f0=0.010)

op = ForwardOperator(model, geo, space_order=4,
                     opt=('advanced', {'openmp': True}))

# Dump the C source
with open('ForwardSH.c', 'w') as f:
    f.write(str(op))

print('Written ForwardSH.c')
print('Signature: see the ForwardSH() function at the top of the file')
```

Run once:
```bash
python generate.py
```

The generated `ForwardSH.c` is self-contained — no Devito headers needed at runtime.

---

## Step 2 — Inspect the generated function signature

Open `ForwardSH.c` and look for the function header. It will look similar to:

```c
int ForwardSH(
    const float b,                           // scalar buoyancy  (constant model)
    struct dataobj *restrict damp_vec,       // absorbing boundary mask [nx × nz]
    const float mu,                          // scalar shear modulus (constant model)
    struct dataobj *restrict rec_vec,        // receiver data  [nt × nrec]
    struct dataobj *restrict rec_coords_vec, // receiver coords [nrec × 2]
    struct dataobj *restrict src_vec,        // source wavelet  [nt × 1]
    struct dataobj *restrict src_coords_vec, // source coords   [1 × 2]
    struct dataobj *restrict tau_xy_vec,     // stress field    [2 × nx_pad × nz_pad]
    struct dataobj *restrict tau_zy_vec,
    struct dataobj *restrict v_vec,          // velocity field  [2 × nx_pad × nz_pad]
    const int x_M, const int x_m,           // spatial loop bounds
    const int y_M, const int y_m,
    const float dt,
    const float o_x, const float o_y,       // grid origin [m]
    ...,
    const int nthreads,                      // OMP threads for stencil
    const int nthreads_nonaffine,            // OMP threads for sparse (src/rec)
    struct profiler *timers
)
```

For **heterogeneous models** (`mu`, `b` arrays), those scalar arguments become
`struct dataobj *restrict mu_vec` and `struct dataobj *restrict b_vec` instead.

`struct dataobj` is defined in `ForwardSH.c` itself:

```c
struct dataobj {
    void *restrict data;   // pointer to the flat float array
    int  *size;            // array of dimension sizes, e.g. {2, nx, nz}
    unsigned long nbytes;
    unsigned long *npsize;
    unsigned long *dsize;
    int  *hsize;
    int  *hofs;
    int  *oofs;
    void *dmap;            // MPI only — set to NULL for single-node use
};
```

---

## Step 3 — Write the multi-shot C wrapper

```c
/* multishot.c */
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "ForwardSH.h"   /* copy the struct dataobj + ForwardSH declaration here */

/* Helper: fill a dataobj from a pre-allocated buffer and size array */
static struct dataobj make_obj(void *data, int *size, int ndim) {
    struct dataobj o = {0};
    o.data  = data;
    o.size  = size;
    /* other fields (npsize, dsize, hsize, hofs, oofs) are only used by MPI mode */
    return o;
}

int main(int argc, char **argv) {
    /* ---- problem parameters (must match generate.py) ---- */
    const int nx = 500, nz = 300;   /* padded grid (physical + nbl on each side) */
    const int nt = 1000;
    const int nrec = 200;
    const float dt = 1.0f;          /* read from Devito's model.critical_dt */
    const float b  = 1.0f;
    const float mu = 9.0f;          /* vs^2 * rho */

    /* loop bounds (physical domain inside padding) */
    const int x_m = 0, x_M = nx - 5;
    const int y_m = 0, y_M = nz - 5;
    const float o_x = 0.f, o_y = 0.f;

    /* ---- shared, read-only arrays (same for all shots) ---- */
    float *damp_data = calloc(nx * nz, sizeof(float));
    /* ... fill damp_data from file or formula ... */
    int damp_size[] = {nx, nz};
    struct dataobj damp_obj = make_obj(damp_data, damp_size, 2);

    float *rec_coords_data = calloc(nrec * 2, sizeof(float));
    /* ... fill receiver coordinates ... */
    int rcoord_size[] = {nrec, 2};
    struct dataobj rec_coords_obj = make_obj(rec_coords_data, rcoord_size, 2);

    /* ---- shot positions (one row per shot) ---- */
    const int nshots = 50;
    float src_coords_all[nshots][2];   /* [shot][x, z] */
    for (int s = 0; s < nshots; s++) {
        src_coords_all[s][0] = 50.f + s * 80.f;   /* example: spread along x */
        src_coords_all[s][1] = 10.f;
    }

    /* ---- output: one shot record per shot ---- */
    float *all_records = calloc((size_t)nshots * nt * nrec, sizeof(float));

    /* ---- parallel shot loop ---- */
    /* Set nthreads=1: outer OMP occupies all cores across shots. */
    /* Tune N_PARALLEL so that N_PARALLEL * memory_per_shot fits in RAM. */
    const int N_PARALLEL = 4;   /* shots running simultaneously */

    #pragma omp parallel for num_threads(N_PARALLEL) schedule(dynamic)
    for (int shot = 0; shot < nshots; shot++) {

        /* Per-shot wavefields: ring buffer (2 time slots) */
        float *v_data      = calloc(2 * (size_t)nx * nz, sizeof(float));
        float *tau_xy_data = calloc(2 * (size_t)nx * nz, sizeof(float));
        float *tau_zy_data = calloc(2 * (size_t)nx * nz, sizeof(float));
        float *rec_data    = calloc((size_t)nt * nrec,    sizeof(float));
        float *src_data    = calloc((size_t)nt,            sizeof(float));
        /* ... fill src_data with Ricker wavelet ... */

        int field_size[] = {2, nx, nz};
        int rec_size[]   = {nt, nrec};
        int src_size[]   = {nt, 1};
        int scoord_size[]= {1, 2};

        struct dataobj v_obj      = make_obj(v_data,      field_size, 3);
        struct dataobj tau_xy_obj = make_obj(tau_xy_data, field_size, 3);
        struct dataobj tau_zy_obj = make_obj(tau_zy_data, field_size, 3);
        struct dataobj rec_obj    = make_obj(rec_data,    rec_size,   2);
        struct dataobj src_obj    = make_obj(src_data,    src_size,   2);
        struct dataobj scoord_obj = make_obj(src_coords_all[shot], scoord_size, 2);

        struct profiler timers = {0};

        ForwardSH(b, &damp_obj, mu,
                  &rec_obj, &rec_coords_obj,
                  &src_obj, &scoord_obj,
                  &tau_xy_obj, &tau_zy_obj, &v_obj,
                  x_M, x_m, y_M, y_m, dt, o_x, o_y,
                  /* ... remaining scalar args from signature ... */
                  1,   /* nthreads = 1: outer loop is the parallelism */
                  1,   /* nthreads_nonaffine */
                  &timers);

        /* collect result */
        memcpy(all_records + (size_t)shot * nt * nrec,
               rec_data, (size_t)nt * nrec * sizeof(float));

        free(v_data); free(tau_xy_data); free(tau_zy_data);
        free(rec_data); free(src_data);
    }

    /* ... write all_records to file ... */

    free(damp_data); free(rec_coords_data); free(all_records);
    return 0;
}
```

---

## Step 4 — Compile

```bash
gcc -O3 -march=native -fopenmp \
    ForwardSH.c multishot.c \
    -lm -o survey
```

Or with Intel compiler for better AVX-512 auto-vectorisation:
```bash
icx -O3 -xHost -qopenmp ForwardSH.c multishot.c -lm -o survey
```

---

## Step 5 — Run

```bash
# N_PARALLEL shots at a time, each single-threaded inside the stencil
OMP_PROC_BIND=close OMP_PLACES=cores ./survey
```

`OMP_PROC_BIND=close` keeps each shot's thread on a nearby core cluster,
improving cache reuse across time steps.

---

## Memory budget

Each shot needs its own wavefields:

| Array | Size |
|-------|------|
| `v`, `tau_xy`, `tau_zy` | 3 × 2 × nx × nz × 4 bytes |
| `rec` | nt × nrec × 4 bytes |
| `src` | nt × 4 bytes |

For nx=500, nz=300, nt=1000, nrec=200 (float32):
- wavefields ≈ 3 × 2 × 600 000 × 4 ≈ **14 MB per shot**
- records ≈ 200 000 × 4 ≈ 0.8 MB per shot

With N_PARALLEL=4 that is ~60 MB peak — negligible.
Scale N_PARALLEL up until you approach your RAM or LLC budget.

---

## Key rules

1. **`nthreads=1`** inside `ForwardSH` when the outer loop is OpenMP — don't double-stack threads.
2. **`damp` and model arrays are read-only** → safe to share across all shot threads.
3. **Wavefields and receiver buffers must be private** → allocate inside the shot loop.
4. **`struct dataobj.dmap = NULL`** — the MPI field is unused in single-node builds.
5. **Recompile `ForwardSH.c` whenever** the model shape, `space_order`, or `opt` level changes — the loop bounds and FD coefficients are baked in.
