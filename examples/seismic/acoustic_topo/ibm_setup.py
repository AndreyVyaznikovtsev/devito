"""
IBM precomputation: ghost/mirror geometry and Lagrange stencil coefficients.

Everything here is pure NumPy; no Devito symbols are created.
Called once during ModelTopo construction; results are stored on the model
and reused each time step by the IBM correction operator.

Coordinate convention
---------------------
Grid indices  : [ix, iz]  — Devito's (x, y) dimension order.
Physical coords: [x, z]   — same ordering.
All ghost/intercept/mirror arrays are shaped (N_ghost, 2) with columns in
this [x, z] / [ix, iz] order.
"""

import numpy as np

__all__ = ['setup_ibm', 'find_ghost_points', 'find_intercept_points',
           'find_mirror_points', 'lagrange_coefficients',
           'copy_model_params_to_ghosts']


def find_ghost_points(is_ghost):
    """
    Convert a boolean ghost mask to an integer index array.

    Parameters
    ----------
    is_ghost : ndarray bool, shape (nx_pad, nz_pad)
        is_ghost[ix, iz] is True for ghost grid points.

    Returns
    -------
    ghost_coords : ndarray int, shape (N_ghost, 2)
        Each row is [ix_pad, iz_pad] in the full padded grid.
    """
    ix, iz = np.where(is_ghost)
    return np.column_stack([ix, iz]).astype(np.int32)


def find_intercept_points(ghost_coords, topo, origin, spacing, nbl,
                          n_oversample=100):
    """
    Nearest-point search on the surface for each ghost point.

    The surface curve z = topo(x) is sampled at n_oversample × (nx − 1) + 1
    evenly-spaced x points using linear interpolation of the discrete topo
    values.  The Euclidean nearest sample to each ghost point is the intercept.

    Parameters
    ----------
    ghost_coords : (N_ghost, 2) int — [ix_pad, iz] padded indices.
    topo : (nx,) float — surface z-coordinate at physical x-column ix_phys.
    origin : (ox, oz) float — physical (not padded) grid origin.
    spacing : (dx, dz) float.
    nbl : int — PML layers in x (ix_phys = ix_pad − nbl).
    n_oversample : int — fine-sampling factor per grid cell (default 100).

    Returns
    -------
    intercept_coords : (N_ghost, 2) float64 — [x, z] physical.
    """
    dx, dz = spacing
    ox, oz = origin
    nx = len(topo)

    x_phys = ox + np.arange(nx) * dx
    n_fine = (nx - 1) * n_oversample + 1
    x_fine = np.linspace(x_phys[0], x_phys[-1], n_fine)
    z_fine = np.interp(x_fine, x_phys, topo.astype(np.float64))

    # Physical coordinates of ghost points.
    # Padded x-origin is ox - nbl*dx.
    padded_ox = ox - nbl * dx
    x_g = padded_ox + ghost_coords[:, 0].astype(np.float64) * dx
    z_g = oz         + ghost_coords[:, 1].astype(np.float64) * dz

    N = len(ghost_coords)
    intercept_coords = np.empty((N, 2), dtype=np.float64)

    # Search in chunks to avoid a (N_ghost × N_fine) memory spike.
    chunk = 128
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        xg = x_g[start:end, np.newaxis]           # (c, 1)
        zg = z_g[start:end, np.newaxis]           # (c, 1)
        d2 = (x_fine - xg)**2 + (z_fine - zg)**2  # (c, N_fine)
        j = np.argmin(d2, axis=1)                 # (c,)
        intercept_coords[start:end, 0] = x_fine[j]
        intercept_coords[start:end, 1] = z_fine[j]

    return intercept_coords


def find_mirror_points(ghost_coords, intercept_coords, origin, spacing, nbl):
    """
    Mirror point = 2 × intercept − ghost  (in physical coordinates).

    Parameters
    ----------
    ghost_coords : (N_ghost, 2) int — [ix_pad, iz].
    intercept_coords : (N_ghost, 2) float — [x, z] physical.
    origin, spacing, nbl : as above.

    Returns
    -------
    mirror_coords : (N_ghost, 2) float64 — [x, z] physical.
    """
    dx, dz = spacing
    ox, oz = origin
    padded_ox = ox - nbl * dx

    x_g = padded_ox + ghost_coords[:, 0].astype(np.float64) * dx
    z_g = oz         + ghost_coords[:, 1].astype(np.float64) * dz

    mirror_x = 2.0 * intercept_coords[:, 0] - x_g
    mirror_z = 2.0 * intercept_coords[:, 1] - z_g
    return np.column_stack([mirror_x, mirror_z])


def _lagrange_weights_1d(t, n):
    """
    1-D Lagrange weights at fractional position t within a stencil of n nodes.

    Node j sits at integer position j (j = 0, …, n−1).
    Returns L_j(t) = ∏_{k≠j} (t − k) / (j − k).
    """
    nodes = np.arange(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)
    for j in range(n):
        for k in range(n):
            if k != j:
                weights[j] *= (t - nodes[k]) / (nodes[j] - nodes[k])
    return weights


def lagrange_coefficients(mirror_coords, origin, spacing, nbl, grid_shape,
                           stencil_half):
    """
    Precompute 2-D tensor-product Lagrange coefficients for mirror-point
    interpolation.

    The stencil uses n = 2 × stencil_half nodes per dimension; the mirror
    point is positioned within the interior of the stencil so that
    interpolation (not extrapolation) is always used.

    Parameters
    ----------
    mirror_coords : (N_ghost, 2) float — [x, z] physical.
    origin : (ox, oz) physical origin.
    spacing : (dx, dz).
    nbl : int.
    grid_shape : (nx_pad, nz_pad).
    stencil_half : int — half-width; M = (2*stencil_half)² nodes total.

    Returns
    -------
    coeff_indices : (N_ghost, M, 2) int32 — padded [ix, iz] of each node.
    coeff_weights : (N_ghost, M) float64 — Lagrange weights (row sums = 1).
    """
    dx, dz = spacing
    ox, oz = origin
    nx_pad, nz_pad = grid_shape
    padded_ox = ox - nbl * dx
    n = 2 * stencil_half
    M = n * n
    N = len(mirror_coords)

    # Fractional padded-grid indices of each mirror point.
    ix_frac = (mirror_coords[:, 0] - padded_ox) / dx
    iz_frac = (mirror_coords[:, 1] - oz) / dz

    coeff_indices = np.zeros((N, M, 2), dtype=np.int32)
    coeff_weights = np.zeros((N, M), dtype=np.float64)

    for g in range(N):
        # Left edge of the n-node stencil: mirror point lands in cell
        # [stencil_half-1, stencil_half) relative to ix0.
        ix0 = int(np.floor(ix_frac[g])) - stencil_half + 1
        iz0 = int(np.floor(iz_frac[g])) - stencil_half + 1
        # Clamp so stencil stays within padded grid bounds.
        ix0 = max(0, min(ix0, nx_pad - n))
        iz0 = max(0, min(iz0, nz_pad - n))

        # Fractional position of mirror point relative to stencil origin.
        t_x = ix_frac[g] - ix0
        t_z = iz_frac[g] - iz0

        Lx = _lagrange_weights_1d(t_x, n)
        Lz = _lagrange_weights_1d(t_z, n)

        m = 0
        for jx in range(n):
            for jz in range(n):
                coeff_indices[g, m, 0] = ix0 + jx
                coeff_indices[g, m, 1] = iz0 + jz
                coeff_weights[g, m] = Lx[jx] * Lz[jz]
                m += 1

    return coeff_indices, coeff_weights


def copy_model_params_to_ghosts(b_data, m_data, ghost_coords, mirror_coords,
                                 origin, spacing, nbl):
    """
    Set b and m at ghost points to bilinearly-interpolated mirror-point values.

    Called once after b and m are initialized.  Mirror points are below the
    surface, so the bilinear stencil always reads from physical-domain cells.

    Parameters
    ----------
    b_data, m_data : ndarray float, shape (nx_pad, nz_pad) — field data arrays
        modified in-place.
    ghost_coords : (N_ghost, 2) int — [ix_pad, iz].
    mirror_coords : (N_ghost, 2) float — [x, z] physical.
    origin : (ox, oz), spacing : (dx, dz), nbl : int.
    """
    dx, dz = spacing
    ox, oz = origin
    padded_ox = ox - nbl * dx

    # Strip Devito data wrappers so element-wise scalar assignment works.
    b_data = np.asarray(b_data)
    m_data = np.asarray(m_data)
    nx_pad, nz_pad = b_data.shape

    ix_frac = (mirror_coords[:, 0] - padded_ox) / dx
    iz_frac = (mirror_coords[:, 1] - oz) / dz

    for g in range(len(ghost_coords)):
        ix_g = int(ghost_coords[g, 0])
        iz_g = int(ghost_coords[g, 1])

        ix_lo = int(np.floor(ix_frac[g]))
        iz_lo = int(np.floor(iz_frac[g]))
        ix_lo = max(0, min(ix_lo, nx_pad - 2))
        iz_lo = max(0, min(iz_lo, nz_pad - 2))

        tx = ix_frac[g] - ix_lo
        tz = iz_frac[g] - iz_lo

        def _bilinear(arr):
            return ((1 - tx) * (1 - tz) * arr[ix_lo,     iz_lo    ] +
                    tx       * (1 - tz) * arr[ix_lo + 1, iz_lo    ] +
                    (1 - tx) * tz       * arr[ix_lo,     iz_lo + 1] +
                    tx       * tz       * arr[ix_lo + 1, iz_lo + 1])

        b_data[ix_g, iz_g] = _bilinear(b_data)
        m_data[ix_g, iz_g] = _bilinear(m_data)


def setup_ibm(model, n_oversample=100):
    """
    Run all IBM precomputation steps and store results on *model*.

    After this call the model has:
    - model.ghost_coords  : (N_ghost, 2) int32  — padded [ix, iz].
    - model.mirror_coords : (N_ghost, 2) float64 — physical [x, z].
    - model.coeff_indices : (N_ghost, M, 2) int32 — Lagrange stencil nodes.
    - model.coeff_weights : (N_ghost, M) float64  — Lagrange weights.

    Parameters
    ----------
    model : ModelTopo
    n_oversample : int — surface sampling overresolution for intercepts.
    """
    is_ghost = model._build_surface_mask()
    ghost_coords = find_ghost_points(is_ghost)

    if len(ghost_coords) == 0:
        model.ghost_coords = ghost_coords
        model.mirror_coords = np.zeros((0, 2), dtype=np.float64)
        model.coeff_indices = np.zeros((0, 0, 2), dtype=np.int32)
        model.coeff_weights = np.zeros((0, 0), dtype=np.float64)
        return

    intercept_coords = find_intercept_points(
        ghost_coords, model.topo, model.origin, model.spacing, model.nbl,
        n_oversample=n_oversample,
    )
    mirror_coords = find_mirror_points(
        ghost_coords, intercept_coords, model.origin, model.spacing, model.nbl
    )
    stencil_half = model.space_order // 2
    coeff_indices, coeff_weights = lagrange_coefficients(
        mirror_coords, model.origin, model.spacing, model.nbl,
        model.grid.shape, stencil_half,
    )
    copy_model_params_to_ghosts(
        model.b.data, model.m.data,
        ghost_coords, mirror_coords,
        model.origin, model.spacing, model.nbl,
    )

    model.ghost_coords = ghost_coords
    model.mirror_coords = mirror_coords
    model.coeff_indices = coeff_indices
    model.coeff_weights = coeff_weights
