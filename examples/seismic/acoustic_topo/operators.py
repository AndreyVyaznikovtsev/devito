import numpy as np
from devito import Eq, Operator, TimeFunction, solve

__all__ = ['ibm_step', 'ForwardOperator']


def ibm_step(p_2d, model, n_iter=20, tol=1e-6):
    """
    Apply the IBM ghost-point correction to a single 2-D pressure slice.

    After the FD update the ghost points contain unphysical values.
    This function enforces p = 0 at the free surface by iterating:

        p_ghost ← − Lagrange_interp(p, mirror_point)

    until convergence (typically 3–5 iterations).

    Parameters
    ----------
    p_2d : ndarray, shape (nx_pad, nz_pad)
        One time-buffer slice of the pressure field, modified in-place.
    model : ModelTopo
        Provides ghost_coords, coeff_indices, coeff_weights.
    n_iter : int
        Maximum iterations (default 20).
    tol : float
        L∞ convergence tolerance on ghost-point change (default 1e-6).
    """
    gc = model.ghost_coords   # (N, 2)  [ix, iz]
    ci = model.coeff_indices  # (N, M, 2)
    cw = model.coeff_weights  # (N, M)

    if len(gc) == 0:
        return

    # Strip the Devito data wrapper so fancy indexing with 2-D index arrays
    # works without hitting Devito's OOB-detection limitation.
    p = np.asarray(p_2d)

    # Zero ghost points before first iteration so the interpolation starts
    # from a clean state.
    p[gc[:, 0], gc[:, 1]] = 0.0

    prev = np.zeros(len(gc), dtype=p.dtype)
    for _ in range(n_iter):
        # Gather p at the Lagrange stencil nodes for every ghost point.
        p_nodes = p[ci[:, :, 0], ci[:, :, 1]]      # (N, M)
        # Weighted sum → mirror-point interpolant.
        p_mirror = (cw * p_nodes).sum(axis=1)       # (N,)
        # Antisymmetric BC: p_ghost = −p_mirror.
        p_new = (-p_mirror).astype(p.dtype)

        err = float(np.max(np.abs(p_new - prev)))
        p[gc[:, 0], gc[:, 1]] = p_new
        if err < tol:
            break
        prev = p_new


def ForwardOperator(model, geometry, space_order=4, save=False, **kwargs):
    """
    Forward modelling operator for variable-density acoustic wave propagation.

    Discretises:
        (1/v²) ∂²p/∂t² = ∇·(b ∇p) + src,   b = 1/rho

    The IBM free-surface correction (`ibm_step`) is applied after each FD
    time step by the wavesolver; it is not embedded in this operator.

    Parameters
    ----------
    model : ModelTopo
        Physical model with b, m, damp (bcs="mask").
    geometry : AcquisitionGeometry
        Source and receiver geometry.
    space_order : int, optional
        Spatial discretisation order (default 4).
    save : bool, optional
        If True, allocate a full-time-history pressure field.

    Returns
    -------
    Operator
    """
    p = TimeFunction(name='p', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)

    m, b, damp = model.m, model.b, model.damp
    s = model.grid.stepping_dim.spacing

    # Variable-density acoustic PDE residual.
    # In Devito 2-D: spatial dimensions are (x, y); the "depth" axis is y.
    pde = m * p.dt2 - (b * p.dx).dx - (b * p.dy).dy
    stencil = Eq(p.forward, damp * solve(pde, p.forward))

    src = geometry.src
    rec = geometry.rec
    src_term = src.inject(field=p.forward, expr=src * s**2 / m)
    rec_term = rec.interpolate(expr=p)

    return Operator([stencil] + src_term + rec_term,
                    subs=model.spacing_map,
                    name='ForwardTopo', **kwargs)
