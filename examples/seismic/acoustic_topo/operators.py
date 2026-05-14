import numba
import numpy as np
from devito import Eq, Inc, Operator, TimeFunction, Function, solve

__all__ = ['ForwardOperator', 'GradientOperator', 'ibm_step', 'ibm_step_numba']


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


@numba.njit
def ibm_step_numba(p, ghost_coords, coeff_indices, coeff_weights,
                   n_iter=20, tol=1e-6):
    """
    Numba-JIT-compiled IBM ghost-point correction.

    Same logic as ``ibm_step`` but operates on raw numpy arrays without
    Python overhead.  Call from Python as::

        ibm_step_numba(np.asarray(p_2d), model.ghost_coords,
                       model.coeff_indices, model.coeff_weights,
                       n_iter=n_iter, tol=tol)

    Parameters
    ----------
    p : ndarray, shape (nx_pad, nz_pad)
        One time-buffer slice of the pressure field, modified in-place.
    ghost_coords : ndarray int, shape (N, 2)  — [ix, iz] padded indices.
    coeff_indices : ndarray int, shape (N, M, 2) — Lagrange stencil nodes.
    coeff_weights : ndarray float, shape (N, M) — Lagrange weights.
    n_iter : int
        Maximum iterations (default 20).
    tol : float
        L∞ convergence tolerance on ghost-point change (default 1e-6).
    """
    N = ghost_coords.shape[0]
    if N == 0:
        return

    ix = ghost_coords[:, 0]
    iz = ghost_coords[:, 1]

    # Zero ghost points before first iteration
    for g in range(N):
        p[ix[g], iz[g]] = 0.0

    prev = np.zeros(N, dtype=p.dtype)
    M = coeff_weights.shape[1]
    cix = coeff_indices[:, :, 0]
    ciz = coeff_indices[:, :, 1]

    for _ in range(n_iter):
        err = 0.0
        for g in range(N):
            val = 0.0
            for m in range(M):
                val += coeff_weights[g, m] * p[cix[g, m], ciz[g, m]]
            p_new = -val
            diff = abs(p_new - prev[g])
            if diff > err:
                err = diff
            p[ix[g], iz[g]] = p_new
            prev[g] = p_new
        if err < tol:
            break


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
    x, y = model.grid.dimensions

    # Variable-density acoustic PDE:  m ∂²p/∂t² = ∇·(b ∇p) + src
    # Staggered 1st-derivative operators give the compact (non-wide) stencil
    # for the divergence term, avoiding the checkerboard null-space that arises
    # from composing two centered 1st derivatives on the same grid.
    pde = m * p.dt2 - (b * p.dx(x0=x + x.spacing/2)).dx(x0=x - x.spacing/2) \
                    - (b * p.dy(x0=y + y.spacing/2)).dy(x0=y - y.spacing/2)
    stencil = Eq(p.forward, damp * solve(pde, p.forward))

    src = geometry.src
    rec = geometry.rec
    src_term = src.inject(field=p.forward, expr=src * s**2 / m)
    rec_term = rec.interpolate(expr=p)

    return Operator([stencil] + src_term + rec_term,
                    subs=model.spacing_map,
                    name='ForwardTopo', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True, **kwargs):
    """
    Gradient operator for variable-density acoustic wave propagation.

    Back-propagates the adjoint wavefield and accumulates the gradient of
    the misfit functional with respect to slowness-squared ``m``.

    The operator is single-step (one time level per ``apply`` call) so the
    caller can interleave IBM ghost-point correction between time steps.

    Parameters
    ----------
    model : ModelTopo
        Physical model with b, m, damp (bcs="mask").
    geometry : AcquisitionGeometry
        Source and receiver geometry.
    space_order : int, optional
        Spatial discretisation order (default 4).
    save : bool, optional
        Must be True (forward wavefield ``u`` must be stored).

    Returns
    -------
    Operator
    """
    grad = Function(name='grad', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    rec = geometry.rec

    m, b, damp = model.m, model.b, model.damp
    s = model.grid.stepping_dim.spacing
    x, y = model.grid.dimensions

    # Adjoint wave equation (same spatial operator, backward in time)
    #   m * v.dt2 = div(b * grad(v)) + adj_src
    pde = m * v.dt2 - (b * v.dx(x0=x + x.spacing/2)).dx(x0=x - x.spacing/2) \
                    - (b * v.dy(x0=y + y.spacing/2)).dy(x0=y - y.spacing/2)
    eq_time = solve(pde, v.backward)
    eqn = Eq(v.backward, damp * eq_time)

    # Adjoint source injection (receiver residuals)
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Gradient of the misfit w.r.t. slowness-squared m:
    #   g(x) = -∫ u(x,t) * ∂²v/∂t²(x,t) dt   (OT2 kernel only)
    gradient_update = Inc(grad, - u * v.dt2)

    return Operator([eqn] + receivers + [gradient_update],
                    subs=model.spacing_map,
                    name='GradientTopo', **kwargs)
