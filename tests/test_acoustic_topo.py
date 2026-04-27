import numpy as np
import pytest

from examples.seismic.acoustic_topo.ibm_setup import (
    _lagrange_weights_1d,
    lagrange_coefficients,
    find_ghost_points,
    find_intercept_points,
    find_mirror_points,
)
from examples.seismic.acoustic_topo.model import ModelTopo
from examples.seismic.acoustic_topo.operators import ibm_step
from examples.seismic.acoustic_topo.wavesolver import AcousticTopoSolver
from examples.seismic.utils import AcquisitionGeometry


# ── helpers ──────────────────────────────────────────────────────────────────

def _flat_model(nx=40, nz=40, dx=10., dz=10., topo_z=0., nbl=10,
                space_order=4):
    """Constant-velocity ModelTopo with a flat free surface at z = topo_z."""
    shape = (nx, nz)
    v = 1.5 * np.ones(shape, dtype=np.float32)
    rho = np.ones(shape, dtype=np.float32)
    topo = topo_z * np.ones(nx)
    return ModelTopo(origin=(0., 0.), spacing=(dx, dz), shape=shape,
                     space_order=space_order, v=v, rho=rho, topo=topo,
                     nbl=nbl, bcs='mask')


def _geometry(model, tn=50.):
    """Single-shot geometry with source at the model centre."""
    nx, nz = model.shape
    dx, dz = model.spacing
    ox, oz = model.origin
    cx = ox + nx * dx / 2.
    cz = oz + nz * dz / 2.
    pos = np.array([[cx, cz]])
    return AcquisitionGeometry(model, pos, pos, t0=0., tn=tn,
                               src_type='Ricker', f0=0.025)


# ── IBM geometry unit tests ───────────────────────────────────────────────────

def test_lagrange_weights_1d_partition_of_unity():
    """1-D Lagrange weights at any fractional position must sum to 1."""
    for n in [2, 4, 6]:
        for t in np.linspace(0.5, n - 0.5, 9):
            w = _lagrange_weights_1d(t, n)
            assert np.isclose(w.sum(), 1.0, atol=1e-12), (
                f"n={n}, t={t:.3f}: sum={w.sum()}")


def test_lagrange_weights_2d_partition_of_unity():
    """Tensor-product Lagrange weights must sum to 1 for every ghost point."""
    model = _flat_model(topo_z=15., nx=30, nz=30, nbl=5)
    if len(model.coeff_weights) == 0:
        pytest.skip("no ghost points in this configuration")
    row_sums = model.coeff_weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10), (
        f"max deviation = {np.max(np.abs(row_sums - 1.)):.2e}")


def test_mirror_midpoint_equals_intercept():
    """Midpoint of (ghost, mirror) must lie on the surface (= intercept)."""
    topo_z = 15.
    model = _flat_model(topo_z=topo_z, nx=20, nz=20, dx=10., dz=10., nbl=5)
    gc = model.ghost_coords
    mc = model.mirror_coords
    if len(gc) == 0:
        pytest.skip("no ghost points")

    dz = model.spacing[1]
    oz = model.origin[1]
    z_g = oz + gc[:, 1].astype(float) * dz
    # For a flat surface the intercept z-coordinate equals topo_z exactly.
    mid_z = 0.5 * (z_g + mc[:, 1])
    # Tolerance = half a dz (intercept search is discrete, not analytical).
    assert np.allclose(mid_z, topo_z, atol=0.5 * dz)


def test_flat_surface_at_origin_no_ghosts():
    """Flat topo at z = origin[1] = 0 must produce zero ghost points."""
    model = _flat_model(topo_z=0.)
    assert len(model.ghost_coords) == 0


def test_ghost_count_increases_with_topo():
    """More surface elevation → more ghost rows → more ghost points."""
    m0 = _flat_model(topo_z=0.,  nx=20, nz=20, nbl=5)
    m1 = _flat_model(topo_z=10., nx=20, nz=20, nbl=5)
    m2 = _flat_model(topo_z=20., nx=20, nz=20, nbl=5)
    assert len(m0.ghost_coords) < len(m1.ghost_coords) < len(m2.ghost_coords)


# ── IBM correction unit tests ─────────────────────────────────────────────────

def test_ibm_step_zero_field_unchanged():
    """ibm_step on an all-zero field must leave every value at zero."""
    model = _flat_model(topo_z=15., nx=20, nz=20, nbl=5)
    nx_pad, nz_pad = model.grid.shape
    p = np.zeros((nx_pad, nz_pad), dtype=np.float32)
    ibm_step(p, model)
    assert np.all(p == 0.)


def test_ibm_step_antisymmetry():
    """At convergence p_ghost + p_mirror_interp = 0 for every ghost point."""
    model = _flat_model(topo_z=15., nx=30, nz=30, nbl=5)
    gc = model.ghost_coords
    if len(gc) == 0:
        pytest.skip("no ghost points")

    nx_pad, nz_pad = model.grid.shape
    rng = np.random.default_rng(0)
    p = rng.standard_normal((nx_pad, nz_pad)).astype(np.float32)
    p[gc[:, 0], gc[:, 1]] = 0.

    ibm_step(p, model, n_iter=50, tol=1e-12)

    ci = model.coeff_indices   # (N, M, 2)
    cw = model.coeff_weights   # (N, M)
    p_mirror = (cw * p[ci[:, :, 0], ci[:, :, 1]]).sum(axis=1)
    p_ghost = p[gc[:, 0], gc[:, 1]]
    np.testing.assert_allclose(p_ghost + p_mirror, 0., atol=1e-5)


def test_ibm_step_no_ghostpoints_is_noop():
    """ibm_step with no ghost points must not raise and must not alter data."""
    model = _flat_model(topo_z=0.)   # no ghost points
    nx_pad, nz_pad = model.grid.shape
    rng = np.random.default_rng(1)
    p = rng.standard_normal((nx_pad, nz_pad)).astype(np.float32)
    p_orig = p.copy()
    ibm_step(p, model)
    np.testing.assert_array_equal(p, p_orig)


# ── Forward propagation integration tests ────────────────────────────────────

def test_forward_flat_topo_no_nan():
    """Forward propagation with flat surface at origin must not produce NaN/Inf."""
    model = _flat_model(topo_z=0., nx=40, nz=40, nbl=10)
    geom = _geometry(model, tn=50.)
    solver = AcousticTopoSolver(model, geom, space_order=4)
    rec, p, _ = solver.forward()
    assert np.all(np.isfinite(rec.data)), "rec contains NaN or Inf"
    assert np.all(np.isfinite(p.data)),  "p contains NaN or Inf"


def test_forward_with_topo_no_nan():
    """Forward propagation with shallow topo must not produce NaN or Inf."""
    model = _flat_model(topo_z=15., nx=40, nz=40, nbl=10)
    geom = _geometry(model, tn=50.)
    solver = AcousticTopoSolver(model, geom, space_order=4)
    rec, p, _ = solver.forward()
    assert np.all(np.isfinite(rec.data)), "rec contains NaN or Inf"
    assert np.all(np.isfinite(p.data)),  "p contains NaN or Inf"


def test_forward_save_no_nan():
    """Forward with save=True (full history) must not produce NaN or Inf."""
    model = _flat_model(topo_z=15., nx=30, nz=30, nbl=8)
    geom = _geometry(model, tn=30.)
    solver = AcousticTopoSolver(model, geom, space_order=4)
    rec, p, _ = solver.forward(save=True)
    assert np.all(np.isfinite(rec.data)), "rec contains NaN or Inf"
    assert np.all(np.isfinite(p.data)),  "p contains NaN or Inf"


def test_forward_save_shape():
    """With save=True, p.data must have the full time-history shape (nt, nx, nz)."""
    model = _flat_model(topo_z=15., nx=30, nz=30, nbl=8)
    geom = _geometry(model, tn=30.)
    solver = AcousticTopoSolver(model, geom, space_order=4)
    _, p, _ = solver.forward(save=True)
    nt = geom.nt
    assert p.data.shape[0] == nt, (
        f"Expected p.data.shape[0]={nt}, got {p.data.shape[0]}")


def test_forward_ibm_zero_topo_matches_no_ibm_correction():
    """With topo at z=0 (no ghost points) IBM iterations do nothing, so the
    result must be finite and have nonzero amplitude at the receiver."""
    model = _flat_model(topo_z=0., nx=40, nz=40, nbl=10)
    assert len(model.ghost_coords) == 0  # safety check
    geom = _geometry(model, tn=100.)
    solver = AcousticTopoSolver(model, geom, space_order=4)
    rec, _, _ = solver.forward()
    assert np.max(np.abs(rec.data)) > 0., "receiver traces are all zero"
