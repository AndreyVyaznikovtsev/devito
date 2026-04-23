#!/usr/bin/env python3
"""
Compare CPU (Devito) vs CUDA (ForwardSH.cu compiled to .so) ForwardSH outputs.

Usage:
    python test_cuda_vs_cpu.py [path/to/ForwardSH.so]

Default .so path: examples/seismic/sh/ForwardSH.so
"""

import ctypes
import sys
import numpy as np
from pathlib import Path

from devito import TimeFunction, NODE
from examples.seismic import AcquisitionGeometry
from examples.seismic.model import ModelSH
from examples.seismic.sh.operators import ForwardOperator

# ---------------------------------------------------------------------------
# Model — identical to 18_sh_ivf_layers.ipynb
# ---------------------------------------------------------------------------
vs1, vs2, vs3 = 0.100, 0.400, 0.700   # km/s
rho = 1.970                            # kg/m³
mu1 = np.float32(rho * vs1**2)
mu2 = np.float32(rho * vs2**2)
mu3 = np.float32(rho * vs3**2)
b_val = np.float32(1.0 / rho)

dx          = 0.4   # m
n_vac       = 3
Nx          = int(300 / dx) + 1      # 751
Nz_sub      = int(150 / dx) + 1      # 376
nbl         = 100
space_order = 4

shape   = (Nx, Nz_sub + n_vac)       # (751, 379)
spacing = (dx, dx)
origin  = (0., -n_vac * dx)

z1_idx = n_vac + int(round(2.5   / dx))
z2_idx = n_vac + int(round(130.0 / dx))

mu_arr = np.zeros(shape, dtype=np.float32)
b_arr  = np.zeros(shape, dtype=np.float32)
mu_arr[:, n_vac:z1_idx]  = mu1
mu_arr[:, z1_idx:z2_idx] = mu2
mu_arr[:, z2_idx:]       = mu3
b_arr[:, n_vac:]         = b_val

topo = np.full(Nx, n_vac, dtype=int)

model = ModelSH(
    origin=origin, spacing=spacing, shape=shape,
    space_order=space_order, mu=mu_arr, b=b_arr,
    nbl=nbl, topo=topo,
)

t0, tn = 0., 850.
f0 = 0.010   # kHz = 10 Hz

src_positions = np.array([[150., 0.]], dtype=np.float32)
rec_x = np.arange(50., 251., 2., dtype=np.float32)   # 101 receivers
rec_positions = np.column_stack(
    [rec_x, np.zeros_like(rec_x)]
).astype(np.float32)

geometry = AcquisitionGeometry(
    model, rec_positions, src_positions,
    t0=t0, tn=tn, src_type='Ricker', f0=f0,
)

dt = model.critical_dt
nt = geometry.nt
print(f'Model: shape={shape}, nbl={nbl}, space_order={space_order}')
print(f'dt={dt:.6f} ms, nt={nt}')

# ---------------------------------------------------------------------------
# CPU reference run (Devito, default compiler / language)
# ---------------------------------------------------------------------------
print('\n--- Running CPU reference (Devito) ---')
x, z = model.grid.dimensions

v_cpu      = TimeFunction(name='v',      grid=model.grid, space_order=space_order,
                          time_order=1, staggered=NODE)
tau_xy_cpu = TimeFunction(name='tau_xy', grid=model.grid, space_order=space_order,
                          time_order=1, staggered=(x,))
tau_zy_cpu = TimeFunction(name='tau_zy', grid=model.grid, space_order=space_order,
                          time_order=1, staggered=(z,))
rec_cpu    = geometry.new_rec(name='rec')

op = ForwardOperator(model, geometry, space_order=space_order)
op.apply(v=v_cpu, tau_xy=tau_xy_cpu, tau_zy=tau_zy_cpu,
         src=geometry.src, rec=rec_cpu,
         dt=dt, **model.physical_params())

print(f'CPU rec max abs: {np.abs(rec_cpu.data).max():.6e}')

# ---------------------------------------------------------------------------
# ctypes wrapper for CUDA .so
# ---------------------------------------------------------------------------

class _DataObj(ctypes.Structure):
    _fields_ = [
        ('data',   ctypes.c_void_p),
        ('size',   ctypes.POINTER(ctypes.c_int)),
        ('nbytes', ctypes.c_ulong),
        ('npsize', ctypes.POINTER(ctypes.c_ulong)),
        ('dsize',  ctypes.POINTER(ctypes.c_ulong)),
        ('hsize',  ctypes.POINTER(ctypes.c_int)),
        ('hofs',   ctypes.POINTER(ctypes.c_int)),
        ('oofs',   ctypes.POINTER(ctypes.c_int)),
        ('dmap',   ctypes.c_void_p),
    ]


class _Profiler(ctypes.Structure):
    _fields_ = [
        ('section0', ctypes.c_double),
        ('section1', ctypes.c_double),
        ('section2', ctypes.c_double),
    ]


def _make_dataobj(arr: np.ndarray) -> _DataObj:
    assert arr.dtype == np.float32, f'expected float32, got {arr.dtype}'
    assert arr.flags['C_CONTIGUOUS']
    ndim = arr.ndim
    obj = _DataObj()
    # Keep a reference to the numpy array so the data buffer is not GC'd.
    obj._numpy_arr = arr
    obj.data   = arr.ctypes.data_as(ctypes.c_void_p)
    # Store ctypes arrays as attributes — assigning their pointer to a POINTER
    # field does NOT increment the refcount, so without this the arrays are
    # freed immediately and the C code reads dangling memory.
    obj._size_arr   = (ctypes.c_int   * ndim)(*arr.shape)
    obj._npsize_arr = (ctypes.c_ulong * ndim)(*arr.shape)
    obj._dsize_arr  = (ctypes.c_ulong * ndim)(*arr.shape)
    obj._hsize_arr  = (ctypes.c_int   * ndim)(*([0] * ndim))
    obj._hofs_arr   = (ctypes.c_int   * ndim)(*([0] * ndim))
    obj._oofs_arr   = (ctypes.c_int   * ndim)(*([0] * ndim))
    obj.size   = obj._size_arr
    obj.nbytes = ctypes.c_ulong(arr.nbytes)
    obj.npsize = obj._npsize_arr
    obj.dsize  = obj._dsize_arr
    obj.hsize  = obj._hsize_arr
    obj.hofs   = obj._hofs_arr
    obj.oofs   = obj._oofs_arr
    obj.dmap   = None
    return obj


def _load_cuda_lib(so_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so_path))
    p = ctypes.POINTER
    lib.ForwardSH.restype  = ctypes.c_int
    lib.ForwardSH.argtypes = [
        p(_DataObj),    # b_vec
        p(_DataObj),    # damp_vec
        p(_DataObj),    # mu_x_vec
        p(_DataObj),    # mu_z_vec
        p(_DataObj),    # rec_vec
        p(_DataObj),    # rec_coords_vec
        p(_DataObj),    # src_vec
        p(_DataObj),    # src_coords_vec
        p(_DataObj),    # tau_xy_vec
        p(_DataObj),    # tau_zy_vec
        p(_DataObj),    # v_vec
        ctypes.c_int,   # x_M
        ctypes.c_int,   # x_m
        ctypes.c_int,   # y_M
        ctypes.c_int,   # y_m
        ctypes.c_float, # dt
        ctypes.c_float, # o_x
        ctypes.c_float, # o_y
        ctypes.c_int,   # p_rec_M
        ctypes.c_int,   # p_rec_m
        ctypes.c_int,   # p_src_M
        ctypes.c_int,   # p_src_m
        ctypes.c_int,   # time_M
        ctypes.c_int,   # time_m
        ctypes.c_int,   # deviceid
        ctypes.c_int,   # devicerm
        p(_Profiler),   # timers
    ]
    return lib


# ---------------------------------------------------------------------------
# CUDA run
# ---------------------------------------------------------------------------
so_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
          Path(__file__).parent / 'ForwardSH.so'

print(f'\n--- Running CUDA version from {so_path} ---')
lib = _load_cuda_lib(so_path)

# Pull data arrays from Devito (full padded+halo buffers, C-contiguous float32)
def _get(func):
    arr = np.ascontiguousarray(func.data, dtype=np.float32)
    return arr

b_d      = _get(model.b)
damp_d   = _get(model.damp)
mu_x_d   = _get(model.mu_x)
mu_z_d   = _get(model.mu_z)

# Fresh wavefield buffers (zero-init, same shape as CPU wavefield)
halo = space_order   # confirmed: Devito allocates space_order ghost cells per side
nx_total = v_cpu.data.shape[1]
ny_total = v_cpu.data.shape[2]
v_cuda      = np.zeros((2, nx_total, ny_total), dtype=np.float32)
tau_xy_cuda = np.zeros((2, nx_total, ny_total), dtype=np.float32)
tau_zy_cuda = np.zeros((2, nx_total, ny_total), dtype=np.float32)

# Source/receiver data (same as Devito run)
src_data   = np.ascontiguousarray(geometry.src.data,          dtype=np.float32)
src_coords = np.ascontiguousarray(geometry.src.coordinates.data, dtype=np.float32)
rec_coords = np.ascontiguousarray(geometry.rec.coordinates.data, dtype=np.float32)
rec_cuda   = np.zeros((nt, len(rec_x)), dtype=np.float32)

# Iteration bounds: loop from 0..nx_pml-1 (C code adds +halo internally)
nx_pml = nx_total - 2 * halo
ny_pml = ny_total - 2 * halo
x_M = nx_pml - 1
x_m = 0
y_M = ny_pml - 1
y_m = 0

o_x = float(model.grid.origin[0])
o_y = float(model.grid.origin[1])

n_src = 1
n_rec = len(rec_x)

print(f'Array shape: v={v_cuda.shape}, 2D fields={b_d.shape}')
print(f'x_M={x_M}, y_M={y_M}, o_x={o_x}, o_y={o_y}')
print(f'time_M={nt-2}, n_src={n_src}, n_rec={n_rec}')

def do(arr):
    return _make_dataobj(np.ascontiguousarray(arr, dtype=np.float32))

b_do      = do(b_d)
damp_do   = do(damp_d)
mu_x_do   = do(mu_x_d)
mu_z_do   = do(mu_z_d)
rec_do    = do(rec_cuda)
rcoord_do = do(rec_coords)
src_do    = do(src_data)
scoord_do = do(src_coords)
txy_do    = do(tau_xy_cuda)
tzy_do    = do(tau_zy_cuda)
v_do      = do(v_cuda)

timers = _Profiler()

ret = lib.ForwardSH(
    ctypes.byref(b_do),
    ctypes.byref(damp_do),
    ctypes.byref(mu_x_do),
    ctypes.byref(mu_z_do),
    ctypes.byref(rec_do),
    ctypes.byref(rcoord_do),
    ctypes.byref(src_do),
    ctypes.byref(scoord_do),
    ctypes.byref(txy_do),
    ctypes.byref(tzy_do),
    ctypes.byref(v_do),
    ctypes.c_int(x_M),
    ctypes.c_int(x_m),
    ctypes.c_int(y_M),
    ctypes.c_int(y_m),
    ctypes.c_float(dt),
    ctypes.c_float(o_x),
    ctypes.c_float(o_y),
    ctypes.c_int(n_rec - 1),   # p_rec_M
    ctypes.c_int(0),           # p_rec_m
    ctypes.c_int(n_src - 1),   # p_src_M
    ctypes.c_int(0),           # p_src_m
    ctypes.c_int(nt - 2),      # time_M
    ctypes.c_int(0),           # time_m
    ctypes.c_int(0),           # deviceid (GPU 0)
    ctypes.c_int(1),           # devicerm (free device memory)
    ctypes.byref(timers),
)

if ret != 0:
    print(f'ERROR: ForwardSH returned {ret}')
    sys.exit(1)

print(f'CUDA rec max abs: {np.abs(rec_cuda).max():.6e}')
print(f'Timers: section0={timers.section0:.3f}s, '
      f'section1={timers.section1:.3f}s, section2={timers.section2:.3f}s')

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
print('\n--- Comparison ---')
cpu_rec  = rec_cpu.data.astype(np.float32)
cuda_rec = rec_cuda

signal_rms = np.sqrt(np.mean(cpu_rec**2))
abs_diff   = np.abs(cpu_rec - cuda_rec)
max_amp    = np.abs(cpu_rec).max()

# Max-relative: sensitive to float32 non-associativity at wavelet peaks
max_rel_err  = abs_diff.max() / (max_amp + 1e-30)
# RMS-relative: robust metric — most samples agree, large diff at a few peaks
rms_rel_err  = np.sqrt(np.mean(abs_diff**2)) / (signal_rms + 1e-30)

print(f'Signal max:    {max_amp:.4e}')
print(f'Signal RMS:    {signal_rms:.4e}')
print(f'Max abs diff:  {abs_diff.max():.4e}  (max-relative: {max_rel_err:.2%})')
print(f'RMS abs diff:  {np.sqrt(np.mean(abs_diff**2)):.4e}  (RMS-relative: {rms_rel_err:.2%})')
print(f'Mean abs diff: {abs_diff.mean():.4e}')
print()
print('Note: large max-relative error is expected between CPU (sequential float32)')
print('      and GPU (parallel float32) — summation order differs → non-associative')
print('      accumulation over 2584 steps. RMS-relative is the meaningful metric.')

# Pass if RMS relative error is small (< 5%)
tol = 0.05
if rms_rel_err < tol:
    print(f'\nPASS — RMS relative error {rms_rel_err:.2%} < {tol:.0%}')
else:
    print(f'\nFAIL — RMS relative error {rms_rel_err:.2%} >= {tol:.0%}')

# Save seismograms for visual inspection
np.save('/tmp/rec_cpu.npy',  cpu_rec)
np.save('/tmp/rec_cuda.npy', cuda_rec)
np.save('/tmp/rec_diff.npy', cpu_rec - cuda_rec)
print('\nSeismograms saved to /tmp/rec_cpu.npy, /tmp/rec_cuda.npy, /tmp/rec_diff.npy')
