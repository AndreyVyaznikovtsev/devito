import os
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

from devito import norm
from examples.seismic import AcquisitionGeometry, Receiver
from examples.seismic.acoustic import AcousticWaveSolver

from config import *
from config import setup_model_and_geometry

# ---------------------------------------------------------------------------- #
# Utility functions
# ---------------------------------------------------------------------------- #
def load_and_interpolate_wavelet(shot_id, target_samples):
    """Load source wavelet and interpolate to match observed data samples."""
    wav_data_source = np.load(f"{PATH_WAVELETS}/wavelet_{shot_id}_norm.npy")
    wav_time = np.arange(0, WAVELETS_TMAX + WAVELETS_DT, WAVELETS_DT)
    new_time = np.linspace(0, TMAX, target_samples)

    interp_func = interp1d(
        wav_time, wav_data_source, kind='linear', bounds_error=False, fill_value=0.0
    )
    return interp_func(new_time)

def create_geometry(model, sx, sz, rec_x, rec_z, wav_data):
    """Build acquisition geometry for given shot."""
    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T

    return AcquisitionGeometry(
        model, rec_pos, src_pos,
        t0=0.0, tn=TMAX, f0=0.25,
        src_type=None, wav_data=wav_data
    )

# ---------------------------------------------------------------------------- #
# Forward modeling
# ---------------------------------------------------------------------------- #
def compute_forward_snaps(solver, dataset, shot_id):
    """Compute forward propagated wavefield and save snapshots & synthetic gather."""
    d_obs, sx, sz, rec_x, rec_z = dataset[shot_id]

    wav_data = load_and_interpolate_wavelet(shot_id, d_obs.shape[1])
    geometry = create_geometry(solver.model, sx, sz, rec_x, rec_z, wav_data)

    # Update the geometry of the solver for this shot
    solver.geometry = geometry

    d_syn = Receiver(
        name='d_syn', grid=solver.model.grid, time_range=solver.geometry.time_axis,
        coordinates=solver.geometry.rec_positions
    )
    _, u0, _ = solver.forward(
        vp=solver.model.vp, save=True, nsnaps=NSNAPS, rec=d_syn,
        space_subsample=(SUBSAMPLING, SUBSAMPLING)
    )
    np.save(f"{OUTPUT_DIRS['forward_snaps']}/{shot_id+1}.npy",
            u0.data[:, NBL//SUBSAMPLING:-NBL//SUBSAMPLING,
                    NBL//SUBSAMPLING:-NBL//SUBSAMPLING])
    np.save(f"{OUTPUT_DIRS['forward_snaps']}/recon_gather_{shot_id+1}.npy",
            d_syn.data[:])

# ---------------------------------------------------------------------------- #
# Residual calculation
# ---------------------------------------------------------------------------- #
def incorp_residual(residual, d_syn, d_obs):
    """Normalize synthetic & observed data and compute residual."""
    assert d_syn.shape == d_obs.shape, "Synthetic and observed data must match."

    ds, do = d_syn.ravel(), d_obs.ravel()
    ds_norm, do_norm = np.linalg.norm(ds), np.linalg.norm(do)

    if ds_norm == 0 or do_norm == 0:
        residual.data[:] = 0
        return

    r = (ds / ds_norm) - (do / do_norm)
    residual.data[:] = r.reshape(residual.data.shape)

# ---------------------------------------------------------------------------- #
# Adjoint wavefields
# ---------------------------------------------------------------------------- #
def compute_adjoint_wavefields(solver, dataset, shot_id):
    """Run adjoint simulation using residual and save adjoint wavefield."""
    d_obs, sx, sz, rec_x, rec_z = dataset[shot_id]

    wav_data = load_and_interpolate_wavelet(shot_id, d_obs.shape[1])
    geometry = create_geometry(solver.model, sx, sz, rec_x, rec_z, wav_data)

    # Update the geometry of the solver for this shot
    solver.geometry = geometry

    d_syn_data = np.load(f"{OUTPUT_DIRS['forward_snaps']}/recon_gather_{shot_id+1}.npy")

    residual = Receiver(
        name='residual', grid=solver.model.grid, time_range=solver.geometry.time_axis,
        coordinates=solver.geometry.rec_positions
    )

    incorp_residual(residual, d_syn_data, d_obs.T)

    _, v, _ = solver.adjoint(
        vp=solver.model.vp, rec=residual, save=True, nsnaps=NSNAPS,
        space_subsample=(SUBSAMPLING, SUBSAMPLING)
    )

    np.save(f"{OUTPUT_DIRS['adjoint_snaps']}/{shot_id+1}.npy",
            v.data[:, NBL//SUBSAMPLING:-NBL//SUBSAMPLING,
                    NBL//SUBSAMPLING:-NBL//SUBSAMPLING])

    return 0.5 * norm(residual)**2

# ---------------------------------------------------------------------------- #
# Main driver
# ---------------------------------------------------------------------------- #
def main():
    # Prepare output dirs
    for key in OUTPUT_DIRS:
        os.makedirs(OUTPUT_DIRS[key], exist_ok=True)

    # Setup model & dataset
    model, dataset, _ = setup_model_and_geometry(PATH_DATA_DPLUS)
    dataset._dt_r = model.critical_dt
    dataset._t_max_r = TMAX
    dataset.resample_on()

    print("Num samples:", dataset._t_max / model.critical_dt)

    # Initialize solver once
    d_obs, sx, sz, rec_x, rec_z = dataset[0]
    geometry = create_geometry(model, sx, sz, rec_x, rec_z, np.zeros(d_obs.shape[1]))
    solver = AcousticWaveSolver(model, geometry, space_order=SO)

    # Forward modeling for all shots
    for i in range(len(dataset)):
        compute_forward_snaps(solver, dataset, i)

    objective = 0.0
    for i in range(len(dataset)):
        objective += compute_adjoint_wavefields(solver, dataset, i)
        if i % 5 == 0:
            print(f"\033[1m{i+1}. Current objective - {objective/len(dataset):.8f}\033[0m")

    objective /= len(dataset)
    print(f"\033[1mFinal Objective: {objective:.8f}\033[0m")


if __name__ == "__main__":
    main()
