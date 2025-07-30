import time
import numpy as np
from examples.seismic import SeismicModel, AcquisitionGeometry, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.datasets import SeismogramDataset, VelocityModel
from devito import info, TimeFunction, Function, Eq, Operator, norm
import os
from config import *
import argparse

def setup_model_and_geometry():
    """Set up the velocity model and acquisition geometry"""
    dataset = SeismogramDataset(PATH_DATA, "sou", invert_elevs=True)
    xmin, xmax = min(dataset.x_coords.min(), dataset.opposite_x.min()), max(dataset.x_coords.max(), dataset.opposite_x.max())
    spacing = (0.025, 0.025)
    
    velmodel = VelocityModel(
        PATH_MODEL,
        dx=spacing[0],
        dz=spacing[1],
        clip=True,
        xmin=xmin - 3,
        xmax=xmax + 3,
        zmin=-318,
    )
    velmodel.pad_left(4 + 2)
    velmodel.pad_right(8 * int(0.5 / spacing[0]) + 2)
    velmodel.pad_bottom(10 * int(0.5 / spacing[0]) + 2)
    velmodel.pad_top(7 * int(0.5 / spacing[0]))

    origin = velmodel.x[0], velmodel.z[0]
    vp = velmodel.vp.T
    
    model = SeismicModel(
        vp=vp,
        origin=origin,
        shape=vp.shape,
        spacing=spacing,
        space_order=SO,
        nbl=NBL,
        bcs="damp",
        fs=False,
    )
    
    return model, dataset

def compute_forward_wavefield(model, dataset, shot_id):
    """Compute and save forward wavefield snaps"""
    t0 = 0.0
    tn = dataset._t_max
    d_obs, sx, sz, rec_x, rec_z = dataset[shot_id]
    
    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T
    wav_data = np.load(f"{PATH_WAVELETS}/{shot_id+1}.npy")
    geometry = AcquisitionGeometry(
        model, rec_pos, src_pos,
        t0, tn, f0=0.25,
        src_type=None, wav_data=wav_data
    )
    d_syn = Receiver(name='d_syn', grid=model.grid, time_range=geometry.time_axis,
                     coordinates=geometry.rec_positions)
    solver = AcousticWaveSolver(model, geometry, space_order=SO)
    _, _, u0, _ = solver.forward(vp=model.vp, save=True, nsnaps=NSNAPS, rec=d_syn)
    scale_factor = np.sqrt(np.sum(d_obs.ravel()**2))/np.sqrt(np.sum(d_syn.data[:].ravel()**2))
    
    # Save snaps to binary file
    os.makedirs(OUTPUT_DIRS['forward_snaps'], exist_ok=True)
    u0.data.tofile(f"{OUTPUT_DIRS['forward_snaps']}/{shot_id+1}.bin")
    np.save(f"{OUTPUT_DIRS['forward_snaps']}/scale_factor_{shot_id+1}.npy", scale_factor)
    return u0

def compute_adjoint_wavefield(model, dataset, shot_id, dm):
    """Compute and save adjoint wavefield snaps"""
    t0 = 0.0
    tn = dataset._t_max
    d_obs, sx, sz, rec_x, rec_z = dataset[shot_id]
    
    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T
    wav_data = np.load(f"{PATH_WAVELETS}/{shot_id+1}.npy")
    scale_factor = np.load(f"{OUTPUT_DIRS['forward_snaps']}/scale_factor_{shot_id+1}.npy")
    wav_data *= scale_factor
    geometry = AcquisitionGeometry(
        model, rec_pos, src_pos,
        t0, tn, f0=0.25, src_type=None,
        wav_data=wav_data
    )
    solver = AcousticWaveSolver(model, geometry, space_order=SO)

    d_syn = Receiver(name='d_syn', grid=model.grid, time_range=geometry.time_axis,
                     coordinates=geometry.rec_positions)
    solver.jacobian(dmin=dm, vp=model.vp, rec=d_syn)
    d_syn.data.tofile(f"{OUTPUT_DIRS['adjoint_snaps']}/d_syn_{shot_id+1}.bin")
    residual = Receiver(name='residual', grid=model.grid, time_range=geometry.time_axis,
                       coordinates=geometry.rec_positions)
    residual.data[:] = d_syn.data[:] - d_obs.T

    _, v, _ = solver.adjoint(vp=model.vp, rec=residual, save=True, nsnaps=NSNAPS)
    
    os.makedirs(OUTPUT_DIRS['adjoint_snaps'], exist_ok=True)
    v.data.tofile(f"{OUTPUT_DIRS['adjoint_snaps']}/{shot_id+1}.bin")
    objective = 0.5*norm(residual)**2
    return v, residual, objective


def load_current_image(iter_num, shape):
    """Load image from previous iteration"""
    path = f"{OUTPUT_DIRS['images']}/image_iter_{iter_num}.bin"
    return np.fromfile(path).reshape(shape).astype(np.float64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['forward', 'adjoint'], required=True)
    parser.add_argument('--iter', type=int, default=0)
    args = parser.parse_args()

    model, dataset = setup_model_and_geometry()
    os.makedirs(OUTPUT_DIRS['results'], exist_ok=True)
    log_file = os.path.join(OUTPUT_DIRS['results'], 'optimization.log')
    
    if args.mode == 'forward':
        # Only compute forward wavefields if they don't exist
        if not os.path.exists(f"{OUTPUT_DIRS['forward_snaps']}/1.bin"):
            for i in range(len(dataset)):
            # for i in range(0, len(dataset), 10):
                compute_forward_wavefield(model, dataset, i)
    else:
        # For adjoint, load current image
        dm = Function(name='dm', grid=model.grid)
        if args.iter == 0:
            pass
        else:
            dm.data[:] = load_current_image(args.iter - 1, model.vp.shape)
        
        dataset._dt_r = model.critical_dt
        dataset._t_max_r = dataset._t_max
        dataset.resample_on()
        # Compute adjoint wavefields
        objective = 0.
        # for i in range(0, len(dataset), 10):
        for i in range(len(dataset)):
            _, _, obj_shot = compute_adjoint_wavefield(model, dataset, i, dm)
            objective += obj_shot
            print('\033[1m' + f'Current objective - {objective:.5f}' + '\033[0m')
        
        message = f"Iteration {args.iter+1} - Objective: {objective:.5f}"
        print('\033[1m' + message + '\033[0m')
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(message + '\n')

if __name__== "__main__":
    main()