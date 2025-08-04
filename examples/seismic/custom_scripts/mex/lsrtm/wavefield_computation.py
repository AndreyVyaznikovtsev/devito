import time
import numpy as np
from examples.seismic import SeismicModel, AcquisitionGeometry, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.datasets import SeismogramDataset, VelocityModel
from devito import info, TimeFunction, Function, Eq, Operator, norm
from scipy.signal import resample
from scipy.interpolate import interp1d
import os
from config import *
import argparse
from datetime import datetime
from matplotlib import pyplot as plt



def setup_model_and_geometry(iter_num):
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
    # if iter_num == 0:
    #     path = f"{OUTPUT_DIRS['images']}/image_iter_{iter_num}.npy"
    #     np.save(path, model.vp.data[NBL:-NBL, NBL:-NBL])
    #     print(model.vp.data[NBL:-NBL, NBL:-NBL].shape)
    
    return model, dataset

def load_current_dm(iter_num):
    dm = np.load(f"{OUTPUT_DIRS['images']}/image_iter_{iter_num}.npy")
    dm = np.pad(dm, ((NBL, NBL), (NBL, NBL)), mode='constant', constant_values=0.)
    return dm

def compute_forward_snaps(model, dataset, shot_id):
    """Compute and save forward wavefield snaps"""
    t0 = 0.0
    tn = TMAX
    d_obs, sx, sz, rec_x, rec_z = dataset[shot_id]
    
    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T
    wav_data_source = np.load(f"{PATH_WAVELETS}/Mex_wavelet_{shot_id}_norm.npy")
    scale_factor = np.load(f"{PATH_WAVELETS}/Mex_wavelet_{shot_id}_norm_scalar.npy")
    wav_time = np.arange(0, WAVELETS_TMAX + WAVELETS_DT, WAVELETS_DT)
    
    new_time = np.linspace(0, TMAX, d_obs.shape[1])
    interp_func = interp1d(wav_time, wav_data_source, kind='linear', 
                      bounds_error=False, fill_value=0.0)
    wav_data = interp_func(new_time)
    
    geometry = AcquisitionGeometry(
        model, rec_pos, src_pos,
        t0, tn, f0=0.25,
        src_type=None, wav_data=wav_data*scale_factor
    )
    d_syn = Receiver(name='d_syn', grid=model.grid, time_range=geometry.time_axis,
                     coordinates=geometry.rec_positions)
    solver = AcousticWaveSolver(model, geometry, space_order=SO)
    _, u0, _ = solver.forward(vp=model.vp, save=True, nsnaps=NSNAPS, rec=d_syn)
    np.save(f"{OUTPUT_DIRS['forward_snaps']}/{shot_id+1}.npy", u0.data[:, NBL//SUBSAMPLING:-NBL//SUBSAMPLING, NBL//SUBSAMPLING:-NBL//SUBSAMPLING])


def compute_wavefields(model, dataset, shot_id, dm):
    """Compute and save forward wavefield snaps"""
    objective = 0.0
    t0 = 0.0
    tn = TMAX
    d_obs, sx, sz, rec_x, rec_z = dataset[shot_id]
    
    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T
    wav_data_source = np.load(f"{PATH_WAVELETS}/Mex_wavelet_{shot_id}_norm.npy")
    scale_factor = np.load(f"{PATH_WAVELETS}/Mex_wavelet_{shot_id}_norm_scalar.npy")
    wav_time = np.arange(0, WAVELETS_TMAX + WAVELETS_DT, WAVELETS_DT)
    
    new_time = np.linspace(0, TMAX, d_obs.shape[1])
    interp_func = interp1d(wav_time, wav_data_source, kind='linear', 
                      bounds_error=False, fill_value=0.0)
    wav_data = interp_func(new_time)
    
    geometry = AcquisitionGeometry(
        model, rec_pos, src_pos,
        t0, tn, f0=0.25,
        src_type=None, wav_data=wav_data*scale_factor
    )
    d_syn = Receiver(name='d_syn', grid=model.grid, time_range=geometry.time_axis,
                     coordinates=geometry.rec_positions)
    
    solver = AcousticWaveSolver(model, geometry, space_order=SO)

    solver.jacobian(dmin=dm, vp=model.vp, rec=d_syn)
    # np.save(f"{OUTPUT_DIRS['adjoint_snaps']}/recon_gather_{shot_id+1}.npy", d_syn.data[:])

    residual = Receiver(name='residual', grid=model.grid, time_range=geometry.time_axis,
                    coordinates=geometry.rec_positions)
    residual.data[:] = d_syn.data[:] - d_obs.T
    
    _, v, _ = solver.adjoint(vp=model.vp, rec=residual, save=True, nsnaps=NSNAPS)
    np.save(f"{OUTPUT_DIRS['adjoint_snaps']}/{shot_id+1}.npy", v.data[:, NBL//SUBSAMPLING:-NBL//SUBSAMPLING, NBL//SUBSAMPLING:-NBL//SUBSAMPLING])
    objective = 0.5*norm(residual)**2/scale_factor

    return objective

def main():
    os.makedirs(OUTPUT_DIRS['results'], exist_ok=True)
    os.makedirs(OUTPUT_DIRS['forward_snaps'], exist_ok=True)
    os.makedirs(OUTPUT_DIRS['adjoint_snaps'], exist_ok=True)
    os.makedirs(OUTPUT_DIRS['images'], exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--recon', type=int, default=0)
    parser.add_argument('--inv', type=int, default=0)
    args = parser.parse_args()

    model, dataset = setup_model_and_geometry(args.iter)
    dataset._dt_r = model.critical_dt
    dataset._t_max_r = TMAX
    dataset.resample_on()
    print("Num samples :", dataset._t_max/model.critical_dt)
    log_file = os.path.join(OUTPUT_DIRS['results'], 'optimization.log')
    if args.iter == 0:
        with open(log_file, 'a') as f:
            f.write(f"LSRTM experiment at {datetime.now()}, recon - {args.recon}, inv - {args.inv}" + '\n')

    
    # if args.iter == 0:
    #     for i in range(len(dataset)):
    #         compute_forward_snaps(model, dataset, i, iter_num=args.iter, recon=1)
    
    dm = load_current_dm(args.iter)

    objective = 0.
    for i in range(len(dataset)):
    # for i in SHOT_IDS:
        scale_factor = np.load(f"{PATH_WAVELETS}/Mex_wavelet_{i}_norm_scalar.npy")
        dmin = Function(name='dm', grid=model.grid)
        # dmin.data[:] = dm*len(dataset)
        dmin.data[:] = dm*np.sqrt(scale_factor)*len(dataset)

        obj_shot = compute_wavefields(model, dataset, i,
                                      dm=dmin
                                      )
        objective += obj_shot
        print('\033[1m' + f'{i+1}. Current objective - {objective/len(dataset):.5f}' + '\033[0m')
    objective /= len(dataset)
    message = f"Iteration {args.iter} - Objective: {objective:.5f}"
    print('\033[1m' + message + '\033[0m')
    with open(log_file, 'a') as f:
        f.write(message + '\n')

if __name__== "__main__":
    main()