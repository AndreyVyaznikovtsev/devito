import numpy as np
from examples.seismic import AcquisitionGeometry, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from devito import info, TimeFunction, Function, Eq, Operator, norm
from scipy.interpolate import interp1d
import os
from config import *
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
from config import setup_model_and_geometry

def load_current_dm(iter_num):
    if iter_num == 0:
        return 0.
    else:
        dm = np.load(f"{OUTPUT_DIRS['images']}/image_iter_{iter_num-1}.npy")
        dm = np.pad(dm, ((NBL, NBL), (NBL, NBL)), mode='constant', constant_values=0.)
    return dm

def compute_forward_snaps(model, dataset, shot_id):
    """Compute and save forward wavefield snaps"""
    t0 = 0.0
    tn = TMAX
    d_obs, sx, sz, rec_x, rec_z = dataset[shot_id]
    
    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T
    wav_data_source = np.load(f"{PATH_WAVELETS}/wavelet_{shot_id}_norm.npy")
    scale_factor = np.load(f"{PATH_WAVELETS}/wavelet_scale_{shot_id}.npy")
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
    _, u0, _ = solver.forward(vp=model.vp, save=True, nsnaps=NSNAPS, rec=d_syn, space_subsample=(SUBSAMPLING, SUBSAMPLING)) #  
    np.save(f"{OUTPUT_DIRS['forward_snaps']}/{shot_id+1}.npy", u0.data[:, NBL//SUBSAMPLING:-NBL//SUBSAMPLING, NBL//SUBSAMPLING:-NBL//SUBSAMPLING])


def compute_wavefields(model, dataset, shot_id, dm, iter_num):
    """Compute and save forward wavefield snaps"""
    objective = 0.0
    t0 = 0.0
    tn = TMAX
    d_obs, sx, sz, rec_x, rec_z = dataset[shot_id]
    
    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T
    wav_data_source = np.load(f"{PATH_WAVELETS}/wavelet_{shot_id}_norm.npy")
    scale_factor = np.load(f"{PATH_WAVELETS}/wavelet_scale_{shot_id}.npy")
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

    if iter_num == 0:
        solver.forward(vp=model.vp, rec=d_syn)
    else:
        solver.jacobian(dmin=dm, vp=model.vp, rec=d_syn)

    np.save(f"{OUTPUT_DIRS['adjoint_snaps']}/recon_gather_{shot_id+1}.npy", d_syn.data[:])

    residual = Receiver(name='residual', grid=model.grid, time_range=geometry.time_axis,
                    coordinates=geometry.rec_positions)
    residual.data[:] = d_syn.data[:] - d_obs.T
    
    _, v, _ = solver.adjoint(vp=model.vp, rec=residual, save=True, nsnaps=NSNAPS, space_subsample=(SUBSAMPLING, SUBSAMPLING))
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

    model, dataset, _ = setup_model_and_geometry(PATH_DATA_DSUB)
    dataset._dt_r = model.critical_dt
    dataset._t_max_r = TMAX
    dataset.resample_on()
    print("Num samples :", dataset._t_max/model.critical_dt)
    log_file = os.path.join(OUTPUT_DIRS['results'], 'optimization.log')
    if args.iter == 0:
        with open(log_file, 'a') as f:
            f.write(f"LSRTM experiment at {datetime.now()}, recon - {args.recon}, inv - {args.inv}" + '\n')

    
    if args.iter == 0:
        for i in range(len(dataset)):
            compute_forward_snaps(model, dataset, i)
    
    dm = load_current_dm(args.iter)

    objective = 0.
    for i in range(len(dataset)):
    # for i in SHOT_IDS:
        scale_factor = np.load(f"{PATH_WAVELETS}/wavelet_scale_{i}.npy")
        dmin = Function(name='dm', grid=model.grid)
        # dmin.data[:] = dm*len(dataset)
        # dmin.data[:] = dm*np.sqrt(scale_factor)*len(dataset)
        dmin.data[:] = dm


        obj_shot = compute_wavefields(model, dataset, i,
                                      dm=dmin,
                                      iter_num=args.iter
                                      )
        objective += obj_shot
        if i % 5 == 0:
            print('\033[1m' + f'{i+1}. Current objective - {objective/len(dataset):.8f}' + '\033[0m')
    objective /= len(dataset)
    message = f"Iteration {args.iter} - Objective: {objective:.8f}"
    print('\033[1m' + message + '\033[0m')
    with open(log_file, 'a') as f:
        f.write(message + '\n')

if __name__== "__main__":
    main()