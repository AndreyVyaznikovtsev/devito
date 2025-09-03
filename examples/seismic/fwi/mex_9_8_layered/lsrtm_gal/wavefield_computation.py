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

    geometry = AcquisitionGeometry(
        model, rec_pos, src_pos,
        t0, tn, f0=0.5,
        src_type='Ricker'
    )
    d_syn = Receiver(name='d_syn', grid=model.grid, time_range=geometry.time_axis,
                     coordinates=geometry.rec_positions)
    solver = AcousticWaveSolver(model, geometry, space_order=SO)
    _, u0, _ = solver.forward(vp=model.vp, save=True, nsnaps=NSNAPS, rec=d_syn, space_subsample=(SUBSAMPLING, SUBSAMPLING))
    np.save(f"{OUTPUT_DIRS['forward_snaps']}/{shot_id+1}.npy", u0.data[:, NBL//SUBSAMPLING:-NBL//SUBSAMPLING, NBL//SUBSAMPLING:-NBL//SUBSAMPLING])


def compute_wavefields(model_sm, model_true, dataset, shot_id):
    """Compute and save forward wavefield snaps"""
    objective = 0.0
    t0 = 0.0
    tn = TMAX
    d_obs, sx, sz, rec_x, rec_z = dataset[shot_id]
    
    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T

    geometry = AcquisitionGeometry(
        model_true, rec_pos, src_pos,
        t0, tn, f0=0.5,
        src_type='Ricker',
    )
    d_obs = Receiver(name='d_obs', grid=model_true.grid, time_range=geometry.time_axis,
                     coordinates=geometry.rec_positions)
    d_syn = Receiver(name='d_syn', grid=model_sm.grid, time_range=geometry.time_axis,
                     coordinates=geometry.rec_positions)
    solver = AcousticWaveSolver(model_true, geometry, space_order=SO)
    solver.forward(vp=model_true.vp, rec=d_obs)
    # solver.forward(vp=model_sm.vp, rec=d_obs)
    
    _, u0, _ = solver.forward(vp=model_sm.vp, rec=d_syn, save=True, nsnaps=NSNAPS, space_subsample=(SUBSAMPLING, SUBSAMPLING))
    # plt.imshow(np.array(d_obs.data[:]).T, aspect='auto', vmin=-1e-5, vmax=1e-5)
    # plt.show()
    # plt.imshow(np.array(d_syn.data[:]).T, aspect='auto', vmin=-1e-5, vmax=1e-5)
    # plt.show()
    # plt.imshow(np.array(d_syn.data[:] - d_obs.data[:]).T, aspect='auto', vmin=-1e-5, vmax=1e-5)
    # plt.show()
    residual = Receiver(name='residual', grid=model_true.grid, time_range=geometry.time_axis,
                    coordinates=geometry.rec_positions)
    residual.data[:] = d_syn.data[:] - d_obs.data[:]
    
    _, v, _ = solver.adjoint(vp=model_true.vp, rec=residual, save=True, nsnaps=NSNAPS, space_subsample=(SUBSAMPLING, SUBSAMPLING))
    np.save(f"{OUTPUT_DIRS['forward_snaps']}/{shot_id+1}.npy", u0.data[:, NBL//SUBSAMPLING:-NBL//SUBSAMPLING, NBL//SUBSAMPLING:-NBL//SUBSAMPLING])
    np.save(f"{OUTPUT_DIRS['adjoint_snaps']}/{shot_id+1}.npy", v.data[:, NBL//SUBSAMPLING:-NBL//SUBSAMPLING, NBL//SUBSAMPLING:-NBL//SUBSAMPLING])
    objective = 0.5*norm(residual)**2
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

    model_sm, model_true, dataset, _ = setup_truemodel_and_geometry(PATH_DATA_GAL)
    dataset._dt_r = model_true.critical_dt
    dataset._t_max_r = TMAX
    dataset.resample_on()
    print("Num samples :", dataset._t_max_r/model_true.critical_dt)
    log_file = os.path.join(OUTPUT_DIRS['results'], 'optimization.log')
    if args.iter == 0:
        with open(log_file, 'a') as f:
            f.write(f"LSRTM experiment at {datetime.now()}, recon - {args.recon}, inv - {args.inv}" + '\n')

    
    # if args.iter == 0:
    #     for i in range(len(dataset)):
    #         compute_forward_snaps(model, dataset, i)
    
    dm = load_current_dm(args.iter)

    objective = 0.
    for i in range(len(dataset)):
    # for i in SHOT_IDS:
        dmin = Function(name='dm', grid=model_true.grid)
        dmin.data[:] = dm


        obj_shot = compute_wavefields(model_sm, model_true, dataset, i)
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