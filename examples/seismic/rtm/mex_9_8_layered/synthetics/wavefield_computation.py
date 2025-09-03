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
from examples.seismic.acoustic import EikonalSolver



def compute_wavefields(models, dataset, shot_id, eikonals):
    """Compute and save forward wavefield snaps"""
    t0 = 0.0
    tn = TMAX
    _, sx, sz, rec_x, rec_z = dataset[shot_id]
    
    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T

    model_nogal, model_gal = models
    geometry_nogal = AcquisitionGeometry(
        model_nogal, rec_pos, src_pos,
        t0, tn, f0=1.0,
        src_type="Ricker",
    )
    geometry_gal = AcquisitionGeometry(
        model_nogal, rec_pos, src_pos,
        t0, tn, f0=1.0,
        src_type="Ricker",
    )
    # d_syn_nogal = Receiver(name='d_syn_nogal', grid=model_nogal.grid, time_range=geometry_nogal.time_axis,
    #                  coordinates=geometry_nogal.rec_positions)
    
    # d_syn_gal = Receiver(name='d_syn_gal', grid=model_nogal.grid, time_range=geometry_gal.time_axis,
    #                  coordinates=geometry_gal.rec_positions)

    solver_nogal = AcousticWaveSolver(model_nogal, geometry_nogal, space_order=SO)
    solver_gal = AcousticWaveSolver(model_nogal, geometry_gal, space_order=SO)


    # solver_nogal.forward(vp=model_nogal.vp, rec=d_syn_nogal)
    # solver_gal.forward(vp=model_gal.vp, rec=d_syn_gal)
    # print(d_syn_gal.data.shape)
    _, u0, _ = solver_nogal.forward(vp=model_nogal.vp, save=True, nsnaps=NSNAPS, space_subsample=(SUBSAMPLING, SUBSAMPLING))
    np.save(f"{OUTPUT_DIRS['nogal']}/snap_{shot_id+1}.npy", u0.data[:, NBL//SUBSAMPLING:-NBL//SUBSAMPLING, NBL//SUBSAMPLING:-NBL//SUBSAMPLING])
    _, u1, _ = solver_gal.forward(vp=model_gal.vp, save=True, nsnaps=NSNAPS, space_subsample=(SUBSAMPLING, SUBSAMPLING))
    np.save(f"{OUTPUT_DIRS['gal']}/snap_{shot_id+1}.npy", u1.data[:, NBL//SUBSAMPLING:-NBL//SUBSAMPLING, NBL//SUBSAMPLING:-NBL//SUBSAMPLING])
    # d_syn_nogal.resample(dt=0.0625)
    # d_syn_gal.resample(dt=0.0625)
    # print(d_syn_gal.data.shape)
    # eikonal_nogal, eikonal_gal = eikonals
    # _, hod_nogal = eikonal_nogal.solve_single(shot_id)
    # _, hod_gal = eikonal_gal.solve_single(shot_id)
    # res = np.zeros((hod_nogal.size, 6))
    # res[:, 0] = sx
    # res[:, 1] = sz
    # res[:, 2] = rec_x
    # res[:, 3] = rec_z
    # res[:, 4] = hod_nogal
    # res[:, 5] = hod_gal

    # np.save(f"{OUTPUT_DIRS['nogal']}/gather_{shot_id+1}.npy", d_syn_nogal.data[:])
    # np.save(f"{OUTPUT_DIRS['gal']}/gather_{shot_id+1}.npy", d_syn_gal.data[:])
    # np.save(f"{OUTPUT_DIRS['hods']}/hods_{shot_id+1}.npy", res)




def main():
    os.makedirs(OUTPUT_DIRS['nogal'], exist_ok=True)
    os.makedirs(OUTPUT_DIRS['gal'], exist_ok=True)
    os.makedirs(OUTPUT_DIRS['hods'], exist_ok=True)


    model_nogal, model_gal, dataset, velmodel, eikonal_nogal, eikonal_gal = setup_model_and_geometry(PATH_DATA_DSUB)

    print(np.unique(dataset.x_coords))
    print(model_nogal.critical_dt)
    print(model_gal.critical_dt)

    # for i in range(len(dataset)):
    for i in range(10):
        compute_wavefields([model_nogal, model_gal], dataset, i, [eikonal_nogal, eikonal_gal])
        if i % 5 == 0:
            print('\033[1m' + f'{i+1}. Done' + '\033[0m')

if __name__== "__main__":
    main()