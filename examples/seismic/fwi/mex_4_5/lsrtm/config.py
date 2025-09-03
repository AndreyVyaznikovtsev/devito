# ./rtm/mex_4_5/config.py
import numpy as np
from examples.seismic.datasets import VelocityModel, SeismogramDataset
from examples.seismic import SeismicModel

# Paths
PATH_DATA_BASE = "/home/andrey/devito-vti/examples/seismic/data"
OUTPUT_BASE = "/mnt/myshare/Андрей/!Аспирантура" + "/mex_4_5"

PATH_MODEL = PATH_DATA_BASE + "/North_04-05.dat"
PATH_DATA_DPLUS = PATH_DATA_BASE + "/04-05.sgy"
PATH_DATA_DSUB = PATH_DATA_BASE + "/04-05_dsub.sgy"
PATH_WAVELETS = OUTPUT_BASE + "/adjoint_wavelets"

# Processing parameters
NUM_SHOTS = 50
SO = 4
WAVELET = "Ricker"
NBL = 500
FS = False
SUBSAMPLING = 5
NSNAPS = 500
dtype = np.float32
NITER = 10
TMAX = 40.0
WAVELETS_DT = 0.003235
WAVELETS_TMAX = TMAX
SHOT_IDS = range(0, NUM_SHOTS, 10)
NUM_X = 815
NUM_Z = 1980

# FK Filter parameters
FK_PARAMS = {
    'dx': 0.03 * SUBSAMPLING,
    'dz': 0.03 * SUBSAMPLING,
    'dt': TMAX/NSNAPS,  # Will be set during runtime
    'sigma_x': 1,
    'sigma_z': 1,
    'min_slope_down': -0.5e-1,
    'max_slope_down': -1e-5,
    'min_slope_up': 1e-5,
    'max_slope_up': 0.5e-1,
    'gaussian_sigma': 1e-5,
    'lower_min': 0.0,
    'upper_min': 0.0,
    'low_cut': 0.05,
    'high_cut': 1.5,
    'low_slope': 6,
    'high_slope': 2

}

# Output directories

OUTPUT_DIRS = {
    'forward_snaps': OUTPUT_BASE + "/forward_snaps",
    'adjoint_snaps': OUTPUT_BASE + "/adjoint_snaps",
    'gradients': OUTPUT_BASE + "/gradients",
    'images': OUTPUT_BASE + "/images",
    'results': OUTPUT_BASE + "/logs",
}

def setup_model_and_geometry(iter):
    """Set up the velocity model and acquisition geometry"""
    dataset_dplus = SeismogramDataset(PATH_DATA_DPLUS, "sou", invert_elevs=True)
    dataset_dsub = SeismogramDataset(PATH_DATA_DSUB, "sou", invert_elevs=True)

    xmin, xmax = min(dataset_dplus.x_coords.min(), dataset_dplus.opposite_x.min()), max(dataset_dplus.x_coords.max(), dataset_dplus.opposite_x.max())
    spacing = (0.03, 0.03)
    velmodel = VelocityModel(
        PATH_MODEL,
        dx=spacing[0],
        dz=spacing[1],
        clip=True,
        xmin=xmin - 3,
        xmax=xmax + 3,
    )
    vp = velmodel.vp.T[3:, 1:]
    assert(vp.shape[0] == NUM_X)
    assert(vp.shape[1] == NUM_Z)

    # Save initial vp as image iteration 0
    if iter == 0:
        import os
        os.makedirs(OUTPUT_DIRS["images"], exist_ok=True)
        np.save(f"{OUTPUT_DIRS['images']}/vp_iter_0.npy", vp)
    else:
        vp = np.load(f"{OUTPUT_DIRS['images']}/vp_iter_{iter}.npy")

    origin = velmodel.x[0], velmodel.z[0]
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
    return model, dataset_dplus, dataset_dsub, velmodel

