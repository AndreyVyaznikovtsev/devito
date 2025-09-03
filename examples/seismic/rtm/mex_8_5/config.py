# config.py - Configuration variables for the LSRTM workflow
import numpy as np
from examples.seismic.datasets import VelocityModel, SeismogramDataset
from examples.seismic import SeismicModel

# Paths
PATH_DATA_BASE = "/home/andrey/devito-vti/examples/seismic/data"
OUTPUT_BASE = "/home/andrey/devito-vti/examples/seismic/rtm" + "/mex_8_5"

PATH_MODEL = PATH_DATA_BASE + "/North_04-05.dat" # Pr2
PATH_DATA_DPLUS = PATH_DATA_BASE + "/08-05_2.sgy"
PATH_DATA_DSUB = PATH_DATA_BASE + "/08-05_dsub_2.sgy"
# PATH_WAVELETS = OUTPUT_BASE + "/adjoint_wavelets"
PATH_WAVELETS = OUTPUT_BASE + "/minphase_wavelets"

# Processing parameters
NUM_SHOTS = 47
SO = 4
WAVELET = "Ricker"
NBL = 500
FS = False
SUBSAMPLING = 5
NSNAPS = 500
dtype = np.float32
NITER = 10
TMAX = 30.0
WAVELETS_DT = 0.003272
WAVELETS_TMAX = 35.0
SHOT_IDS = range(15, NUM_SHOTS)
NUM_X = 585
NUM_Z = 1985

INVERT_WELLS = True

min_sl = 1e-3
leaky = -1
FK_PARAMS = {
    'dx': 0.03 * SUBSAMPLING,
    'dz': 0.03 * SUBSAMPLING,
    'dt': TMAX/NSNAPS,  # Will be set during runtime
    'sigma_x': 0.2,
    'sigma_z': 0.2,
    'min_slope_down': -min_sl,
    'max_slope_down': leaky,
    'min_slope_up': min_sl,
    'max_slope_up': leaky,
    'gaussian_sigma': 0,
    'lower_min': 0.0,
    'upper_min': 0.0,
    'low_cut': 0.05,
    'high_cut': 1.2,
    'low_slope': 6,
    'high_slope': 2
}

OUTPUT_DIRS = {
    'forward_snaps': OUTPUT_BASE + "/forward_snaps",
    'adjoint_snaps': OUTPUT_BASE + "/adjoint_snaps",
    'gradients': OUTPUT_BASE + "/gradients",
    'images': OUTPUT_BASE + "/images",
    'results': OUTPUT_BASE + "/logs",
}

def setup_model_and_geometry(iter, external=None):
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
    # velmodel.pad_left(3)
    velmodel.pad_bottom(4)

    vp = velmodel.vp.T
    vp[vp<=0.6] = 0.6
    assert(vp.shape[0] == NUM_X)
    assert(vp.shape[1] == NUM_Z)

    if iter == 0:
        import os
        os.makedirs(OUTPUT_DIRS["images"], exist_ok=True)
        np.save(f"{OUTPUT_DIRS['images']}/vp_iter_0.npy", vp)
    else:
        vp = np.load(f"{OUTPUT_DIRS['images']}/vp_iter_{iter}.npy")
    if external is not None:
        vp = np.load(f"{OUTPUT_DIRS['images']}/vp_iter_{external}.npy")

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
