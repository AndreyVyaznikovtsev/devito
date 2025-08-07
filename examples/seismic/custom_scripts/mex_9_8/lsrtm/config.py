# config.py - Configuration variables for the LSRTM workflow
import numpy as np
from examples.seismic.datasets import VelocityModel, SeismogramDataset
from examples.seismic import SeismicModel


# Paths
PATH_MODEL = "../../../data/North_07_08_09.dat"
PATH_DATA_DPLUS = "../../../data/09-08.sgy"
PATH_DATA_DSUB = "../../../data/09-08_dsub.sgy"
PATH_WAVELETS = "../adjoint_wavelets"

# Processing parameters
NUM_SHOTS = 53
SO = 4
WAVELET = "Ricker"
NBL = 500
NFKL = 100
FS = False
SUBSAMPLING = 5
NSNAPS = 500
dtype = np.float32
NITER = 10
TMAX = 40.0
WAVELETS_DT = 0.003053
WAVELETS_TMAX = 40.0
SHOT_IDS = range(0, NUM_SHOTS, 10)
NUM_X = 805
NUM_Z = 2005

# FK Filter parameters
FK_PARAMS = {
    'dx': 0.03 * SUBSAMPLING,
    'dz': 0.03 * SUBSAMPLING,
    'dt': 0.003053,  # Will be set during runtime
    'sigma_x': 1,
    'sigma_z': 1,
    'min_slope_down': -1e9,
    'max_slope_down': -1e-4,
    'min_slope_up': 1e-4,
    'max_slope_up': 1e9,
    'gaussian_sigma': 0.001,
    'lower_min': 0.0,
    'upper_min': 0.0
}

# Output directories
OUTPUT_DIRS = {
    'forward_snaps': "./forward_snaps",
    'adjoint_snaps': "./adjoint_snaps",
    'gradients': "./gradients",
    'images': "./images",
    'results': "./logs"
}

def setup_model_and_geometry(path_data):
    """Set up the velocity model and acquisition geometry"""
    dataset = SeismogramDataset(path_data, "sou", invert_elevs=True)
    xmin, xmax = min(dataset.x_coords.min(), dataset.opposite_x.min()), max(dataset.x_coords.max(), dataset.opposite_x.max())
    spacing = (0.03, 0.03)
    velmodel = VelocityModel(
        PATH_MODEL,
        dx=spacing[0],
        dz=spacing[1],
        clip=True,
        xmin=xmin - 3,
        xmax=xmax + 3,
    )
    velmodel.pad_left(3)
    velmodel.pad_bottom(2)

    vp = velmodel.vp.T
    vp[vp<=0.6] = 0.6
    assert(vp.shape[0] == NUM_X)
    assert(vp.shape[1] == NUM_Z)

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
    return model, dataset, velmodel