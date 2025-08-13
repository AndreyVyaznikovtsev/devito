# config.py - Configuration variables for the LSRTM workflow
import numpy as np
from examples.seismic.datasets import VelocityModel, SeismogramDataset
from examples.seismic import SeismicModel


# Paths
PATH_MODEL = "../../../data/61_80_68_5Oc_LS_Ani.dat"
PATH_DATA_DPLUS = "../../../data/80-61.sgy"
PATH_DATA_DSUB = "../../../data/80-61_dsub.sgy"
PATH_WAVELETS = "../adjoint_wavelets"

# Processing parameters
NUM_SHOTS = 128
SO = 4
WAVELET = "Ricker"
NBL = 500
NFKL = 100
FS = True
SUBSAMPLING = 5
NSNAPS = 500
dtype = np.float32
NITER = 10
TMAX = 100.0
WAVELETS_DT = 0.009189
WAVELETS_TMAX = 50.0
SHOT_IDS = range(0, NUM_SHOTS, 10)
NUM_X = 885
NUM_Z = 3270

# FK Filter parameters
FK_PARAMS = {
    'dx': 0.05 * SUBSAMPLING,
    'dz': 0.05 * SUBSAMPLING,
    'dt': 0.009189,  # Will be set during runtime
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
    spacing = (0.05, 0.05)
    velmodel = VelocityModel(
        PATH_MODEL,
        dx=spacing[0],
        dz=spacing[1],
        # clip=True,
        xmin=xmin - 3,
        xmax=xmax + 3,
        invert_elevs=True
    )
    velmodel.pad_left(4)
    velmodel.pad_bottom(9)

    vp = velmodel.vp.T
    vp[vp<1.2] = 1.2
    eps = velmodel.epsilon.T
    delta = velmodel.delta.T
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
        fs=FS,
        vti=True,
        epsilon=eps,
        delta=delta

    )
    return model, dataset, velmodel