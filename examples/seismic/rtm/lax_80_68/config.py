# config.py - Configuration variables for the LSRTM workflow
import numpy as np
from examples.seismic.datasets import VelocityModel, SeismogramDataset
from examples.seismic import SeismicModel
from devito import gaussian_smooth

PATH_DATA_BASE = "/home/andrey/devito-vti/examples/seismic/data"
OUTPUT_BASE = "/home/andrey/devito-vti/examples/seismic/rtm" + "/lax_80_68"


PATH_MODEL = PATH_DATA_BASE + "/61_80_68_5Oc_LS_Ani.dat"
PATH_DATA_DPLUS = PATH_DATA_BASE + "/80-68.sgy"
PATH_DATA_DSUB = PATH_DATA_BASE + "/80-68_dsub.sgy"
# PATH_WAVELETS = OUTPUT_BASE + "/adjoint_wavelets"
# PATH_WAVELETS = OUTPUT_BASE + "/minphase_wavelets"
PATH_WAVELETS = OUTPUT_BASE + "/minphase_wavelets"

# Processing parameters
NUM_SHOTS = 107
SO = 4
WAVELET = "Ricker"
NBL = 500
NFKL = 100
FS = False
SUBSAMPLING = 5
NSNAPS = 500
dtype = np.float32
NITER = 10
TMAX = 100.0
WAVELETS_DT = 0.009329
WAVELETS_TMAX = 50.0
SHOT_IDS = range(0, NUM_SHOTS, 10)
NUM_X = 885
# NUM_Z = 3270
NUM_Z = 3270
ISO=False


min_sl = 1e-3
leaky = -1
# FK Filter parameters
FK_PARAMS = {
    'dx': 0.05 * SUBSAMPLING,
    'dz': 0.05 * SUBSAMPLING,
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
    'high_cut': 1.5,
    'low_slope': 6,
    'high_slope': 6
}

OUTPUT_DIRS = {
    'forward_snaps': OUTPUT_BASE + "/forward_snaps",
    'adjoint_snaps': OUTPUT_BASE + "/adjoint_snaps",
    'gradients': OUTPUT_BASE + "/gradients",
    'images': OUTPUT_BASE + "/images",
    'results': OUTPUT_BASE + "/logs",
}

def setup_model_and_geometry(path_data):
    """Set up the velocity model and acquisition geometry"""
    dataset_dplus = SeismogramDataset(PATH_DATA_DPLUS, "sou", invert_elevs=True)
    dataset = SeismogramDataset(PATH_DATA_DSUB, "sou", invert_elevs=True)
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
    velmodel.pad_left(3)
    velmodel.pad_bottom(9)

    vp = velmodel.vp.T
    vp[vp<1.2] = 1.2
    eps = velmodel.epsilon.T
    delta = 1.2 * velmodel.epsilon.T
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
    gaussian_smooth(model.vp, sigma=(3, 40))
    gaussian_smooth(model.epsilon, sigma=(3, 40))
    gaussian_smooth(model.delta, sigma=(3, 40))


    return model, dataset_dplus, dataset, velmodel