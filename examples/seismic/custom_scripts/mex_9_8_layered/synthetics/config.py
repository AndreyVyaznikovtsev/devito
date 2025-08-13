# config.py - Configuration variables for the LSRTM workflow
import numpy as np
from examples.seismic.datasets import VelocityModel, SeismogramDataset
from examples.seismic import SeismicModel
from examples.seismic.acoustic import EikonalSolver


# Paths
PATH_MODEL = "../../../data/North_07_08_09.dat"
PATH_DATA_DSUB = "../../../data/09-08_dsub.sgy"
PATH_DATA_NOGAL = "../../../data/09-08_nogal.sgy"
PATH_DATA_GAL = "../../../data/09-08_gal.sgy"

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
TMAX = 50.0
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
    'nogal': "./nogal",
    'gal': "./gal",
    'hods': "./hods",
    'gradients': "./gradients",
    'images': "./images",
    'results': "./logs"
}

def setup_model_and_geometry(path_data):
    """Set up the velocity model and acquisition geometry"""
    dataset = SeismogramDataset(path_data, "sou", invert_elevs=True, monkey_x=True)

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

    layered = velmodel.create_layered_vp([[[8, 32], [-318, -318]],
                                        [[8, 32], [-317, -315.5]],
                                        [[8, 32], [-314, -314]],
                                        [[8, 32], [-310, -311.5]],
                                        [[8, 32], [-304.5, -304.5]],
                                        [[8, 32], [-301, -301]],
                                        [[8, 32], [-296]*2],
                                        [[8, 32], [-291]*2],
                                        [[8, 32], [-287.5]*2],
                                        [[8, 32], [-279]*2],
                                        [[8, 32], [-274]*2],
                                        [[8, 32], [-271]*2],
                                        [[8, 32], [-269]*2],
                                        [[8, 32], [-265]*2]])

    origin = velmodel.x[0], velmodel.z[0]

    nz, nx = layered.shape
    layered_gal = np.copy(layered)
    layered_gal[nz - nz//4 - 67:nz - nz//4, int(nx/2.7)-33:int(nx/2.7)+33] = 1.5

    velmodel._current_model["vel"] = layered
    eikonal_nogal = EikonalSolver(velmodel, dataset)

    velmodel._current_model["vel"] = layered_gal
    eikonal_gal = EikonalSolver(velmodel, dataset)

    model_nogal = SeismicModel(
        vp=layered.T,
        origin=origin,
        shape=vp.shape,
        spacing=spacing,
        space_order=SO,
        nbl=NBL,
        bcs="damp",
        fs=False,
    )
    model_gal = SeismicModel(
        vp=layered_gal.T,
        origin=origin,
        shape=vp.shape,
        spacing=spacing,
        space_order=SO,
        nbl=NBL,
        bcs="damp",
        fs=False,
    )
    return model_nogal, model_gal, dataset, velmodel, eikonal_nogal, eikonal_gal