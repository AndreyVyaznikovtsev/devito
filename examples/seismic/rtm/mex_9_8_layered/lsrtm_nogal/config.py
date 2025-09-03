# config.py - Configuration variables for the LSRTM workflow
import numpy as np
from examples.seismic.datasets import VelocityModel, SeismogramDataset
from examples.seismic import SeismicModel
from examples.seismic.acoustic import EikonalSolver
from devito import gaussian_smooth

# Paths
PATH_MODEL = "/home/andrey/devito-vti/examples/seismic/data/9_8_synth.dat"
PATH_MODEL_TRUE = "/home/andrey/devito-vti/examples/seismic/data/North_07_08_09.dat"
PATH_DATA_DSUB = "/home/andrey/devito-vti/examples/seismic/data/09-08_dsub.sgy" #real one
PATH_DATA_NOGAL = "/home/andrey/devito-vti/examples/seismic/data/09-08_nogal.sgy"
PATH_DATA_GAL = "/home/andrey/devito-vti/examples/seismic/data/09-08_gal.sgy"

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
TMAX = 35.0
SHOT_IDS = range(0, NUM_SHOTS, 10)
NUM_X = 805
NUM_Z = 2005

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
    'cutoff': 0.9,
    'order': 6
}
# Output directories
OUTPUT_DIRS = {
    'forward_snaps' : "/mnt/myshare/Андрей/!Аспирантура/synth_nogal/forward_snaps",
    'adjoint_snaps' : "/mnt/myshare/Андрей/!Аспирантура/synth_nogal/adjoint_snaps",
    'nogal': "/mnt/myshare/Андрей/!Аспирантура/synth_nogal/nogal",
    'gal': "/mnt/myshare/Андрей/!Аспирантура/synth_nogal/gal",
    'hods': "/mnt/myshare/Андрей/!Аспирантура/synth_nogal/hods",
    'gradients': "/mnt/myshare/Андрей/!Аспирантура/synth_nogal/gradients",
    'images': "/mnt/myshare/Андрей/!Аспирантура/synth_nogal/images",
    'results': "/mnt/myshare/Андрей/!Аспирантура/synth_nogal/logs"
}

def setup_model_and_geometry(path_data):
    """Set up the velocity model and acquisition geometry"""
    dataset = SeismogramDataset(path_data, "sou", invert_elevs=False)

    xmin, xmax = min(dataset.x_coords.min(), dataset.opposite_x.min()), max(dataset.x_coords.max(), dataset.opposite_x.max())
    spacing = (0.03, 0.03)
    velmodel = VelocityModel(
        PATH_MODEL,
        dx=spacing[0],
        dz=spacing[1],
        clip=False,
        xmin=xmin - 3,
        xmax=xmax + 3,
    )
    velmodel.pad_left(5)
    velmodel.pad_top(50)
    velmodel.pad_bottom(6)

    vp = velmodel.vp.T
    print(vp.shape)
    
    # vp[vp<=0.6] = 0.6
    assert(vp.shape[0] == NUM_X)
    assert(vp.shape[1] == NUM_Z)

    origin = velmodel.x[0], velmodel.z[0]
    print(origin)
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


def setup_truemodel_and_geometry(path_data):
    """Set up the velocity model and acquisition geometry"""
    dataset = SeismogramDataset(path_data, "sou", invert_elevs=False)

    xmin, xmax = min(dataset.x_coords.min(), dataset.opposite_x.min()), max(dataset.x_coords.max(), dataset.opposite_x.max())
    spacing = (0.03, 0.03)
    velmodel = VelocityModel(
        PATH_MODEL_TRUE,
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
    print(vp.shape)
    assert(vp.shape[0] == 805)
    assert(vp.shape[1] == 2005)

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
    layered_gal[nz - nz//6 - 67+55:nz - nz//6+55, int(nx/2.7)-33:int(nx/2.7)+33] = 1.5

    # velmodel._current_model["vel"] = layered
    # eikonal_nogal = EikonalSolver(velmodel, dataset)

    # velmodel._current_model["vel"] = layered_gal
    # eikonal_gal = EikonalSolver(velmodel, dataset)

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
    gaussian_smooth(model_nogal.vp, sigma=(20, 20))
    # model_gal = SeismicModel(
    #     vp=layered_gal.T,
    #     origin=origin,
    #     shape=vp.shape,
    #     spacing=spacing,
    #     space_order=SO,
    #     nbl=NBL,
    #     bcs="damp",
    #     fs=False,
    # )
    return model_nogal, dataset, velmodel