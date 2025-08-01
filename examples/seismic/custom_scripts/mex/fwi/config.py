# config.py - Configuration variables for the LSRTM workflow
import numpy as np

# Paths
PATH_MODEL = "../../../data/South_ForMigr_2.dat"
PATH_DATA = "../../../data/21-20.sgy"
PATH_WAVELETS = "../conventional_wavelets"

# Processing parameters
SO = 4
WAVELET = "Ricker"
NBL = 500
NFKL = 100
FS = False
SUBSAMPLING = 10
NSNAPS = 500
dtype = np.float32
NITER = 10
TMAX = 40.0
WAVELETS_DT = 0.0030840000
WAVELETS_TMAX = 69.3780000000

# FK Filter parameters
FK_PARAMS = {
    'dx': 0.025 * SUBSAMPLING,
    'dz': 0.025 * SUBSAMPLING,
    'dt': 0.000138756,  # Will be set during runtime
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