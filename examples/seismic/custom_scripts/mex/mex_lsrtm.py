import time
import numpy as np
from examples.seismic import SeismicModel, AcquisitionGeometry, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.datasets import SeismogramDataset, VelocityModel
from devito import info, TimeFunction, Eq, Operator, norm
# from examples.seismic.fk_filter import FKFilter3D
from scipy.interpolate import interpn

# PATH_MODEL = "../../data/South_ForMigr_2.dat"
# PATH_DATA = path = "../../data/21-20.sgy"
PATH_MODEL = "../../data/South_ForMigr_2.dat"
PATH_DATA = path = "../../data/21-20.sgy"
SO = 4
WAVELET = "Ricker"
NBL = 500
NFKL = 100
FS = False
SUBSAMPLING = 10
slic = NBL - NFKL
slices = (
    (slice(None), slice(slic, -slic, SUBSAMPLING), slice(None, -slic, SUBSAMPLING))
    if FS
    else (slice(None), slice(slic, -slic, SUBSAMPLING), slice(slic, -slic, SUBSAMPLING))
)
NSNAPS = 500
NITER = 10
dtype = np.float32


def main():
    dataset = SeismogramDataset(PATH_DATA, "sou", invert_elevs=True)
    xmin, xmax = min(dataset.x_coords.min(), dataset.opposite_x.min()), max(dataset.x_coords.max(), dataset.opposite_x.max())
    spacing = (0.025, 0.025)
    velmodel = VelocityModel(
        PATH_MODEL,
        dx=spacing[0],
        dz=spacing[1],
        clip=True,
        xmin=xmin - 3,
        xmax=xmax + 3,
        zmin=-318,
    )
    velmodel.pad_left(4 + 2)
    velmodel.pad_right(8 * int(0.5 / spacing[0]) + 2)
    velmodel.pad_bottom(10 * int(0.5 / spacing[0]) + 2)
    velmodel.pad_top(7 * int(0.5 / spacing[0]))

    origin = velmodel.x[0], velmodel.z[0]
    vp = velmodel.vp.T
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
    t0 = 0.0
    tn = dataset._t_max
    dataset._dt_r = model.critical_dt
    dataset._t_max_r = tn
    dataset.resample_on()
    f0 = 0.25
    
    # fk_down = FKFilter3D(dx=model.spacing[0]*SUBSAMPLING, dz=model.spacing[1]*SUBSAMPLING,
    #             dt=(tn/NSNAPS)/1000, sigma_x=1, sigma_z=1, min_slope=-1e9, max_slope=-1e-4, gaussian_sigma=0.001, lower_min=0.0, upper_min=0.0)
    # fk_down._compute_filter(*snapsObj_f.T.shape)
    # fk_up = FKFilter3D(dx=model.spacing[0]*SUBSAMPLING, dz=model.spacing[1]*SUBSAMPLING,
    #                 dt=(tn/NSNAPS)/1000, sigma_x=1, sigma_z=1, min_slope=1e-4, max_slope=1e9, gaussian_sigma=0.001, lower_min=0.0, upper_min=0.0)
    # fk_up._compute_filter(*snapsObj_f.T.shape)


    nx, nz = model.grid.shape  # Original dimensions
    sub_nx = nx // SUBSAMPLING + 1
    sub_nz = nz // SUBSAMPLING + 1

    image_up_dev = np.zeros((sub_nx, sub_nz), dtype)
    image = np.zeros((sub_nx, sub_nz), dtype)
    history = np.zeros((NITER, 1))
    image_prev = np.zeros((sub_nx, sub_nz), dtype)
    grad_prev  = np.zeros((sub_nx, sub_nz), dtype)
    yk  = np.zeros((sub_nx, sub_nz), dtype)
    sk = np.zeros((sub_nx, sub_nz), dtype)
    
    for k in range(NITER):
        info(f'LSRTM Iteration {k+1}')
        dm = upsample_image(image_up_dev, nx_orig=nx, nz_orig=nz)
        objective, grad_full = lsrtm_gradient(dm=dm, model=model, dataset=dataset)
        history[k] = objective
        yk = grad_full - grad_prev
        sk = image_up_dev - image_prev
        alfa = get_alfa(yk,sk,k, grad_full=grad_full)
        grad_prev = grad_full
        image_prev = image_up_dev
        image_up_dev = image_up_dev - alfa*grad_full.data
        if k == 0: # Saving the first migration using Born operator.
            image = image_up_dev

def lsrtm_gradient(dm, model, dataset):
    nx, nz = model.grid.shape  # Original dimensions
    sub_nx = nx // SUBSAMPLING + 1
    sub_nz = nz // SUBSAMPLING + 1


    grad_full = np.zeros((sub_nx, sub_nz), dtype=dtype)
    grad_illum = np.zeros((sub_nx, sub_nz), dtype=dtype)
    src_illum = np.zeros((sub_nx, sub_nz), dtype=dtype)

    objective = 0.

    t0 = 0.0
    tn = dataset._t_max
    f0 = 0.3
    _, sx, sz, rec_x, rec_z = dataset[0]

    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T
    geometry = AcquisitionGeometry(model, rec_pos, src_pos,
                                    t0, tn, f0=f0, src_type=None,
                                    wav_data=np.load(f"conventional_wavelets/{i+1}.npy")*0
                                    )
    solver = AcousticWaveSolver(model, geometry, space_order=SO)
    for i in range(len(dataset)):
        d_obs, sx, sz, rec_x, rec_z = dataset[i]

        residual = Receiver(name='residual', grid=model.grid, time_range=geometry.time_axis,
                            coordinates=geometry.rec_positions)
        d_syn = Receiver(name='d_syn', grid=model.grid,time_range=geometry.time_axis,
                         coordinates=geometry.rec_positions)
        solver.jacobian(dm, vp=model.vp, rec = d_syn)
        print(d_syn.data[:].min(), d_syn.data[:].max())
        residual.data[:] = d_syn.data[:] - d_obs.T
        u0 = snap_fromfile(f"forward_snaps/{i+1}.bin", shape=(NSNAPS, sub_nx, sub_nz))
        _, v, _ = solver.adjoint(vp=model.vp, rec=residual, save=True, nsnaps=NSNAPS)
        grad_shot = calc_grad_full_numpy(u0, v.data[:], model.critical_dt*geometry.time_axis.time_values.size/NSNAPS)
        
        src_illum += np.sum(u0**2, axis=0)
        grad_full += grad_shot

        objective += .5*norm(residual)**2
        
    grad_illum = grad_full/(src_illum+1e-9)
     
    return objective, grad_illum

def calc_grad_full_numpy(u0, v_data, dt):
    """
    Pure NumPy gradient computation with finite-difference dt2.
    
    Args:
        u0 (np.ndarray): Forward wavefield (nt, nx, nz)
        v_data (np.ndarray): Adjoint wavefield (nt, nx, nz)
        dt (float): Time step size
        
    Returns:
        np.ndarray: Gradient (nx, nz)
    """
    # Pad v_data with zeros at t=-1 and t=nt
    v_padded = np.pad(v_data, ((1, 1), (0, 0), (0, 0)), mode='constant')
    
    # Compute second time derivative (centered FD)
    v_dt2 = (v_padded[2:] - 2*v_padded[1:-1] + v_padded[:-2]) / (dt**2)
    
    # Gradient = -∑(u0 * v_dt2) over time
    return -np.sum(u0 * v_dt2, axis=0)

def snap_fromfile(path, shape, back=False):
    fobj = open(path, "rb")
    snapsObj_f = np.fromfile(fobj, dtype=np.float32)
    snapsObj_f = np.reshape(snapsObj_f, shape)
    if back:
        snapsObj_f = np.copy(snapsObj_f[::-1])
    fobj.close()
    return snapsObj_f

def upsample_image(image_subsampled, nx_orig, nz_orig):
    """
    Upsample a subsampled image to original dimensions using bilinear interpolation.
    
    Args:
        image_subsampled (np.ndarray): Subsampled image of shape (sub_nx, sub_nz).
        nx_orig (int): Original x-dimension.
        nz_orig (int): Original z-dimension.
    
    Returns:
        np.ndarray: Upsampled image of shape (nx_orig, nz_orig).
    """
    sub_nx, sub_nz = image_subsampled.shape
    
    # Coordinates of the subsampled grid
    x_coarse = np.linspace(0, nx_orig - 1, sub_nx)
    z_coarse = np.linspace(0, nz_orig - 1, sub_nz)
    
    # Coordinates of the fine grid
    x_fine = np.arange(nx_orig)
    z_fine = np.arange(nz_orig)
    
    X_fine, Z_fine = np.meshgrid(x_fine, z_fine, indexing='ij')
    points_fine = np.column_stack([X_fine.ravel(), Z_fine.ravel()])
    
    # Interpolate
    image_upsampled = interpn(
        (x_coarse, z_coarse),
        image_subsampled,
        points_fine,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    ).reshape(nx_orig, nz_orig)
    
    return image_upsampled

def get_alfa(grad_iter, image_iter, niter_lsrtm, grad_full):
    term1 = np.dot(image_iter.reshape(-1), image_iter.reshape(-1))
    term2 = np.dot(image_iter.reshape(-1), grad_iter.reshape(-1))
    term3 = np.dot(grad_iter.reshape(-1), grad_iter.reshape(-1))
    if niter_lsrtm == 0:
        alfa = .05 / np.max(grad_full)
    else:
        abb1 = term1 / term2
        abb2 = term2 / term3
        abb3 = abb2 / abb1
        if abb3 > 0 and abb3 < 1:
            alfa = abb2
        else:
            alfa = abb1
            
    return alfa  

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    info(f"Execution time {end - start:.2f} seconds")
