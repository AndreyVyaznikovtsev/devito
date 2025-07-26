from examples.seismic.datasets import VelocityModel, SeismogramDataset
import numpy as np
from examples.seismic.acoustic import AcousticWaveSolver
from matplotlib import pyplot as plt
from examples.seismic import SeismicModel, AcquisitionGeometry
import time
from devito import info
from examples.seismic.plotting import overlay_wiggle_plot
from devito import Function, norm
from examples.seismic import Receiver
from devito import Eq, Operator
from examples.seismic import plot_image
from devito import mmax

PATH_MODEL = "../data/South_ForMigr_2.dat"
PATH_DATA = path = "../data/21-20.sgy"
SO = 4
NBL = 100

# Computes the residual between observed and synthetic data into the residual
def compute_residual(residual, dobs, dsyn):
    # A simple data difference is enough in serial
    residual.data[:] = dsyn.data[:] - dobs.T
    return residual

def fwi_gradient(vp_in, model, geometry, dataset, nshots, solver):    
    # Create symbols to hold the gradient
    grad = Function(name="grad", grid=model.grid)
    # Create placeholders for the data residual and data
    residual = Receiver(name='residual', grid=model.grid,
                        time_range=geometry.time_axis, 
                        coordinates=geometry.rec_positions)
    d_syn = Receiver(name='d_syn', grid=model.grid,
                     time_range=geometry.time_axis, 
                     coordinates=geometry.rec_positions)
    objective = 0.
    for i in range(nshots):
        # Update source location
        d_obs, sx, sz, rec_x, rec_z = dataset[i]
        d_obs *= -1
        sz *= -1
        
        geometry.src_positions[0, :] = np.array([sx, sz])[None, :]

        # Compute smooth data and full forward wavefield u0
        _, u0, _ = solver.forward(vp=vp_in, save=True, rec=d_syn)
        
        compute_residual(residual, d_obs, d_syn)
        objective += .5*norm(residual)**2
        solver.gradient(rec=residual, u=u0, vp=vp_in, grad=grad, checkpointing=True)
    
    return objective, grad


def main():
    dataset = SeismogramDataset(PATH_DATA, "rec")
    # for i in range(len(dataset)):
    #     dataset.plot_spectrum_map(i, db_scale=False, n_bins=250, figsize=(8, 2), max_freq=8000, quant=1, cmap='jet')
    xmin, xmax = min(dataset.x_coords.min(), dataset.opposite_x.min()), max(dataset.x_coords.max(), dataset.opposite_x.max())
    print(xmin, xmax)

    spacing = (0.1, 0.1)
    velmodel = VelocityModel(PATH_MODEL, dx=spacing[0], dz=spacing[1], clip=True, xmin=xmin-3, xmax=xmax+3, zmin=-318)
    velmodel.pad_left(4)
    velmodel.pad_right(8*int(0.5/0.1))
    velmodel.pad_bottom(10*int(0.5/0.1))
    velmodel.pad_top(7*int(0.5/0.1))

    vp = velmodel.vp.T
    print(np.max(vp))
    print(np.min(vp))

    dx_critical = 500/(10*2000)
    print(dx_critical)

    # fig, axs = velmodel.plot_vp(show=False, figsize=(9, 5), dpi=100)
    # axs[0].scatter(dataset.x_coords, -dataset.elevations, c='k', s=2)
    # axs[0].scatter(dataset.opposite_x, -dataset.opposite_elev, c='k', s=2)
    # plt.show()

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
    t0 = 0
    tn = dataset._t_max
    tn = 40.0
    dataset.dt_r = model.critical_dt
    dataset.t_max_r = tn
    dataset.resample_on()
    
    d_2, sx, sz, rec_x, rec_z = dataset[0]
    rec_z *= -1
    sz *= -1
    src_pos = np.array([sx, sz])[None, :]
    rec_pos = np.vstack([rec_x, rec_z]).T
    f0=0.4

    geometry =  AcquisitionGeometry(model, rec_pos, src_pos, t0, tn, f0=f0*2, src_type="Gabor")
    solver = AcousticWaveSolver(model, geometry, space_order=4)

    ff, update = fwi_gradient(model.vp, model, geometry, dataset, len(dataset), solver)
    alpha = .5 / mmax(update)
    plot_image(-update.data, cmap="jet")
    plot_image(model.vp.data + alpha*update.data, cmap="jet")
    # smooth_d, _, _ = solver.forward(vp=model.vp)

    # fig, ax = overlay_wiggle_plot(
    #     np.array(smooth_d.data[:]), d_2.T, time_axis=geometry.time_axis.time_values, xrec=rec_z, title="Original vs Processed"
    # )
    # plt.savefig(f"mex/Forward + Initial.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    # plt.close()

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    info(f"Execution time {end - start:.2f} seconds")
