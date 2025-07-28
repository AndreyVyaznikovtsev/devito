import os
import time
import numpy as np
from matplotlib import pyplot as plt
from examples.seismic.utils import taper_wavelet, estimate_centroid_frequency_gather
from examples.seismic import SeismicModel, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.plotting import plot_two_wavelets, overlay_wiggle_plot
from examples.seismic.datasets import SeismogramDataset, VelocityModel
from devito import info

PATH_MODEL = "../data/South_ForMigr_2.dat"
PATH_DATA = path = "../data/21-20.sgy"
SO = 4
NBL = 100
WAVELET = "Gabor"

def main():

    os.makedirs(f"mex/forward_initial_{WAVELET}", exist_ok=True)
    os.makedirs(f"mex/stfs_{WAVELET}", exist_ok=True)
    os.makedirs(f"mex/wavelets_{WAVELET}", exist_ok=True)
    os.makedirs(f"mex/forward_inverted_{WAVELET}", exist_ok=True)

    dataset = SeismogramDataset(PATH_DATA, "rec")
    xmin, xmax = min(dataset.x_coords.min(), dataset.opposite_x.min()), max(dataset.x_coords.max(), dataset.opposite_x.max())
    spacing = (0.025, 0.025)
    velmodel = VelocityModel(PATH_MODEL, dx=spacing[0], dz=spacing[1], clip=True, xmin=xmin-3, xmax=xmax+3, zmin=-318)
    velmodel.pad_left(4)
    velmodel.pad_right(8*int(0.5/spacing[0]))
    velmodel.pad_bottom(10*int(0.5/spacing[0]))
    velmodel.pad_top(7*int(0.5/spacing[0]))

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
    dataset.dt_r = model.critical_dt
    dataset.t_max_r = tn
    dataset.resample_on()

    f0 = 0.25

    geometry = AcquisitionGeometry(
        model, np.array([[origin[0], origin[1]]]).T, np.array([[origin[0], origin[1]]]).T, t0, tn, f0=f0, src_type=WAVELET
    )
    solver = AcousticWaveSolver(model, geometry, space_order=SO)

    for i in range(len(dataset)):
        # for i in range(0, len(dataset)):
        info(f"Inverting {i+1}-th gather wavelet")
        d_2, sx, sz, rec_x, rec_z = dataset[i]
        d_2 *= -1
        f0 = estimate_centroid_frequency_gather(d_2.T, model.critical_dt) / 1e3
        info(f"Estimated gather centroid frequency: {f0*1e3:.2f} Hz, Wavelet length: {1/f0:.2f} ms")

        sz *= -1
        rec_z *= -1
        src_pos = np.array([sx, sz])[None, :]
        rec_pos = np.vstack([rec_x, rec_z]).T
        f0 = f0 if WAVELET == "Ricker" else f0*2
        geometry = AcquisitionGeometry(model, rec_pos, src_pos, t0, tn, f0=f0*2, src_type=WAVELET)
        solver = AcousticWaveSolver(model, geometry, space_order=SO)

        d_1, _, _ = solver.forward(vp=model.vp, save=False)
        wav1 = geometry.src.data
        fig, ax = overlay_wiggle_plot(
            np.array(d_1.data[:]), d_2.T, time_axis=geometry.time_axis.time_values, xrec=rec_z, title="Original vs Processed"
        )
        plt.savefig(f"mex/forward_initial_{WAVELET}/Forward + Initial {i+1}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        stfs = []
        for norm in [True, False]:
            for kill in [True, False]:
                stf = wiener_deconvolution(
                    d_2.T, d_1.data, eps=1e-8, normalize=norm,
                    kill_offset=kill, sz=sz, rec_z=rec_z,
                    offset_threshold=10.0
                )

                wav3 = np.convolve(wav1.squeeze(), stf)[: d_1.shape[0]]
                tap_f0 = 1/f0 if WAVELET == "Ricker" else 2/f0
                wav3_tapered, taper = taper_wavelet(wav3, geometry.time_axis.time_values, tap_f0, 0.5 * tap_f0)
                normstr = "norm" if norm else "nonnorm"
                killstr = "kill" if kill else "nonkill"
                np.savetxt(f"mex/stfs_{WAVELET}/{normstr}_{killstr}_{i+1}.txt", wav3_tapered)

                geometry = AcquisitionGeometry(model, rec_pos, src_pos, t0, tn, f0=f0 * 2, src_type=None, wav_data=wav3_tapered)
                solver = AcousticWaveSolver(model, geometry, space_order=SO)
                d_3, _, _  = solver.forward(vp=model.vp, src=geometry.src, save=False)

                fig, axs = plot_two_wavelets(geometry.time_axis.time_values, wav1, wav3_tapered, taper)
                plt.savefig(f"mex/wavelets_{WAVELET}/{normstr}_{killstr}_{i+1}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
                plt.close()
                fig, ax = overlay_wiggle_plot(
                    np.array(d_3.data[:]), d_2.T, time_axis=geometry.time_axis.time_values, xrec=rec_z, title="Original vs Processed"
                )
                plt.savefig(f"mex/forward_inverted_{WAVELET}/{normstr}_{killstr}_{i+1}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
                plt.close()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    info(f"Execution time {end - start:.2f} seconds")
