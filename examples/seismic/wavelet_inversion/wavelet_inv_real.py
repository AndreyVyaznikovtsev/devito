import time
import numpy as np
from matplotlib import pyplot as plt
from examples.seismic.utils import wiener_deconvolution, taper_wavelet, estimate_centroid_frequency_gather
from examples.seismic import SeismicModel, AcquisitionGeometry
from examples.seismic.vti import VTIWaveSolver
from examples.seismic.plotting import plot_two_wavelets, overlay_wiggle_plot
from examples.seismic.datasets import SeismogramDataset, VelocityModel
from devito import info

PATH_DATA = path = "../datasets/80_61_PreReady.sgy"
PATH_MODEL = "61_80_var1.dat"
SO = 4
N_GATHER = 100


def main():
    # region Model definition
    spacing = (0.075, 0.075)
    velmodel = VelocityModel(PATH_MODEL, dx=spacing[0], dz=spacing[1])

    origin = velmodel.x[0], velmodel.z[0]

    vp = velmodel.vp.T * 1.03
    epsilon = velmodel.epsilon.T
    epsilon = velmodel.epsilon.T
    delta = velmodel.epsilon.T

    # velmodel.plot(show=True)

    nbl = 500

    model = SeismicModel(
        vp=vp,
        origin=origin,
        shape=vp.shape,
        spacing=spacing,
        space_order=SO,
        epsilon=epsilon,
        delta=delta,
        nbl=nbl,
        bcs="damp",
        vti=True,
        fs=True,
    )
    model.smooth(("vp", "epsilon"), sigma=3)
    # endregion
    print(model.padsizes)
    # region Dataset definition
    dataset = SeismogramDataset(PATH_DATA, "rec")
    dataset.dt_r = model.critical_dt
    tn = dataset.t_max
    dataset.t_max_r = tn
    f0 = 1.0
    t0 = 0.0

    geometry = AcquisitionGeometry(
        model, np.array([[origin[0], origin[1]]]).T, np.array([[origin[0], origin[1]]]).T, t0, tn, f0=f0, src_type="Gabor"
    )
    solver = VTIWaveSolver(model, geometry, space_order=SO)
    dataset.resample_on()

    for i in range(100, 105):
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

        geometry = AcquisitionGeometry(model, rec_pos, src_pos, t0, tn, f0=2 * f0, src_type="Gabor")
        solver = VTIWaveSolver(model, geometry, space_order=SO)

        d_1, _, _ = solver.forward(vp=model.vp, save=False)
        wav1 = geometry.src.data

        stf = wiener_deconvolution(d_2.T, d_1.data, eps=1e-8)

        wav3 = np.convolve(wav1.squeeze(), stf)[: d_1.shape[0]]
        wav3_tapered, taper = taper_wavelet(wav3, geometry.time_axis.time_values, 2 / f0, 0.5 * (2 / f0))

        geometry = AcquisitionGeometry(model, rec_pos, src_pos, t0, tn, f0=f0 * 2, src_type=None, wav_data=wav3_tapered)
        solver = VTIWaveSolver(model, geometry, space_order=SO)
        d_3, _, psave, _  = solver.forward(vp=model.vp, src=geometry.src, save=True, nsnaps=100)
        print(psave.shape)
        filename = f"snaps/snaps {i+1}.bin"
        psave.data[:, nbl:-nbl, :-nbl].tofile(filename)
        fig, ax = overlay_wiggle_plot(
            np.array(d_1.data[:]), d_2.T, time_axis=geometry.time_axis.time_values, xrec=rec_z, title="Original vs Processed"
        )
        plt.savefig(f"forward_initial/Forward + Initial {i+1}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        fig, axs = plot_two_wavelets(geometry.time_axis.time_values, wav1, wav3_tapered, taper)
        plt.savefig(f"wavelets/Wavelet {i+1}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        fig, ax = overlay_wiggle_plot(
            np.array(d_3.data[:]), d_2.T, time_axis=geometry.time_axis.time_values, xrec=rec_z, title="Original vs Processed"
        )
        plt.savefig(f"forward_inverted/Forward + Inverted {i+1}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    info(f"Execution time {end - start:.2f} seconds")
