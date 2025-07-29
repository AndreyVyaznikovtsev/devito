import time
import numpy as np
from examples.seismic import SeismicModel, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver, EikonalSolver
from examples.seismic.datasets import SeismogramDataset, VelocityModel
from devito import info

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
slices = (slice(None), slice(slic, -slic, SUBSAMPLING), slice(None, -slic, SUBSAMPLING)) if FS else (slice(None), slice(slic, -slic, SUBSAMPLING), slice(slic, -slic, SUBSAMPLING))

def main():
    dataset = SeismogramDataset(PATH_DATA, "sou", invert_elevs=True)
    xmin, xmax = min(dataset.x_coords.min(), dataset.opposite_x.min()), max(dataset.x_coords.max(), dataset.opposite_x.max())
    spacing = (0.025, 0.025)
    velmodel = VelocityModel(PATH_MODEL, dx=spacing[0], dz=spacing[1], clip=True, xmin=xmin-3, xmax=xmax+3, zmin=-318)
    velmodel.pad_left(4+2)
    velmodel.pad_right(8*int(0.5/spacing[0])+2)
    velmodel.pad_bottom(10*int(0.5/spacing[0])+2)
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
    f0 = 0.25

    for i in range(len(dataset)):
        info(f"Inverting {i+1}-th gather wavelet")
        _, sx, sz, rec_x, rec_z = dataset[i]

        src_pos = np.array([sx, sz])[None, :]
        rec_pos = np.vstack([rec_x, rec_z]).T
        f0 = 0.3
        wav_data=np.load(f"conventional_wavelets/{i+1}.npy")
        geometry = AcquisitionGeometry(model, rec_pos, src_pos, t0, tn, f0=f0*2, src_type=None, wav_data=wav_data)
        solver = AcousticWaveSolver(model, geometry, space_order=SO)

        _, _, psave, _  = solver.forward(vp=model.vp, src=geometry.src, save=True, nsnaps=500)
        print(psave.shape)
        filename = f"forward_snaps/{i+1}.bin"
        # print(psave.data[slices].shape)
        psave.data[:].tofile(filename)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    info(f"Execution time {end - start:.2f} seconds")