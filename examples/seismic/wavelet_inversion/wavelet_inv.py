

import argparse
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt
from examples.seismic.utils import load_velocity, wiener_deconvolution
from examples.seismic import SeismicModel, AcquisitionGeometry, TimeAxis
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.vti import VTIWaveSolver
from examples.seismic.plotting import plot_seis_double_hor, plot_three_wavelets
from devito import configuration, TimeFunction, Function, gaussian_smooth
from scipy import ndimage
import gc
from scipy.signal.windows import tukey as tukey_w
from PIL import Image  # For creating GIFs
import os

def create_gif(image_folder, output_path, duration=500):
    """Create GIF from a folder of images"""
    images = []
    # Get all PNG files and sort them numerically
    files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    files.sort()
    print(files)
    # files.sort(key=lambda x: float(x.split()[-1].replace('.png', '')))
    
    for file in files:
        file_path = os.path.join(image_folder, file)
        images.append(Image.open(file_path))
    
    # Save as GIF
    images[0].save(output_path,
                  save_all=True,
                  append_images=images[1:],
                  duration=duration,
                  loop=0)


def main():
    pad_x = 5
    vp, vxvz, nz, nx, z, x, dz, dx = load_velocity(path='88_99_EXTENDED.txt', pad_x=pad_x)
    so = 4
    spacing = (dx, dz)
    vp = vp.T
    vxvz = vxvz.T
    epsilon = vxvz - 1
    epsilon[epsilon < 0] = 0.
    delta = np.zeros_like(epsilon)
    vp = (2*vp)/(2+epsilon)
    print(epsilon.min(), epsilon.max())
    print(vp.min(), vp.max())
    print(dx)
    nbl = 50
    print(vp.shape)
    model = SeismicModel(vp=vp/1000,
                  origin=(x.min(), z.min()),
                  shape=vp.shape,
                  spacing=spacing,
                  space_order=so,
                  epsilon=epsilon,
                  delta=delta,
                  nbl=nbl,
                  bcs="damp",
                  vti=True)
    t0 = 0.0
    tn = 150.0
    f0 = 0.1
    indent = 0.0
    sz = -150.0
    sx = 0.
    src_pos = np.array([sx, sz])[None, :]
    rec_x = np.ones_like(z)*(x[-1]-pad_x*dx)
    rec_z = np.copy(z)
    rec_pos = np.vstack([rec_x, rec_z]).T
    Ns = [0.05*i for i in range(21)]

    geometry = AcquisitionGeometry(model, rec_pos, src_pos, t0, tn, f0=f0, src_type='Ricker')
    solver = VTIWaveSolver(model, geometry, space_order=so)
    d_1, _, _ = solver.forward(vp=model.vp)
    wav1 = geometry.src.data
    geometry = AcquisitionGeometry( 
        model, rec_pos, src_pos, t0, tn, f0=f0*2, src_type='Gabor')
    d_2, _, _ = solver.forward(vp=model.vp, src=geometry.src)
    wav2 = geometry.src.data
    titles = ("(а)", "(б)")
    fig, axs = plot_seis_double_hor(np.array(d_1.data[:]),np.array(d_2.data[:]),
                                    geometry.time_axis.time_values, z, titles=titles,
                                    )   
    
    plt.savefig("Forward" + ".png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()

    for N in Ns:

        var_s = np.var(d_2.data)
        var_n = N*var_s
        std_n = np.sqrt(var_n)
        stf = wiener_deconvolution(d_2.data+np.random.normal(scale=std_n, size=d_2.data.shape), d_1.data)
        wav3 = np.convolve(wav1.squeeze(), stf)[:d_1.shape[0]]
        fig, axs = plot_three_wavelets(geometry.time_axis.time_values, wav1, wav2, wav3, noise=N)
        plt.savefig(f"wavelet_frames/Inverted Wavelet {N:.2f}" + ".png", dpi=300,
                    bbox_inches="tight", pad_inches=0.1)
        plt.close()

        geometry = AcquisitionGeometry( 
            model, rec_pos, src_pos, t0, tn, f0=f0*2, src_type=None, wav_data=wav3)
        d_3, _, _ = solver.forward(vp=model.vp, src=geometry.src)
        
        titles = ("(а)", "(б)")
        fig, axs = plot_seis_double_hor(np.array(d_1.data[:]),np.array(d_3.data[:]),
                                        geometry.time_axis.time_values, z, titles=titles,
                                        )   
        plt.savefig(f"forward_frames/Forward + Inverted {N:.2f}" + ".png", dpi=300,
                    bbox_inches="tight", pad_inches=0.1)
        plt.close()


    create_gif('wavelet_frames', 'wavelets.gif')
    create_gif('forward_frames', 'forward_inverted.gif')
    # image = Function(name='image', grid=model.grid)
    # v = TimeFunction(name='v', grid=model.grid, time_order=2,
    #                  space_order=4, save=geometry.nt)
    # op_imaging = ImagingOperator(model, geometry, v)
    # op_imaging(vp=model0.vp, dt=model0.critical_dt, residual=residual)

    # fig, _ = plot_tti_snapshots(
    #     model, np.array(v.data), xx, zz, src_pos, geometry.time_axis, show=False)
    # plt.savefig(args.output + ".png", dpi=300,
    #             bbox_inches="tight", pad_inches=0.1)
    # plt.close()

    # s, _ = resample(u0.data[:, nbl:-nbl, nbl:-nbl],
    #                 geometry.t0, geometry.tn, geometry.dt, dt=0.5)
    # r, _ = resample(v.data[:, nbl:-nbl, nbl:-nbl],
    #                 geometry.t0, geometry.tn, geometry.dt, dt=0.5)
    # del u0, v
    # sd = fk_filtration(s, kz_filter=lambda x: x)
    # rd = fk_filtration(r, kz_filter=lambda x: x)
    # su = fk_filtration(s, kz_filter=lambda x: 1-x)
    # ru = fk_filtration(r, kz_filter=lambda x: 1-x)

    # fig, _ = plot_filtered_snapshots(
    #     model, sd, su, -rd, -ru, xx, zz, src_pos, TimeAxis(start=t0, stop=tn, step=0.5), show=False)
    # plt.savefig(args.output + ".png", dpi=300,
    #             bbox_inches="tight", pad_inches=0.1)
    # plt.close()

    # image = np.sum(sd*ru, axis=0) + np.sum(su*rd, axis=0)
    # sobel_h = ndimage.sobel(vp, 0)  # horizontal gradient
    # sobel_v = ndimage.sobel(vp, 1)  # vertical gradient
    # edges = np.sqrt(sobel_h**2 + sobel_v**2).astype(bool)
    # edges = np.ma.masked_where(edges == 0, edges)

    # fig, _ = plot_image(np.diff(image), xx, zz, edges=None, src=src_pos)
    # plt.savefig(args.output + ".png", dpi=300,
    #             bbox_inches="tight", pad_inches=0.1)
    # plt.close()

    # images = []
    # src_poss = []
    # for sz in [75., 150., 225.]:
    #     src_pos = np.array([args.indent+xx[0, 0], sz])[None, :]
    #     rec_pos = np.vstack([xx[-1, :]-args.indent, zz[-1, :]]).T
    #     geometry = AcquisitionGeometry(
    #         model, rec_pos, src_pos, t0, tn, f0=f0, src_type='Ricker')
    #     solver = AcousticWaveSolver(model, geometry, space_order=4)
    #     sharp_d, _, _, _ = solver.forward(vp=model.vp, fb=False)
    #     smooth_d, u0, _, _ = solver.forward(vp=model0.vp, save=True, fb=False)
    #     residual = smooth_d.data - sharp_d.data
    #     image = Function(name='image', grid=model.grid)
    #     v = TimeFunction(name='v', grid=model.grid, time_order=2,
    #                     space_order=4, save=geometry.nt)
    #     op_imaging = ImagingOperator(model, geometry, v)
    #     op_imaging(vp=model0.vp, dt=model0.critical_dt, residual=residual)
    #     s, _ = resample(u0.data[:, nbl:-nbl, nbl:-nbl],
    #                     geometry.t0, geometry.tn, geometry.dt, dt=0.5)
    #     r, _ = resample(v.data[:, nbl:-nbl, nbl:-nbl],
    #                     geometry.t0, geometry.tn, geometry.dt, dt=0.5)
    #     del u0, v
    #     sd = fk_filtration(s, kz_filter=lambda x: x)
    #     rd = fk_filtration(r, kz_filter=lambda x: x)
    #     su = fk_filtration(s, kz_filter=lambda x: 1-x)
    #     ru = fk_filtration(r, kz_filter=lambda x: 1-x)
    #     image = np.sum(sd*ru, axis=0) + np.sum(su*rd, axis=0)
    #     image = np.diff(image)
    #     images.append(image)
    #     src_poss.append(src_pos)
    #     gc.collect()
        

    # sobel_h = ndimage.sobel(vp, 0)  # horizontal gradient
    # sobel_v = ndimage.sobel(vp, 1)  # vertical gradient
    # edges = np.sqrt(sobel_h**2 + sobel_v**2).astype(bool)
    # edges = np.ma.masked_where(edges == 0, edges)
    # fig, _ = plot_image_tripple(images, xx, zz, edges=edges, src=src_poss)
    # plt.savefig(args.output + ".png", dpi=300,
    #             bbox_inches="tight", pad_inches=0.1)
    # plt.close()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Execution time {end - start:.2f} seconds")