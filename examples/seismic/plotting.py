import numpy as np
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.rc('font', size=16)
    mpl.rc('figure', figsize=(8, 6))
except:
    plt = None
    cm = None


def plot_three_wavelets(time_axis, wavelet1, wavelet2, wavelet3, 
                        labels=["Initial STF", "True STF", "Deconvolved STF"],
                        title="Source Time Function Comparison",
                        normalize=True, noise=0.0):
    """
    Plot three wavelets on the same axes for comparison.
    
    Parameters:
        time_axis (np.ndarray): Time axis (e.g., `np.linspace(0, 1, nt)`).
        wavelet1 (np.ndarray): First wavelet (e.g., true STF).
        wavelet2 (np.ndarray): Second wavelet (e.g., initial guess).
        wavelet3 (np.ndarray): Third wavelet (e.g., Wiener-deconvolved STF).
        labels (list): Labels for each wavelet (length 3).
        title (str): Plot title.
        normalize (bool): If True, normalizes all wavelets to peak amplitude.
    """
    if normalize:
        wavelet1 = wavelet1 / np.max(np.abs(wavelet1))
        wavelet2 = wavelet2 / np.max(np.abs(wavelet2))
        wavelet3 = wavelet3 / np.max(np.abs(wavelet3))

    fig = plt.figure(figsize=(10, 5))
    plt.plot(time_axis, wavelet1, 'k-', linewidth=2, label=labels[0])
    plt.plot(time_axis, wavelet2, 'r-', linewidth=1.5, label=labels[1])
    plt.plot(time_axis, wavelet3, 'b:', linewidth=1.5, label=labels[2])
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (normalized)" if normalize else "Amplitude")
    plt.title(f"{title}, N = {noise*100:.0f} %")
    plt.legend()
    plt.xlim([0, time_axis[len(time_axis)//3]])
    plt.ylim([-1, 1])
    plt.grid(True, linestyle=':')
    # plt.show()
    return fig, plt.gca()

def plot_seis_double_hor(seis1, seis2, time_range, z, titles, show=False):
    t = np.linspace(time_range[0], time_range[-1], seis1.shape[0])
    z = np.linspace(z[-1], z[0], seis1.shape[1])
    zz, tt = np.meshgrid(z, t)
    fig, axs = plt.subplots(1, 2, figsize=(6.375*2, 4), sharey=True, dpi=300)
    vm = np.quantile(seis1, 0.99)
    axs[0].pcolormesh(zz, tt, seis1, vmin=-vm, vmax=vm, cmap="gray")
    axs[1].pcolormesh(zz, tt, seis2, vmin=-vm, vmax=vm, cmap="gray")

    axs[0].invert_yaxis()
    axs[0].set_ylabel("Время, мс", fontsize=16)
    for i, ax in enumerate(axs):
        ax.set_title(titles[i], fontsize=16)
        ax.set_xlabel("Глубина ПП, м", fontsize=16)

    fig.tight_layout()

    if show:
        plt.show()
    else:
        return fig, axs

def plot_perturbation(model, model1, colorbar=True):
    """
    Plot a two-dimensional velocity perturbation from two seismic `Model`
    objects.

    Parameters
    ----------
    model : Model
        The first velocity model.
    model1 : Model
        The second velocity model.
    colorbar : bool
        Option to plot the colorbar.
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]
    dv = np.transpose(model.vp.data) - np.transpose(model1.vp.data)

    plot = plt.imshow(dv, animated=True, cmap=cm.jet,
                      vmin=min(dv.reshape(-1)), vmax=max(dv.reshape(-1)),
                      extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity perturbation (km/s)')
    plt.show()


def plot_velocity(model, source=None, receiver=None, colorbar=True, cmap="jet"):
    """
    Plot a two-dimensional velocity field from a seismic `Model`
    object. Optionally also includes point markers for sources and receivers.

    Parameters
    ----------
    model : Model
        Object that holds the velocity model.
    source : array_like or float
        Coordinates of the source point.
    receiver : array_like or float
        Coordinates of the receiver points.
    colorbar : bool
        Option to plot the colorbar.
    """
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    if getattr(model, 'vp', None) is not None:
        field = model.vp.data[slices]
    else:
        field = model.lam.data[slices]
    plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                      vmin=np.min(field), vmax=np.max(field),
                      extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(1e-3*receiver[:, 0], 1e-3*receiver[:, 1],
                    s=25, c='green', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e-3*source[:, 0], 1e-3*source[:, 1],
                    s=25, c='red', marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity (km/s)')
    plt.show()


def plot_shotrecord(rec, model, t0, tn, colorbar=True):
    """
    Plot a shot record (receiver values over time).

    Parameters
    ----------
    rec :
        Receiver data with shape (time, points).
    model : Model
        object that holds the velocity model.
    t0 : int
        Start of time dimension to plot.
    tn : int
        End of time dimension to plot.
    """
    scale = np.max(rec) / 10.
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0],
              1e-3*tn, t0]

    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray, extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()


def plot_image(data, vmin=None, vmax=None, colorbar=True, cmap="gray"):
    """
    Plot image data, such as RTM images or FWI gradients.

    Parameters
    ----------
    data : ndarray
        Image data to plot.
    cmap : str
        Choice of colormap. Defaults to gray scale for images as a
        seismic convention.
    """
    plot = plt.imshow(np.transpose(data),
                      vmin=vmin or 0.9 * np.min(data),
                      vmax=vmax or 1.1 * np.max(data),
                      cmap=cmap)

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.show()
