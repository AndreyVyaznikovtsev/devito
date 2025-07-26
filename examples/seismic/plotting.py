import numpy as np
from typing import Sequence
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.rc('font', size=10)
    mpl.rc('figure', figsize=(8, 6))
except:
    plt = None
    cm = None

def plot_two_wavelets(time_axis, wavelet1, wavelet2, taper,
                        labels=["Initial STF", "Deconvolved STF", "Taper"],
                        title="Source Time Function Comparison",
                        normalize=True):
    """
    Plot three wavelets on the same axes for comparison.
    
    Parameters:
        time_axis (np.ndarray): Time axis (e.g., `np.linspace(0, 1, nt)`).
        wavelet1 (np.ndarray): Second wavelet (e.g., initial guess).
        wavelet2 (np.ndarray): Third wavelet (e.g., Wiener-deconvolved STF).
        labels (list): Labels for each wavelet (length 3).
        title (str): Plot title.
        normalize (bool): If True, normalizes all wavelets to peak amplitude.
    """
    if normalize:
        wavelet1 = wavelet1 / np.max(np.abs(wavelet1))
        wavelet2 = wavelet2 / np.max(np.abs(wavelet2))

    fig = plt.figure(figsize=(10, 5))
    plt.plot(time_axis, wavelet1, 'k-', linewidth=2, label=labels[0])
    plt.plot(time_axis, wavelet2, 'r-', linewidth=1.5, label=labels[1])
    plt.plot(time_axis, taper, 'b-', linewidth=1.5, label=labels[2])
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (normalized)" if normalize else "Amplitude")
    plt.title(f"{title}")
    plt.legend()
    plt.xlim([0, time_axis[len(time_axis)//3]])
    plt.ylim([-1, 1])
    plt.grid(True, linestyle=':')
    # plt.show()
    return fig, plt.gca()

def overlay_wiggle_plot(
    data1: np.ndarray,
    data2: np.ndarray,
    xrec: Sequence[float] | None = None,
    time_axis: Sequence[float] | None = None,
    *,
    t_scale: float = 1.0,
    gain: float = 1.5,
    title: str = "Seismic Comparison",
    figsize: tuple = (8, 6),
    dpi: int = 300,
    show: bool = False,
):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Set time axis if not provided
    if time_axis is None:
        time_axis = np.arange(data1.shape[0])
    if xrec is None:
        xrec = np.arange(data1.shape[1])
        
    xrec = np.asarray(xrec)
    time_axis = np.asarray(time_axis)
    tg = time_axis ** t_scale
    dx = np.diff(xrec, prepend=xrec[0])
    
    # Set axis limits
    ax.set_ylim(time_axis.max(), time_axis.min())
    ax.set_xlim(xrec.min(), xrec.max())
    model_line = plt.Line2D([0], [0], color='k', linewidth=0.8, label='Модельные данные')
    real_line = plt.Line2D([0], [0], color='r', linewidth=0.6, label='Наблюденные данные')

    # Plot first dataset (black)
    for i, xr in enumerate(xrec):
        trace = tg * data1[:, i]
        if np.max(np.abs(trace)) != 0:
            trace = gain * (dx[i] * trace / np.max(np.abs(trace))) + xr
        else:
            trace = trace + xr
        
        ax.plot(trace, time_axis, 'k-', linewidth=0.8, alpha=0.9)
        ax.fill_betweenx(time_axis, xr, trace, where=trace > xr, color='k', alpha=0.5)
    
    # Plot second dataset (red)
    for i, xr in enumerate(xrec):
        trace = tg * data2[:, i]
        if np.max(np.abs(trace)) != 0:
            trace = gain * (dx[i] * trace / np.max(np.abs(trace))) + xr
        else:
            trace = trace + xr
        
        ax.plot(trace, time_axis, 'r-', linewidth=0.6, alpha=0.7)
        ax.fill_betweenx(time_axis, xr, trace, where=trace > xr, color='r', alpha=0.4)
    ax.legend(handles=[model_line, real_line], loc='upper right', framealpha=0.9, fontsize=8)
    # Add labels and title
    ax.set_xlabel("Глубина, м", fontsize=12)
    ax.set_ylabel("Время, мс", fontsize=12)
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        return fig, ax

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
    fig, axs = plt.subplots(1, 2, figsize=(6.375*2, 4), sharey=True, dpi=300)
    vm1 = np.quantile(seis1, 0.99)
    vm2 = np.quantile(seis2, 0.99)

    
    # Calculate extent [left, right, bottom, top]
    extent = [z[0], z[-1], time_range[-1], time_range[0]]
    
    # Plot using imshow
    axs[0].imshow(seis1, aspect='auto', extent=extent, 
                 vmin=-vm1, vmax=vm1, cmap="gray")
    axs[1].imshow(seis2, aspect='auto', extent=extent, 
                 vmin=-vm2, vmax=vm2, cmap="gray")

    # axs[0].invert_yaxis()
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
    domain_size = np.array(model.domain_size)
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
        plt.scatter(receiver[:, 0], receiver[:, 1],
                    s=25, c='green', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(source[:, 0], source[:, 1],
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


def plot_shotrecord(rec, t0, tn, colorbar=True):
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
    scale = np.max(rec) / 5.
    # extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0],
            #   model.origin[1] + 1e-3*model.domain_size[1], model.origin[1]]
    print(rec.shape)
    plot = plt.imshow(rec.T, vmin=-scale, vmax=scale, cmap=cm.gray, aspect='equal')
    plt.xlabel('X position (km)')
    plt.ylabel('Y position (km)')
    # major_ticks = np.arange(0, rec.shape[0], 500)
    ax = plt.gca()
    # ax.set_xticks(major_ticks)
    # ax.set_yticks(major_ticks)
    plt.grid()

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
