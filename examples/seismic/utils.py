import numpy as np
import pandas as pd
from scipy.signal import windows
from argparse import Action, ArgumentError, ArgumentParser

from devito import error, configuration, warning
from devito.tools import Pickable
from devito.types.sparse import _default_radius

from .source import *

__all__ = ["AcquisitionGeometry", "setup_geometry", "seismic_args"]

import numpy as np
from scipy.fft import fft, ifft


def estimate_centroid_frequency_gather(data, dt, method="median"):
    centroids = []
    ntraces = data.shape[1]  # Assuming data is [samples x traces]

    for trace in data.T:  # Process each trace separately
        n = len(trace)
        fft_data = np.fft.rfft(trace)
        freqs = np.fft.rfftfreq(n, dt / 1e3)

        power_spectrum = np.abs(fft_data) ** 2

        centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        centroids.append(centroid)

    # Use median to be robust against outliers
    if method == "median":
        return np.median(centroids)
    else:
        return np.mean(centroids)


def wiener_deconvolution(
    obs,
    modeled,
    sz,
    rec_z,
    eps=1e-6,
    normalize=True,
    kill_offset=False,
    offset_threshold=0,
):
    """
    Wiener deconvolution to estimate the source time function (STF) with optional offset filtering.

    Parameters:
        obs (np.ndarray): Observed seismograms (shape: [time_samples, traces]).
        modeled (np.ndarray): Synthetic seismograms (shape: [time_samples, traces]).
        sz (float): Source elevation.
        rec_z (np.ndarray): Receiver elevations (shape: [traces]).
        eps (float): Stabilization constant (default: 1e-6).
        normalize (bool): Whether to normalize traces by their maximum amplitude (default: True).
        kill_offset (bool): Whether to exclude near offsets (default: False).
        offset_threshold (float): Minimum absolute offset to include (used if kill_offset=True).

    Returns:
        np.ndarray: Estimated STF (shape: [time_samples]).
    """
    nt, ntr = obs.shape
    nfft = nt  # Next power of 2 for FFT

    # Calculate offsets for each trace
    offsets = np.abs(rec_z - sz)

    # Initialize numerator and denominator
    sumn = np.zeros(nfft, dtype=complex)
    sumd = np.zeros(nfft, dtype=complex)

    # Count of traces actually used
    used_traces = 0

    # Compute FFT of each trace and accumulate sums
    for i in range(ntr):
        # Skip near offsets if kill_offset is True
        if kill_offset and offsets[i] <= offset_threshold:
            continue

        used_traces += 1

        # Normalize if requested
        if normalize:
            obs_norm = obs[:, i] / (np.max(np.abs(obs[:, i])) + eps)
            mod_norm = modeled[:, i] / (np.max(np.abs(modeled[:, i])) + eps)
        else:
            obs_norm = obs[:, i]
            mod_norm = modeled[:, i]

        D_obs = fft(obs_norm)
        D_mod = fft(mod_norm)

        sumn += D_obs * np.conj(D_mod)  # Cross-correlation
        sumd += D_mod * np.conj(D_mod)  # Auto-correlation

    if used_traces == 0:
        raise ValueError("No traces available after offset filtering - check your offset threshold")

    # Stabilization term (Ebar = average energy)
    Ebar = np.mean(np.abs(sumd))

    # Wiener filter in frequency domain
    H = sumn / (sumd + eps * used_traces * Ebar)

    # Inverse FFT to get STF (truncate to original length)
    stf = np.real(ifft(H))[:nt]

    return stf


def taper_wavelet(wav3, t, t_high, t_width):
    """
    Taper a wavelet using the right half of a Tukey window.

    Parameters:
    -----------
    wav3 : numpy array
        The wavelet to be tapered
    t : numpy array
        Time axis corresponding to the wavelet
    t_high : float
        Time point until which the taper is 1
    t_width : float
        Length of the taper transition region

    Returns:
    --------
    tapered_wav3 : numpy array
        The tapered wavelet
    taper_window : numpy array
        The applied taper window
    """
    # Create the taper window
    taper_window = np.ones_like(wav3)

    # Find indices for the transition region
    idx_high = np.argmin(np.abs(t - t_high))
    idx_end = np.argmin(np.abs(t - (t_high + t_width)))

    # Create a Tukey window for the transition region
    # We'll use the right half of the window (alpha=1 gives a Hann window shape)
    tukey_len = 2 * (idx_end - idx_high)
    tukey_window = windows.tukey(tukey_len, alpha=1.0)

    # Take the right half of the Tukey window for our taper
    right_half = tukey_window[tukey_len // 2 :]

    # Apply the taper
    taper_window[idx_high:idx_end] = right_half[: idx_end - idx_high]
    taper_window[idx_end:] = 0.0

    # Apply the taper to the wavelet
    tapered_wav3 = wav3 * taper_window

    return tapered_wav3, taper_window


def setup_geometry(model, tn, f0=0.010, interpolation="linear", **kwargs):
    # Source and receiver geometries
    src_coordinates = np.empty((1, model.dim))
    if model.dim > 1:
        src_coordinates[0, :] = np.array(model.domain_size) * 0.5
        # src_coordinates[0, -1] = model.origin[-1] + model.spacing[-1]
    else:
        src_coordinates[0, 0] = 2 * model.spacing[0]

    rec_coordinates = setup_rec_coords(model)

    r = kwargs.get("r", _default_radius[interpolation])
    geometry = AcquisitionGeometry(
        model,
        rec_coordinates,
        src_coordinates,
        t0=0.0,
        tn=tn,
        src_type="Ricker",
        f0=f0,
        interpolation=interpolation,
        r=r,
    )

    return geometry


def setup_rec_coords(model):
    nrecx = model.shape[0]
    recx = np.linspace(model.origin[0], model.domain_size[0], nrecx)

    if model.dim == 1:
        return recx.reshape((nrecx, 1))
    elif model.dim == 2:
        rec_coordinates = np.empty((nrecx, model.dim))
        rec_coordinates[:, 0] = recx
        rec_coordinates[:, -1] = model.origin[-1] + 2 * model.spacing[-1]
        return rec_coordinates
    else:
        nrecy = model.shape[1]
        recy = np.linspace(model.origin[1], model.domain_size[1], nrecy)
        rec_coordinates = np.empty((nrecx * nrecy, model.dim))
        rec_coordinates[:, 0] = np.array([recx[i] for i in range(nrecx) for j in range(nrecy)])
        rec_coordinates[:, 1] = np.array([recy[j] for i in range(nrecx) for j in range(nrecy)])
        rec_coordinates[:, -1] = model.origin[-1] + 2 * model.spacing[-1]
        return rec_coordinates


class AcquisitionGeometry(Pickable):
    """
    Encapsulate the geometry of an acquisition:
    - source positions and number
    - receiver positions and number

    In practice this would only point to a segy file with the
    necessary information
    """

    __rargs__ = ("grid", "rec_positions", "src_positions", "t0", "tn")
    __rkwargs__ = ("f0", "src_type", "interpolation", "r")

    def __init__(self, model, rec_positions, src_positions, t0, tn, **kwargs):
        """
        In practice would be __init__(segyfile) and all below parameters
        would come from a segy_read (at property call rather than at init)
        """
        src_positions = np.reshape(src_positions, (-1, model.dim))
        rec_positions = np.reshape(rec_positions, (-1, model.dim))
        self.rec_positions = rec_positions
        self._nrec = rec_positions.shape[0]
        self.src_positions = src_positions
        self._nsrc = src_positions.shape[0]
        self._src_type = kwargs.get("src_type")
        assert self.src_type in sources or self.src_type is None
        self._f0 = kwargs.get("f0")
        self._a = kwargs.get("a", None)
        self._t0w = kwargs.get("t0w", None)
        if self._src_type is not None and self._f0 is None:
            error("Peak frequency must be provided in KHz" + " for source of type %s" % self._src_type)

        self._grid = model.grid
        self._model = model
        self._dt = model.critical_dt
        self._t0 = t0
        self._tn = tn
        self._interpolation = kwargs.get("interpolation", "linear")
        self._r = kwargs.get("r", _default_radius[self.interpolation])

        # Initialize to empty, created at new src/rec
        self._src_coordinates = None
        self._rec_coordinates = None
        self.wav_data = kwargs.get("wav_data", None)

    def resample(self, dt):
        self._dt = dt
        return self

    @property
    def time_axis(self):
        return TimeAxis(start=self.t0, stop=self.tn, step=self.dt)

    @property
    def src_type(self):
        return self._src_type

    @property
    def grid(self):
        return self._grid

    @property
    def model(self):
        warning("Model is kept for backward compatibility but should not be" "obtained from the geometry")
        return self._model

    @property
    def f0(self):
        return self._f0

    @property
    def tn(self):
        return self._tn

    @property
    def t0(self):
        return self._t0

    @property
    def dt(self):
        return self._dt

    @property
    def nt(self):
        return self.time_axis.num

    @property
    def nrec(self):
        return self._nrec

    @property
    def nsrc(self):
        return self._nsrc

    @property
    def dtype(self):
        return self.grid.dtype

    @property
    def r(self):
        return self._r

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def rec(self):
        return self.new_rec()

    def new_rec(self, name="rec", coordinates=None):
        coords = coordinates or self.rec_positions
        rec = Receiver(
            name=name,
            grid=self.grid,
            time_range=self.time_axis,
            npoint=self.nrec,
            interpolation=self.interpolation,
            r=self._r,
            coordinates=coords,
        )

        return rec

    @property
    def adj_src(self):
        if self.src_type is None:
            return self.new_rec()
        coords = self.rec_positions
        adj_src = sources[self.src_type](
            name="rec",
            grid=self.grid,
            f0=self.f0,
            time_range=self.time_axis,
            npoint=self.nrec,
            interpolation=self.interpolation,
            r=self._r,
            coordinates=coords,
            t0=self._t0w,
            a=self._a,
        )
        # Revert time axis to have a proper shot record and not compute on zeros
        for i in range(self.nrec):
            adj_src.data[:, i] = adj_src.wavelet[::-1]
        return adj_src

    @property
    def src(self):
        return self.new_src()

    def new_src(self, name="src", src_type="self", coordinates=None):
        coords = coordinates or self.src_positions
        if self.src_type is None or src_type is None:
            # warning("No source type defined, returning uninitiallized (zero) source")
            # Prepare common arguments
            src_args = {
                "name": name,
                "grid": self.grid,
                "time_range": self.time_axis,
                "npoint": self.nsrc,
                "coordinates": coords,
                "interpolation": self.interpolation,
                "r": self._r
            }
            
            # Add wav_data if it exists
            if self.wav_data is not None:
                # warning("Initializing source with custom data")
                src_args["data"] = self.wav_data.reshape(-1, 1)
            else:
                warning("No source type defined, returning uninitialized (zero) source")
            
            src = PointSource(**src_args)
        else:
            src = sources[self.src_type](
                name=name,
                grid=self.grid,
                f0=self.f0,
                time_range=self.time_axis,
                npoint=self.nsrc,
                coordinates=coords,
                t0=self._t0w,
                a=self._a,
                interpolation=self.interpolation,
                r=self._r,
            )

        return src


sources = {"Wavelet": WaveletSource, "Ricker": RickerSource, "Gabor": GaborSource}


def seismic_args(description):
    """
    Command line options for the seismic examples
    """

    class _dtype_store(Action):
        def __call__(self, parser, args, values, option_string=None):
            values = {"float32": np.float32, "float64": np.float64}[values]
            setattr(args, self.dest, values)

    class _opt_action(Action):
        def __call__(self, parser, args, values, option_string=None):
            try:
                # E.g., `('advanced', {'par-tile': True})`
                values = eval(values)
                if not isinstance(values, tuple) and len(values) >= 1:
                    raise ArgumentError(
                        self,
                        ("Invalid choice `%s` (`opt` must be " "either str or tuple)" % str(values)),
                    )
                opt = values[0]
            except NameError:
                # E.g. `'advanced'`
                opt = values
            if opt not in configuration._accepted["opt"]:
                raise ArgumentError(
                    self,
                    ("Invalid choice `%s` (choose from %s)" % (opt, str(configuration._accepted["opt"]))),
                )
            setattr(args, self.dest, values)

    parser = ArgumentParser(description=description)
    parser.add_argument("-nd", dest="ndim", default=3, type=int, help="Number of dimensions")
    parser.add_argument(
        "-d",
        "--shape",
        default=(51, 51, 51),
        type=int,
        nargs="+",
        help="Number of grid points along each axis",
    )
    parser.add_argument(
        "-f",
        "--full",
        default=False,
        action="store_true",
        help="Execute all operators and store forward wavefield",
    )
    parser.add_argument(
        "-so",
        "--space_order",
        default=4,
        type=int,
        help="Space order of the simulation",
    )
    parser.add_argument(
        "--nbl",
        default=40,
        type=int,
        help="Number of boundary layers around the domain",
    )
    parser.add_argument(
        "--constant",
        default=False,
        action="store_true",
        help="Constant velocity model, default is a two layer model",
    )
    parser.add_argument(
        "--checkpointing",
        default=False,
        action="store_true",
        help="Use checkpointing, default is False",
    )
    parser.add_argument(
        "-opt",
        default="advanced",
        action=_opt_action,
        help="Performance optimization level",
    )
    parser.add_argument(
        "-a",
        "--autotune",
        default="off",
        choices=(configuration._accepted["autotuning"]),
        help="Operator auto-tuning mode",
    )
    parser.add_argument("-tn", "--tn", default=0, type=float, help="Simulation time in millisecond")
    parser.add_argument(
        "-dtype",
        action=_dtype_store,
        dest="dtype",
        default=np.float32,
        choices=["float32", "float64"],
    )
    parser.add_argument("-interp", dest="interp", default="linear", choices=["linear", "sinc"])
    return parser
