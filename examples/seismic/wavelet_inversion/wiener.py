import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import windows


def apply_normalization(data, eps=1e-6):
    """Normalize each trace by its maximum absolute value."""
    max_vals = np.max(np.abs(data), axis=0, keepdims=True) + eps
    return data / max_vals


def create_taper_window(nt, time_axis, fb_time, before, after):
    """Create a taper window around a first break time."""
    taper = np.ones(nt)
    dt = time_axis[1] - time_axis[0]

    start_time = fb_time - before
    end_time = fb_time + after

    idx_start = np.argmin(np.abs(time_axis - start_time))
    idx_end = np.argmin(np.abs(time_axis - end_time))
    fb_sample = np.argmin(np.abs(time_axis - fb_time))

    # Left taper (before first break)
    if before > 0 and fb_sample > idx_start:
        left_len = fb_sample - idx_start
        left_taper = windows.tukey(2 * left_len, alpha=1.0)[:left_len]
        taper[idx_start:fb_sample] = left_taper

    # Right taper (after first break)
    if idx_end > fb_sample:
        right_len = idx_end - fb_sample
        right_taper = windows.tukey(2 * right_len, alpha=1.0)[right_len:]
        taper[fb_sample:idx_end] = right_taper

    taper[idx_end:] = 0.0
    return taper


def taper_traces_vectorized(data, first_breaks, time_axis, window_before=0, window_after=0.1):
    """Vectorized implementation of trace tapering."""
    nt, ntr = data.shape
    tapered_data = np.zeros_like(data)

    # Calculate first break times
    fb_times = time_axis[first_breaks]

    # Create taper windows for all traces
    tapers = np.array([create_taper_window(nt, time_axis, fb_times[i], window_before, window_after) for i in range(ntr)]).T

    # Apply tapers
    tapered_data = data * tapers
    return tapered_data


def compute_fft_components(obs, modeled, offsets, offset_threshold):
    """Compute FFT components with optional offset filtering."""
    mask = offsets > (np.max(offsets) - offset_threshold)
    print(mask)
    obs_masked = obs[:, mask]
    mod_masked = modeled[:, mask]

    D_obs = fft(obs_masked, axis=0)
    D_mod = fft(mod_masked, axis=0)

    sumn = np.sum(D_obs * np.conj(D_mod), axis=1)
    sumd = np.sum(D_mod * np.conj(D_mod), axis=1)

    return sumn, sumd, np.sum(mask)


def wiener_deconvolution(
    obs,
    modeled,
    sz,
    rec_z,
    time_axis,
    first_breaks=None,
    eps=1e-6,
    normalize=True,
    kill_offset=False,
    offset_threshold=0,
    taper_before=0,
    taper_after=0.1,
):
    """
    Enhanced Wiener deconvolution with increased decomposition and vectorization.

    Parameters:
        obs (np.ndarray): Observed seismograms (shape: [time_samples, traces]).
        modeled (np.ndarray): Synthetic seismograms (shape: [time_samples, traces]).
        sz (float): Source elevation.
        rec_z (np.ndarray): Receiver elevations (shape: [traces]).
        time_axis (np.ndarray): Time axis corresponding to the data.
        first_breaks (np.ndarray): First break samples (shape: [traces]).
        eps (float): Stabilization constant (default: 1e-6).
        normalize (bool): Whether to normalize traces (default: True).
        kill_offset (bool): Whether to exclude near offsets (default: False).
        offset_threshold (float): Minimum offset to include (default: 0).
        taper_before (float): Time before first break to start taper (default: 0).
        taper_after (float): Time after first break for full taper (default: 0.1).

    Returns:
        np.ndarray: Estimated STF (shape: [time_samples]).
    """
    # Calculate offsets
    offsets = np.abs(rec_z - sz)
    mask = offsets > (np.max(offsets) - offset_threshold)
    used_rec_z = rec_z[mask]
    # Apply first-break tapering if provided
    if first_breaks is not None:
        obs = taper_traces_vectorized(obs, first_breaks, time_axis, taper_before, taper_after)
        modeled = taper_traces_vectorized(modeled, first_breaks, time_axis, taper_before, taper_after)

    # Apply normalization if requested
    if normalize:
        obs = apply_normalization(obs, eps)
        modeled = apply_normalization(modeled, eps)

    # Compute FFT components with offset filtering
    sumn, sumd, used_traces = compute_fft_components(obs, modeled, offsets, offset_threshold if kill_offset else -np.inf)

    if used_traces == 0:
        raise ValueError("No traces available after offset filtering")

    # Compute Wiener filter
    Ebar = np.mean(np.abs(sumd))
    H = sumn / (sumd + eps * used_traces * Ebar)

    # Inverse FFT to get STF
    stf = np.real(ifft(H))[: len(time_axis)]

    return stf, obs[:, mask], modeled[:, mask], used_rec_z
