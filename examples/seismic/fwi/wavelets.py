import numpy as np
from scipy.interpolate import interp1d


def interpolate_wavelet(wav_data_source, source_dt, target_dt, target_time=None):
    wav_time = np.arange(0, wav_data_source.size * source_dt, source_dt)
    if target_time:
        new_time = np.arange(0, target_time, target_dt)
    else:
        new_time = np.arange(0, wav_data_source.size * source_dt, target_dt)

    interp_func = interp1d(wav_time, wav_data_source, kind="linear", bounds_error=False, fill_value=0.0)
    return interp_func(new_time)
