#!/usr/bin/env python3
"""
Usage:
    python plot_seismogram.py rec.npy [rec2.npy ...]

Each .npy file is a (nt, n_rec) float32 seismogram.
Receiver x-axis is inferred from the array width unless --rec-x is given.
"""

import sys
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def plot_seismogram(data: np.ndarray, title: str,
                    rec_x_start: float = 50., rec_x_step: float = 2.,
                    t0: float = 0., tn: float = 850.) -> None:
    n_rec = data.shape[1]
    rec_x_end = rec_x_start + (n_rec - 1) * rec_x_step

    clip = np.percentile(np.abs(data), 85) or 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        data,
        aspect='auto',
        cmap='gray',
        vmin=-clip, vmax=clip,
        extent=[rec_x_start, rec_x_end, tn, t0],
        interpolation='bicubic',
    )
    ax.set_xlabel('Receiver x [m]')
    ax.set_ylabel('Time [ms]')
    ax.set_title(title)
    plt.tight_layout()


default_paths = ['/tmp/rec_cpu.npy', '/tmp/rec_cuda.npy', '/tmp/rec_diff.npy']
paths = sys.argv[1:] if len(sys.argv) > 1 else [p for p in default_paths
                                                  if Path(p).exists()]

for p in paths:
    data = np.load(p)
    plot_seismogram(data, title=Path(p).stem)
    print(f'{Path(p).stem}: max={np.abs(data).max():.4e}  p85={np.percentile(np.abs(data), 85):.4e}')

plt.show()

