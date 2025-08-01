# image_update.py - Image updating for a single LSRTM iteration

import numpy as np
import time
from config import *
import os
from scipy.interpolate import interpn
import argparse
from wavefield_computation import setup_model_and_geometry

def get_model_shape():
    return 2630, 3640

def get_num_shots():
    return 59

def load_gradient(iter_num):
    """Load gradient from binary file for specific iteration"""
    path = f"{OUTPUT_DIRS['gradients']}/grad_full_{iter_num}.npy"
    nx, nz = get_model_shape()
    sub_nx = nx // SUBSAMPLING + 1
    sub_nz = nz // SUBSAMPLING + 1

    buff = np.load(path)
    return upsample_image(buff, *get_model_shape()) # upsample on loading

def load_image(iter_num):
    """Load image from previous iteration"""
    path = f"{OUTPUT_DIRS['images']}/image_iter_{iter_num}.npy"
    return np.load(path)

def upsample_image(image_subsampled, nx_orig, nz_orig):
    """Upsample image using bilinear interpolation"""
    sub_nx, sub_nz = image_subsampled.shape
    
    x_coarse = np.linspace(0, nx_orig - 1, sub_nx)
    z_coarse = np.linspace(0, nz_orig - 1, sub_nz)
    
    x_fine = np.arange(nx_orig)
    z_fine = np.arange(nz_orig)
    
    X_fine, Z_fine = np.meshgrid(x_fine, z_fine, indexing='ij')
    points_fine = np.column_stack([X_fine.ravel(), Z_fine.ravel()])
    
    image_upsampled = interpn(
        (x_coarse, z_coarse),
        image_subsampled,
        points_fine,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    ).reshape(nx_orig, nz_orig)
    
    return image_upsampled


def update_with_box(vp, alpha, dm, vmin=0.5, vmax=4.5):
    assert vp.shape == dm.shape, (vp.shape, dm.shape)
    update = vp + alpha * dm
    vp = np.clip(update, vmin, vmax)
    return vp

def update_image(iter_num):
    """Perform a single image update for the current iteration"""
    start_time = time.time()
    
    image_current = load_image(iter_num) # on model grid
    grad_current = load_gradient(iter_num) # on model grid
    print(image_current.min(), image_current.max())
    alpha = 0.5 / np.max(grad_current)
    print(alpha)
    image_new = update_with_box(image_current, alpha=alpha, dm=grad_current)
    print(image_new.min(), image_new.max())
    np.save(f"{OUTPUT_DIRS['images']}/image_iter_{iter_num+1}.npy", image_new)
    
    elapsed = time.time() - start_time
    print(f"Iteration {iter_num} completed in {elapsed:.2f} seconds")
    
    return image_new

def main():
    """Handle single iteration update"""
    os.makedirs(OUTPUT_DIRS['images'], exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, required=True, help='Current iteration number')
    args = parser.parse_args()
    
    update_image(args.iter)

if __name__ == "__main__":
    main()