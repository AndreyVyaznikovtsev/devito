# image_update.py - Image updating for a single LSRTM iteration

import numpy as np
import time
from config import *
import os
from scipy.interpolate import interpn
import argparse

def get_model_shape():
    return 2630, 3640

def get_num_shots():
    return 59

def load_gradient(iter_num):
    """Load gradient from binary file for specific iteration"""
    path = f"{OUTPUT_DIRS['gradients']}/grad_full_{iter_num}.bin"
    nx, nz = get_model_shape()
    sub_nx = nx // SUBSAMPLING + 1
    sub_nz = nz // SUBSAMPLING + 1

    if iter_num < 0:  # Initial case
        return np.zeros((nx, nz), dtype=dtype)
    
    buff = np.fromfile(path, dtype=dtype).reshape(sub_nx, sub_nz)
    return upsample_image(buff, *get_model_shape()) # upsample on loading

def load_image(iter_num):
    """Load image from previous iteration"""
    nx, nz = get_model_shape()
    sub_nx = nx // SUBSAMPLING + 1
    sub_nz = nz // SUBSAMPLING + 1
    
    if iter_num < 1:  # Initial case
        return np.zeros((nx, nz), dtype=dtype)
    
    path = f"{OUTPUT_DIRS['images']}/image_iter_{iter_num - 1}.bin"
    return np.fromfile(path).reshape(nx, nz)

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

def get_alfa(grad_iter, image_iter, niter_lsrtm, grad_full):
    """Compute step size for LSRTM"""
    term1 = np.dot(image_iter.reshape(-1), image_iter.reshape(-1))
    term2 = np.dot(image_iter.reshape(-1), grad_iter.reshape(-1))
    term3 = np.dot(grad_iter.reshape(-1), grad_iter.reshape(-1))
    
    if niter_lsrtm == 0:
        alfa = .05 / np.max(grad_full)
    else:
        abb1 = term1 / term2
        abb2 = term2 / term3
        abb3 = abb2 / abb1
        if abb3 > 0 and abb3 < 1:
            alfa = abb2
        else:
            alfa = abb1
            
    return alfa

def update_image(iter_num):
    """Perform a single image update for the current iteration"""
    start_time = time.time()
    
    # Load previous image and gradient
    image_prev = load_image(iter_num - 1) # on model grid
    image_current = load_image(iter_num) # on model grid
    grad_prev = load_gradient(iter_num - 1) # on model grid
    grad_illum = load_gradient(iter_num) # on model grid

    # Compute step size
    yk = grad_illum - grad_prev
    sk = image_current - image_prev
    alfa = get_alfa(yk, sk, iter_num, grad_illum)
    # Update image
    image_new = image_prev - alfa * grad_illum
    # Save image
    os.makedirs(OUTPUT_DIRS['images'], exist_ok=True)
    image_new.tofile(f"{OUTPUT_DIRS['images']}/image_iter_{iter_num}.bin")
    
    # Save first iteration as migration result
    if iter_num == 0:
        np.save(f"{OUTPUT_DIRS['images']}/migration_result.npy", image_new)
    print(image_new.shape)
    elapsed = time.time() - start_time
    print(f"Iteration {iter_num} completed in {elapsed:.2f} seconds")
    
    return image_new

def main():
    """Handle single iteration update"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, required=True, help='Current iteration number')
    args = parser.parse_args()
    
    update_image(args.iter)

if __name__ == "__main__":
    main()