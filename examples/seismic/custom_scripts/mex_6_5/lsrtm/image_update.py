# image_update.py - Image updating for a single LSRTM iteration

import numpy as np
import time
from config import *
import os
from scipy.interpolate import interpn
import argparse
from wavefield_computation import setup_model_and_geometry
from grad_computation import get_subn, get_model_shape

def load_gradient(iter_num):
    """Load gradient from binary file for specific iteration"""
    path1 = f"{OUTPUT_DIRS['gradients']}/grad_full_d_{iter_num}.npy"
    path2 = f"{OUTPUT_DIRS['gradients']}/grad_full_u_{iter_num}.npy"
    return np.load(path1), np.load(path2)

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


def update_image(iter_num):
    """Perform a single image update for the current iteration"""
    start_time = time.time()
    
    grad_d, grad_u = load_gradient(iter_num)
    grad_d_prev, grad_u_prev = load_gradient(iter_num - 1) if iter_num > 0 else (grad_d*0, grad_d*0)
    image_current = load_image(iter_num-1) if iter_num > 0 else upsample_image(grad_d*0, *get_model_shape())
    image_prev = load_image(iter_num - 2) if iter_num > 1 else image_current * 0
    grad = grad_u - grad_d
    grad_prev = grad_u_prev - grad_d_prev
    grad = upsample_image(grad, *get_model_shape())
    grad_prev = upsample_image(grad_prev, *get_model_shape())


    sk = image_current - image_prev
    yk = grad - grad_prev
    alpha = get_alfa(yk, sk, iter_num)
    print('\033[1m' + f'{iter_num}. Current alpha - {alpha:.2f}' + '\033[0m')

    image_new = image_current + alpha*grad
    np.save(f"{OUTPUT_DIRS['images']}/image_iter_{iter_num}.npy", image_new)
    elapsed = time.time() - start_time
    print(f"Iteration {iter_num} completed in {elapsed:.2f} seconds")
    
    return image_new

def get_alfa(grad_iter,image_iter,niter_lsrtm):
    term1 = np.dot(image_iter.reshape(-1), image_iter.reshape(-1))
    term2 = np.dot(image_iter.reshape(-1), grad_iter.reshape(-1))
    term3 = np.dot(grad_iter.reshape(-1), grad_iter.reshape(-1))
    
    if niter_lsrtm == 0:
           
        alfa = -0.5
    
    else:
        abb1 = term1 / term2
        abb2 = term2 / term3
        abb3 = abb2 / abb1
        if abb3 > 0 and abb3 < 1:
            alfa = abb2
        else:
            alfa = abb1
            
    return alfa   

def main():
    """Handle single iteration update"""
    os.makedirs(OUTPUT_DIRS['images'], exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, required=True, help='Current iteration number')
    args = parser.parse_args()
    
    update_image(args.iter)

if __name__ == "__main__":
    main()