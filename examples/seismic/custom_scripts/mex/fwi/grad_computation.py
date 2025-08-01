# gradient_computation_batched.py - Batched PyTorch-based FK filtering and gradient computation

import numpy as np
import time
from config import *
from scipy.interpolate import interpn
import os
from examples.seismic.fk_filter import FKFilter3D
import argparse
import torch
from wavefield_computation import setup_model_and_geometry

def get_model_shape():
    return 2630, 3640

def get_num_shots():
    return 59

def load_wavefield_snaps_batch(shot_ids, wavefield_type='forward'):
    dir_path = OUTPUT_DIRS['forward_snaps'] if wavefield_type == 'forward' else OUTPUT_DIRS['adjoint_snaps']
    nx, nz = get_model_shape()
    sub_nx = nx // SUBSAMPLING + 1
    sub_nz = nz // SUBSAMPLING + 1
    
    batch = torch.empty((len(shot_ids), NSNAPS, sub_nx, sub_nz),
                    dtype=torch.float32,
                    pin_memory=True)
    
    for i, shot_id in enumerate(shot_ids):
        path = f"{dir_path}/{shot_id+1}.npy"
        batch[i] = torch.from_numpy(np.load(path))  # Copy forces read into memory
    
    return batch

def compute_gradient_batch(shot_ids, fk_down, fk_up, dt):
    """Compute gradient for a batch of shots"""
    # Load wavefields in batch
    u0_batch = load_wavefield_snaps_batch(shot_ids, 'forward')  # [B, T, X, Z]
    v_batch = load_wavefield_snaps_batch(shot_ids, 'adjoint')   # [B, T, X, Z]
    
    # Apply filters
    # u0_up = fk_up((u0_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # Back to [B, T, X, Z]
    # u0_down = fk_down((u0_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
    # v_up = fk_up((v_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
    # v_down = fk_down((v_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
    
    # Compute gradients for each shot in batch
    # grad_up = calc_grad_batch(u0_up.numpy(), v_down.numpy(), dt=0.13876766)
    # grad_down = calc_grad_batch(u0_down.numpy(), v_up.numpy(), dt=0.13876766)
    
    # # Combine gradients and compute illumination
    # grad_batch = grad_up + grad_down
    grad_batch = calc_grad_batch(u0_batch.numpy(), v_batch.numpy(), dt=dt)
    
    return grad_batch  # [B, X, Z], [B, X, Z]

def calc_grad_batch(u0_batch, v_batch, dt=None):
    """Compute gradient for batch using numpy"""
    if dt is None:
        dt = 1.0  # Default value, should be set from config
    
    # Pad v_data with zeros at t=-1 and t=nt
    v_padded = np.pad(v_batch, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='constant')
    
    # Compute second time derivative (centered FD)
    v_dt2 = (v_padded[:, 2:] - 2*v_padded[:, 1:-1] + v_padded[:, :-2]) / (dt**2)
    
    # Gradient = -∑(u0 * v_dt2) over time
    return -np.sum(u0_batch * v_dt2, axis=1)

def main_compute_gradients_batched(iter, batch_size=4):
    """Main function to compute all gradients using batches"""
    start = time.time()
    model, _ = setup_model_and_geometry(iter)

    # Initialize FK filters
    fk_down = FKFilter3D(
        dx=FK_PARAMS['dx'],
        dz=FK_PARAMS['dz'],
        dt=FK_PARAMS['dt'],
        sigma_x=FK_PARAMS['sigma_x'],
        sigma_z=FK_PARAMS['sigma_z'],
        min_slope=FK_PARAMS['min_slope_down'],
        max_slope=FK_PARAMS['max_slope_down'],
        gaussian_sigma=FK_PARAMS['gaussian_sigma'],
        lower_min=FK_PARAMS['lower_min'],
        upper_min=FK_PARAMS['upper_min'],
        device='cuda'  # Use GPU for batch processing
    )
    
    fk_up = FKFilter3D(
        dx=FK_PARAMS['dx'],
        dz=FK_PARAMS['dz'],
        dt=FK_PARAMS['dt'],
        sigma_x=FK_PARAMS['sigma_x'],
        sigma_z=FK_PARAMS['sigma_z'],
        min_slope=FK_PARAMS['min_slope_up'],
        max_slope=FK_PARAMS['max_slope_up'],
        gaussian_sigma=FK_PARAMS['gaussian_sigma'],
        lower_min=FK_PARAMS['lower_min'],
        upper_min=FK_PARAMS['upper_min'],
        device='cuda'  # Use GPU for batch processing
    )
    
    # Compute shape and initialize filters
    nx, nz = get_model_shape()
    sub_nx = nx // SUBSAMPLING + 1
    sub_nz = nz // SUBSAMPLING + 1
    fk_down._compute_filter(sub_nz, sub_nx, NSNAPS)
    fk_up._compute_filter(sub_nz, sub_nx, NSNAPS)
    
    grad_full = np.zeros((sub_nx, sub_nz), dtype=np.float32)
    
    num_shots = get_num_shots()
    for batch_start in range(0, num_shots, batch_size):
        batch_end = min(batch_start + batch_size, num_shots)
        shot_ids = range(batch_start, batch_end)
        print(f"Computing gradient for shots {batch_start+1}-{batch_end}/{num_shots}")
        grad_batch = compute_gradient_batch(shot_ids, fk_down, fk_up, TMAX/(NSNAPS+1))
        grad_full += np.sum(grad_batch, axis=0)
    
    # shot_ids = range(0, num_shots, 10)
    # grad_batch = compute_gradient_batch(shot_ids, fk_down, fk_up)
    # print(grad_batch.shape)
    # grad_full += np.sum(grad_batch, axis=0)

    # Save results
    np.save(f"{OUTPUT_DIRS['gradients']}/grad_full_{iter}.npy", grad_full)
    
    end = time.time()
    print(f"Gradient computation completed in {end - start:.2f} seconds")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRS['gradients'], exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    main_compute_gradients_batched(args.iter, args.batch_size)