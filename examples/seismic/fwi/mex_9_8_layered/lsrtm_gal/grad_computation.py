import numpy as np
import time
from config import *
import os
from examples.seismic.fk_filter import FKFilter3D
import argparse
import torch

def get_model_shape():
    return NUM_X, NUM_Z

def get_num_shots():
    return NUM_SHOTS

def get_subn():
    return (el//SUBSAMPLING + 1 for el in get_model_shape())

def load_wavefield_snaps_batch(shot_ids, wavefield_type='forward'):
    dir_path = OUTPUT_DIRS['forward_snaps'] if wavefield_type == 'forward' else OUTPUT_DIRS['adjoint_snaps']
    sub_nx, sub_nz = get_subn()
    batch = torch.empty((len(shot_ids), NSNAPS, sub_nx, sub_nz),
                    dtype=torch.float32,
                    pin_memory=False)
    
    for i, shot_id in enumerate(shot_ids):
        path = f"{dir_path}/{shot_id+1}.npy"
        batch[i] = torch.from_numpy(np.load(path))  # Copy forces read into memory
    
    return batch

def load_scalers(shot_ids):
    scalers = torch.empty(len(shot_ids), dtype=torch.float32, pin_memory=True)
    dir_path = PATH_WAVELETS
    for i, shot_id in enumerate(shot_ids):
        path = f"{dir_path}/wavelet_scale_{shot_id}.npy"
        scalers[i] = torch.from_numpy(np.load(path))  # Copy forces read into memory
    
    return scalers

def calc_grad_batch(u0_batch, v_batch, dt=None):
    """Compute gradient for batch using torch"""
    if dt is None:
        dt = 1.0  # Default value, should be set from config
    # v_padded = torch.nn.functional.pad(v_batch, (0, 0, 0, 0, 1, 1), mode='constant', value=0)
    # v_dt2 = (v_padded[:, 2:] - 2*v_padded[:, 1:-1] + v_padded[:, :-2]) / (dt**2)
    u0_padded = torch.nn.functional.pad(u0_batch, (0, 0, 0, 0, 1, 1), mode='constant', value=0)
    u0_dt2 = (u0_padded[:, 2:] - 2*u0_padded[:, 1:-1] + u0_padded[:, :-2]) / (dt**2)   
    # return -torch.sum(u0_batch * v_dt2, dim=1)
    return -torch.sum(u0_dt2 * v_batch, dim=1)


def compute_gradient_batch(shot_ids, fk_down, fk_up, dt, scalers=None):
    """Compute gradient for a batch of shots"""
    # Load wavefields in batch
    u0_batch = load_wavefield_snaps_batch(shot_ids, 'forward').to('cuda')  # [B, T, X, Z]
    v_batch = load_wavefield_snaps_batch(shot_ids, 'adjoint').to('cuda')   # [B, T, X, Z]
    
    # Apply filters
    u0_up = fk_up((u0_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # Back to [B, T, X, Z]
    u0_down = fk_down((u0_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
    v_up = fk_up((v_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
    v_down = fk_down((v_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
    
    # Compute gradients for each shot in batch
    grad_up = calc_grad_batch(u0_up, v_down, dt=dt)
    grad_down = calc_grad_batch(u0_down, v_up, dt=dt)
    
    if scalers is not None:
        grad_up = grad_up / scalers.unsqueeze(-1).unsqueeze(-1)
        grad_down = grad_down / scalers.unsqueeze(-1).unsqueeze(-1)
    
    # return grad_up, grad_down, u0_up, u0_down
    return grad_up, grad_down, v_down, v_up

def main_compute_gradients_batched(iter, batch_size=4):
    """Main function to compute all gradients using batches"""
    start = time.time()

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
        cutoff_freq=FK_PARAMS['cutoff'],
        order=FK_PARAMS['order'],
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
        cutoff_freq=FK_PARAMS['cutoff'],
        order=FK_PARAMS['order'],
        device='cuda'  # Use GPU for batch processing
    )
    
    # Compute shape and initialize filters
    sub_nx, sub_nz = get_subn()
    fk_down._compute_filter(sub_nz, sub_nx, NSNAPS)
    fk_up._compute_filter(sub_nz, sub_nx, NSNAPS)
    
    grad_full_u = torch.zeros((sub_nx, sub_nz), device='cuda')
    grad_full_d = torch.zeros((sub_nx, sub_nz), device='cuda')
    
    # Initialize arrays to store all shot gradients
    num_shots = get_num_shots()
    all_grad_u = torch.zeros((num_shots, sub_nx, sub_nz), device='cuda')
    all_grad_d = torch.zeros((num_shots, sub_nx, sub_nz), device='cuda')
    
    for batch_start in range(0, num_shots, batch_size):
        batch_end = min(batch_start + batch_size, num_shots)
        shot_ids = range(batch_start, batch_end)
        print(f"Computing gradient for shots {batch_start+1}-{batch_end}/{num_shots}")

        # scalers = load_scalers(shot_ids).to('cuda')
        
        # Get gradients for this batch
        grad_u_batch, grad_d_batch, illum_up, illum_down = compute_gradient_batch(
            shot_ids, fk_down=fk_down, fk_up=fk_up, dt=TMAX/(NSNAPS+1), scalers=None
        )
        
        grad_u_batch = grad_u_batch / (torch.sum(illum_up  **2, dim=1) + 1e-12)
        grad_d_batch = grad_d_batch / (torch.sum(illum_down**2, dim=1) + 1e-12)
        
        # Store individual shot gradients
        all_grad_u[batch_start:batch_end] = grad_u_batch
        all_grad_d[batch_start:batch_end] = grad_d_batch
        
        # Accumulate for full gradient
        grad_full_u += torch.sum(grad_u_batch, dim=0)
        grad_full_d += torch.sum(grad_d_batch, dim=0)

    grad_full_u /= num_shots
    grad_full_d /= num_shots

    # Save results
    os.makedirs(OUTPUT_DIRS['gradients'], exist_ok=True)
    np.save(f"{OUTPUT_DIRS['gradients']}/grad_full_u_{iter}.npy", grad_full_u.cpu().numpy())
    np.save(f"{OUTPUT_DIRS['gradients']}/grad_full_d_{iter}.npy", grad_full_d.cpu().numpy())
    
    # Save all individual shot gradients
    np.save(f"{OUTPUT_DIRS['gradients']}/all_grad_u_{iter}.npy", all_grad_u.cpu().numpy())
    np.save(f"{OUTPUT_DIRS['gradients']}/all_grad_d_{iter}.npy", all_grad_d.cpu().numpy())
    
    end = time.time()
    print(f"Gradient computation completed in {end - start:.2f} seconds")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRS['gradients'], exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    main_compute_gradients_batched(args.iter, args.batch_size)