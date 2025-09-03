import os
import time
import numpy as np
import torch
from examples.seismic.fk_filter import FKFilter3D
from config import *

def get_model_shape():
    return NUM_X, NUM_Z


def get_subn():
    return (el // SUBSAMPLING + 1 for el in get_model_shape())


def load_wavefield_snaps_batch(shot_ids, wavefield_type):
    dir_path = (
        OUTPUT_DIRS["forward_snaps"]
        if wavefield_type == "forward"
        else OUTPUT_DIRS["adjoint_snaps"]
    )
    sub_nx, sub_nz = get_subn()
    batch = torch.empty(
        (len(shot_ids), NSNAPS, sub_nx, sub_nz),
        dtype=torch.float32,
        pin_memory=False,
    )

    for i, shot_id in enumerate(shot_ids):
        path = f"{dir_path}/{shot_id+1}.npy"
        batch[i] = torch.from_numpy(np.load(path))

    return batch


def calc_grad_batch(u0_batch, v_batch, dt=1.0):
    """Compute gradient for batch using torch"""
    u0_padded = torch.nn.functional.pad(
        u0_batch, (0, 0, 0, 0, 1, 1), mode="constant", value=0
    )
    u0_dt2 = (u0_padded[:, 2:] - 2 * u0_padded[:, 1:-1] + u0_padded[:, :-2]) / (dt**2)
    return -torch.sum(u0_dt2 * v_batch, dim=1)


def compute_gradient_batch(shot_ids, fk_down, fk_up, dt):
    """Compute gradient for a batch of shots"""
    u0_batch = load_wavefield_snaps_batch(shot_ids, "forward").to("cuda")
    v_batch = load_wavefield_snaps_batch(shot_ids, "adjoint").to("cuda")

    u0_up = fk_up((u0_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
    u0_down = fk_down((u0_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
    v_up = fk_up((v_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
    v_down = fk_down((v_batch).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

    grad_up = calc_grad_batch(u0_up, v_down, dt=dt)
    grad_down = calc_grad_batch(u0_down, v_up, dt=dt)
    grad_common = calc_grad_batch(u0_batch, v_batch, dt=dt)

    preconditioner_u = 1
    preconditioner_d = 1
    preconditioner_c = 1

    # preconditioner_u = torch.sum(u0_up**2, dim=1)
    # preconditioner_d = torch.sum(u0_down**2, dim=1)
    # preconditioner_c = torch.sum(u0_batch**2, dim=1)

    return grad_up/preconditioner_u, grad_down/preconditioner_d, grad_common/preconditioner_c


def main_compute_gradients_batched(iter_num, batch_size=4):
    """Main function to compute all gradients using batches"""
    os.makedirs(OUTPUT_DIRS["gradients"], exist_ok=True)
    start = time.time()

    min_sl = 1e-3
    leaky = -1
    fk_down = FKFilter3D(
        dx=0.05,
        dz=0.05,
        dt=FK_PARAMS["dt"],
        sigma_x=0.2,
        sigma_z=0.2,
        min_slope=-min_sl,
        max_slope=leaky,
        gaussian_sigma=0,
        lower_min=0.0,
        upper_min=0.0,
        low_cut=0.1,
        high_cut=1.0,
        low_slope=6,
        high_slope=2,
        device="cuda",
    )

    fk_up = FKFilter3D(
        dx=0.05,
        dz=0.05,
        dt=FK_PARAMS["dt"],
        sigma_x=0.2,
        sigma_z=0.2,
        min_slope=min_sl,
        max_slope=leaky,
        gaussian_sigma=0,
        lower_min=0.0,
        upper_min=0.0,
        low_cut=0.1,
        high_cut=1.0,
        low_slope=6,
        high_slope=2,
        device="cuda",
    )
     
    sub_nx, sub_nz = get_subn()
    fk_down._compute_filter(sub_nz, sub_nx, NSNAPS)
    fk_up._compute_filter(sub_nz, sub_nx, NSNAPS)

    grad_full_u = torch.zeros((sub_nx, sub_nz), device="cuda")
    grad_full_d = torch.zeros((sub_nx, sub_nz), device="cuda")
    grad_full_c = torch.zeros((sub_nx, sub_nz), device="cuda")


    num_shots = NUM_SHOTS
    all_grad_u = torch.zeros((num_shots, sub_nx, sub_nz), device="cuda")
    all_grad_d = torch.zeros((num_shots, sub_nx, sub_nz), device="cuda")
    all_grad_c = torch.zeros((num_shots, sub_nx, sub_nz), device="cuda")
    

    for batch_start in range(0, num_shots, batch_size):
        batch_end = min(batch_start + batch_size, num_shots)
        shot_ids = range(batch_start, batch_end)
        print(f"Computing gradient for shots {batch_start+1}-{batch_end}/{num_shots}")

        grad_u_batch, grad_d_batch, grad_c_batch = compute_gradient_batch(
            shot_ids, fk_down=fk_down, fk_up=fk_up, dt=TMAX / (NSNAPS + 1),
        )

        all_grad_u[batch_start:batch_end] = grad_u_batch
        all_grad_d[batch_start:batch_end] = grad_d_batch
        all_grad_c[batch_start:batch_end] = grad_c_batch

        grad_full_u += torch.sum(grad_u_batch, dim=0)
        grad_full_d += torch.sum(grad_d_batch, dim=0)
        grad_full_c += torch.sum(grad_c_batch, dim=0)


    grad_full_u /= num_shots
    grad_full_d /= num_shots
    grad_full_c /= num_shots

    np.save(f"{OUTPUT_DIRS['gradients']}/grad_full_u_{iter_num}.npy", grad_full_u.cpu().numpy())
    np.save(f"{OUTPUT_DIRS['gradients']}/grad_full_d_{iter_num}.npy", grad_full_d.cpu().numpy())
    np.save(f"{OUTPUT_DIRS['gradients']}/grad_full_c_{iter_num}.npy", grad_full_c.cpu().numpy())

    np.save(f"{OUTPUT_DIRS['gradients']}/all_grad_u_{iter_num}.npy", all_grad_u.cpu().numpy())
    np.save(f"{OUTPUT_DIRS['gradients']}/all_grad_d_{iter_num}.npy", all_grad_d.cpu().numpy())
    np.save(f"{OUTPUT_DIRS['gradients']}/all_grad_c_{iter_num}.npy", all_grad_c.cpu().numpy())


    end = time.time()
    print(f"Gradient computation completed in {end - start:.2f} seconds")


if __name__== "__main__":
    main_compute_gradients_batched(0, 4)