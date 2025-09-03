import os
import time
import numpy as np
import torch
from examples.seismic.fk_filter import FKFilter3D


def get_model_shape(config):
    return config.NUM_X, config.NUM_Z


def get_subn(config):
    return (el // config.SUBSAMPLING + 1 for el in get_model_shape(config))


def load_wavefield_snaps_batch(shot_ids, wavefield_type, config):
    dir_path = (
        config.OUTPUT_DIRS["forward_snaps"]
        if wavefield_type == "forward"
        else config.OUTPUT_DIRS["adjoint_snaps"]
    )
    sub_nx, sub_nz = get_subn(config)
    batch = torch.empty(
        (len(shot_ids), config.NSNAPS, sub_nx, sub_nz),
        dtype=torch.float32,
        pin_memory=False,
    )

    for i, shot_id in enumerate(shot_ids):
        path = f"{dir_path}/{shot_id+1}.npy"
        batch[i] = torch.from_numpy(np.load(path))

    return batch


def calc_grad_batch(v_batch, u0_batch, dt=1.0):
    """Compute gradient for batch using torch"""
    u0_padded = torch.nn.functional.pad(
        u0_batch, (0, 0, 0, 0, 1, 1), mode="constant", value=0
    )
    u0_dt2 = (u0_padded[:, 2:] - 2 * u0_padded[:, 1:-1] + u0_padded[:, :-2]) / (dt**2)
    return -torch.sum(u0_dt2 * v_batch, dim=1)


def compute_gradient_batch(shot_ids, fk_down, fk_up, dt, config):
    """Compute gradient for a batch of shots"""
    u0_batch = load_wavefield_snaps_batch(shot_ids, "forward", config).to("cuda")
    v_batch = load_wavefield_snaps_batch(shot_ids, "adjoint", config).to("cuda")

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

    # preconditioner_u = torch.sum(v_down**2, dim=1)
    # preconditioner_d = torch.sum(v_up**2, dim=1)
    # preconditioner_c = torch.sum(v_batch**2, dim=1)

    return grad_up/preconditioner_u, grad_down/preconditioner_d, grad_common/preconditioner_c


def main_compute_gradients_batched(iter_num, config, batch_size=4):
    """Main function to compute all gradients using batches"""
    os.makedirs(config.OUTPUT_DIRS["gradients"], exist_ok=True)
    start = time.time()


    fk_down = FKFilter3D(
        dx=config.FK_PARAMS["dx"],
        dz=config.FK_PARAMS["dz"],
        dt=config.FK_PARAMS["dt"],
        sigma_x=config.FK_PARAMS["sigma_x"],
        sigma_z=config.FK_PARAMS["sigma_z"],
        min_slope=config.FK_PARAMS["min_slope_down"],
        max_slope=config.FK_PARAMS["max_slope_down"],
        gaussian_sigma=config.FK_PARAMS["gaussian_sigma"],
        lower_min=config.FK_PARAMS["lower_min"],
        upper_min=config.FK_PARAMS["upper_min"],
        low_cut=config.FK_PARAMS["low_cut"],
        high_cut=config.FK_PARAMS["high_cut"],
        low_slope=config.FK_PARAMS["low_slope"],
        high_slope=config.FK_PARAMS["high_slope"],
        device="cuda",
    )

    fk_up = FKFilter3D(
        dx=config.FK_PARAMS["dx"],
        dz=config.FK_PARAMS["dz"],
        dt=config.FK_PARAMS["dt"],
        sigma_x=config.FK_PARAMS["sigma_x"],
        sigma_z=config.FK_PARAMS["sigma_z"],
        min_slope=config.FK_PARAMS["min_slope_up"],
        max_slope=config.FK_PARAMS["max_slope_up"],
        gaussian_sigma=config.FK_PARAMS["gaussian_sigma"],
        lower_min=config.FK_PARAMS["lower_min"],
        upper_min=config.FK_PARAMS["upper_min"],
        low_cut=config.FK_PARAMS["low_cut"],
        high_cut=config.FK_PARAMS["high_cut"],
        low_slope=config.FK_PARAMS["low_slope"],
        high_slope=config.FK_PARAMS["high_slope"],
        device="cuda",
    )

    sub_nx, sub_nz = get_subn(config)
    fk_down._compute_filter(sub_nz, sub_nx, config.NSNAPS)
    fk_up._compute_filter(sub_nz, sub_nx, config.NSNAPS)

    grad_full_u = torch.zeros((sub_nx, sub_nz), device="cuda")
    grad_full_d = torch.zeros((sub_nx, sub_nz), device="cuda")
    grad_full_c = torch.zeros((sub_nx, sub_nz), device="cuda")


    shot_ids_list = config.SHOT_IDS
    num_shots = len(shot_ids_list)
    print(f"Processing {num_shots} shots: {shot_ids_list}")

    all_grad_u = torch.zeros((num_shots, sub_nx, sub_nz), device="cuda")
    all_grad_d = torch.zeros((num_shots, sub_nx, sub_nz), device="cuda")
    all_grad_c = torch.zeros((num_shots, sub_nx, sub_nz), device="cuda")

    for batch_start in range(0, num_shots, batch_size):
        batch_end = min(batch_start + batch_size, num_shots)
        current_batch_ids = shot_ids_list[batch_start:batch_end]

        # print(f"Computing gradient for shots {batch_start+1}-{batch_end}/{num_shots}")

        grad_u_batch, grad_d_batch, grad_c_batch = compute_gradient_batch(
            current_batch_ids, fk_down=fk_down, fk_up=fk_up, dt=config.TMAX / (config.NSNAPS + 1), config=config
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

    np.save(f"{config.OUTPUT_DIRS['gradients']}/grad_full_u_{iter_num}.npy", grad_full_u.cpu().numpy())
    np.save(f"{config.OUTPUT_DIRS['gradients']}/grad_full_d_{iter_num}.npy", grad_full_d.cpu().numpy())
    np.save(f"{config.OUTPUT_DIRS['gradients']}/grad_full_c_{iter_num}.npy", grad_full_c.cpu().numpy())

    np.save(f"{config.OUTPUT_DIRS['gradients']}/all_grad_u_{iter_num}.npy", all_grad_u.cpu().numpy())
    np.save(f"{config.OUTPUT_DIRS['gradients']}/all_grad_d_{iter_num}.npy", all_grad_d.cpu().numpy())
    np.save(f"{config.OUTPUT_DIRS['gradients']}/all_grad_c_{iter_num}.npy", all_grad_c.cpu().numpy())


    end = time.time()
    print(f"Gradient computation completed in {end - start:.2f} seconds")
