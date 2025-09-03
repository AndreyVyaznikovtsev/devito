# rtm_core/image_update.py
import numpy as np
import os
import time
from scipy.interpolate import interpn


def upsample_gradient(grad, nx_orig, nz_orig):
    """Upsample subsampled gradient to match vp shape"""
    sub_nx, sub_nz = grad.shape

    x_coarse = np.linspace(0, nx_orig - 1, sub_nx)
    z_coarse = np.linspace(0, nz_orig - 1, sub_nz)

    x_fine = np.arange(nx_orig)
    z_fine = np.arange(nz_orig)

    X_fine, Z_fine = np.meshgrid(x_fine, z_fine, indexing="ij")
    points_fine = np.column_stack([X_fine.ravel(), Z_fine.ravel()])

    grad_upsampled = interpn(
        (x_coarse, z_coarse),
        grad,
        points_fine,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    ).reshape(nx_orig, nz_orig)

    return grad_upsampled


def get_alpha(vp, vp_prev, grad, grad_prev, iter_num):
    """Compute Barzilai–Borwein step length"""
    sk = (vp - vp_prev).ravel()
    yk = (grad - grad_prev).ravel()

    term1 = np.dot(sk, sk)
    term2 = np.dot(sk, yk)
    term3 = np.dot(yk, yk)
    print(term1, term2, term3)
    if iter_num == 0:
        return 0.05
    else:
        return 0.05
        # abb1 = term1 / (term2 + 1e-12)
        # abb2 = term2 / (term3 + 1e-12)
        # abb3 = abb2 / (abb1 + 1e-12)
        # print(abb1, abb2)
        # if 0 < abb3 < 1:
        #     return abb2
        # else:
        #     return abb1


def update_dm(config, iter_num):
    """Update velocity model at given iteration"""
    start = time.time()
    nx, nz = config.NUM_X, config.NUM_Z

    vp_curr = np.load(f"{config.OUTPUT_DIRS['images']}/vp_iter_{iter_num}.npy")
    vp_prev = (
        np.load(f"{config.OUTPUT_DIRS['images']}/vp_iter_{iter_num-1}.npy")
        if iter_num > 0 else vp_curr
    )

    grad_curr = np.load(f"{config.OUTPUT_DIRS['gradients']}/grad_full_c_{iter_num}.npy")
    grad_prev = (
        np.load(f"{config.OUTPUT_DIRS['gradients']}/grad_full_c_{iter_num-1}.npy")
        if iter_num > 0 else np.zeros_like(grad_curr)
    )

    # Upsample gradients
    grad_curr = upsample_gradient(grad_curr, nx, nz)
    grad_prev = upsample_gradient(grad_prev, nx, nz)

    # Compute step length
    alpha = get_alpha(vp_curr, vp_prev, grad_curr, grad_prev, iter_num)
    print(f"[Iter {iter_num}] alpha = {alpha:.4f}")

    # grad_norm = grad_curr / (np.max(np.abs(grad_curr)) + 1e-12)
    grad_norm = grad_curr / (np.quantile(grad_curr, 0.99) + 1e-12)


    vp_new = vp_prev + alpha * grad_norm
    vp_new = np.clip(vp_new, 0.6, 5.5)
    os.makedirs(config.OUTPUT_DIRS["images"], exist_ok=True)
    np.save(f"{config.OUTPUT_DIRS['images']}/vp_iter_{iter_num+1}.npy", vp_new)

    print(f"Iteration {iter_num} update completed in {time.time() - start:.2f}s")
    return vp_new
