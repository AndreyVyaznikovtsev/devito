# rtm_core/image_update.py
import numpy as np
import os
import time
from scipy.interpolate import interpn
from scipy import stats
from scipy import ndimage
from scipy.ndimage import laplace, gaussian_filter


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

def process_model(velmodel, dataset, grad_c0, invert_wells=False, scalar=1, 
                 apply_func=lambda x: laplace(x), smooth_transition=True, transition_width=20):
    extent = [velmodel.x[0], velmodel.x[-1], velmodel.z[-1], velmodel.z[0]]

    if invert_wells:
        slope2, intercept2, _, _, _ = stats.linregress(dataset.x_coords - 1, dataset.elevations)
        slope1, intercept1, _, _, _ = stats.linregress(dataset.opposite_x + 1, dataset.opposite_elev)
    else:
        slope1, intercept1, _, _, _ = stats.linregress(dataset.x_coords + 1, dataset.elevations)
        slope2, intercept2, _, _, _ = stats.linregress(dataset.opposite_x - 1, dataset.opposite_elev)        

    # Create meshgrid for masking
    x = np.linspace(extent[0], extent[1], grad_c0.shape[0])
    z = np.linspace(extent[3], extent[2], grad_c0.shape[1])
    xx, zz = np.meshgrid(x, z, indexing='ij')

    # Create distance-based masks for smooth transitions
    if smooth_transition:
        # Distance to well boundaries
        if slope1 != 0:
            dist1 = (xx - (zz - intercept1)/slope1)
        else:
            dist1 = xx - intercept1
        
        if slope2 != 0:
            dist2 = ((zz - intercept2)/slope2 - xx)
        else:
            dist2 = intercept2 - xx
        
        # Create smooth masks using sigmoid function
        mask1 = 1 / (1 + np.exp(-dist1 / (transition_width * 0.1)))
        mask2 = 1 / (1 + np.exp(-dist2 / (transition_width * 0.1)))
        
        # Combine masks - we want areas where both masks are high
        between_wells_mask = mask1 * mask2
        
    else:
        # Original binary mask
        mask1 = xx > (zz - intercept1)/slope1 if slope1 != 0 else xx > intercept1
        mask2 = xx < (zz - intercept2)/slope2 if slope2 != 0 else xx < intercept2
        between_wells_mask = mask1 & mask2
        between_wells_mask = between_wells_mask.astype(np.float32)

    # Create border mask with smooth transition
    border_width = 30
    border_mask = create_smooth_border_mask(grad_c0.shape, border_width)
    grad_c0_masked = apply_func(grad_c0) * between_wells_mask * border_mask
    
    return (grad_c0_masked, 
            between_wells_mask, border_mask)

def create_smooth_border_mask(shape, border_width):
    """
    Create a mask with smooth transition from 1 in center to 0 at edges
    """
    rows, cols = shape
    border_mask = np.ones(shape)
    y_dist = np.minimum(np.arange(rows)[:, np.newaxis], 
                       rows - 1 - np.arange(rows)[:, np.newaxis])
    x_dist = np.minimum(np.arange(cols), cols - 1 - np.arange(cols))
    dist_to_edge = np.minimum(y_dist, x_dist[np.newaxis, :])
    border_mask = np.where(dist_to_edge < border_width, 
                         1 - 0.5 * (1 + np.cos(np.pi * dist_to_edge / border_width)),
                         1.0)
    
    return border_mask

# Alternative: Gaussian smoothing approach
def smooth_mask_transition(mask, sigma=2):
    smoothed = ndimage.gaussian_filter(mask.astype(float), sigma=sigma)
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    return smoothed


def update_vp(config, iter_num):
    """Update velocity model at given iteration"""
    start = time.time()
    nx, nz = config.NUM_X, config.NUM_Z
    model, dataset_dplus, dataset_sub, velmodel = config.setup_model_and_geometry(iter_num)


    vp_curr = np.load(f"{config.OUTPUT_DIRS['images']}/vp_iter_{iter_num}.npy")

    all_grad_u0 = np.load(f"{config.OUTPUT_DIRS['gradients']}/all_grad_u_{iter_num}.npy")
    all_grad_d0 = np.load(f"{config.OUTPUT_DIRS['gradients']}/all_grad_d_{iter_num}.npy")
    for i in range(all_grad_u0.shape[0]):
        all_grad_u0[i] /= np.quantile(all_grad_u0[i], 0.999)
        all_grad_d0[i] /= np.quantile(all_grad_d0[i], 0.999)
    grad_u0 = stats.trim_mean(all_grad_u0, 0.0, axis=0)
    grad_d0 = stats.trim_mean(all_grad_d0, 0.0, axis=0)

    all_grad_c0 = np.load(f"{config.OUTPUT_DIRS['gradients']}/all_grad_c_{iter_num}.npy")
    for i in range(all_grad_c0.shape[0]):
        all_grad_c0[i] /= np.quantile(all_grad_c0[i], 0.999)
    grad_curr = stats.trim_mean(all_grad_c0, 0.0, axis=0)
    # Upsample gradients
    grad_curr = grad_u0+grad_d0
    grad_curr, mask1, mask2 = process_model(velmodel=velmodel, dataset=dataset_dplus, grad_c0=grad_curr,
                                            invert_wells=config.INVERT_WELLS,
                                            apply_func=lambda x: x, #laplace(gaussian_filter(x, sigma=0.75)),
                                            transition_width=5
                                            )
    grad_curr = upsample_gradient(grad_curr, nx, nz)


    grad_norm = grad_curr / np.quantile(grad_curr, 0.99)

    # Calculate learning rate that decreases linearly from 0.2 to 0.05 over 10 iterations
    if iter_num < 10:
        learning_rate = 0.2 - (0.2 - 0.02) * (iter_num / 9)  # Linear decrease
    else:
        learning_rate = 0.02  # Stay at 0.05 after 10 iterations

    vp_new = vp_curr - learning_rate * grad_norm
    vp_new = np.clip(vp_new, 0.6, 6.0)
    os.makedirs(config.OUTPUT_DIRS["images"], exist_ok=True)
    np.save(f"{config.OUTPUT_DIRS['images']}/vp_iter_{iter_num+1}.npy", vp_new)

    print(f"Iteration {iter_num} update completed in {time.time() - start:.2f}s")
    return vp_new
