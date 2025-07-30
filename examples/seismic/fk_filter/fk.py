import numpy as np
import torch
import torch.fft as fft
from math import ceil
from typing import Tuple
from matplotlib import pyplot as plt


class FKFilter3D:
    def __init__(
        self,
        dx: float = 0.075,
        dz: float = 0.075,
        dt: float = 0.001,
        min_slope: float = 0.1,
        max_slope: float = 200.0,
        lower_min: float = 0.05,
        upper_min: float = 0.05,
        sigma_z: float = 10.0,
        sigma_x: float = 2.5,
        eps: float = 1e-4,
        gaussian_sigma: float = 2.0,
        device: str = "cuda",
    ):
        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.min_slope = min_slope * 1000  # m/ms to m/s
        self.max_slope = max_slope * 1000  # m/ms to m/s
        self.lower_min = lower_min
        self.upper_min = upper_min
        self.sigma_z = sigma_z
        self.sigma_x = sigma_x
        self.eps = eps
        self.gaussian_sigma = gaussian_sigma
        self.device = device

        # Frequency vectors (will be initialized)
        self.freq_z = None
        self.freq_x = None
        self.freq_t = None
        self.filter = None
        self.current_shape = None

    def _compute_filter(self, Z: int, X: int, T: int):
        """Compute filter without full 3D meshgrids"""
        # Create 1D frequency vectors
        self.freq_z = fft.fftshift(fft.fftfreq(Z, d=self.dz)).to(self.device)  # (Z,)
        self.freq_x = fft.fftshift(fft.fftfreq(X, d=self.dx)).to(self.device)  # (X,)
        self.freq_t = fft.rfftfreq(T, d=self.dt).to(self.device)  # (T//2+1,)

        # Reshape for broadcasting - we'll work in 2D (Z,X) first
        fzz = self.freq_z.view(-1, 1)  # (Z, 1)
        fxx = self.freq_x.view(1, -1)  # (1, X)

        with torch.no_grad():

            abs_fxx = torch.abs(fxx)
            lower_bound = self.min_slope * abs_fxx + self.lower_min
            upper_bound = self.max_slope * abs_fxx + self.upper_min
            mask = (fzz > lower_bound) & (fzz < upper_bound)

            # Initialize 2D filter
            buff = torch.where(mask, 1.0, 0.0)
            buff = buff * (1 - self.eps) + self.eps

            # Apply Gaussian blur (Z and X only)
            if self.sigma_z > 0 or self.sigma_x > 0:
                buff = self._gaussian_filter_2d(buff.unsqueeze(0)).squeeze(0)  # Assume your function handles 3D

            # Frequency attenuation in 2D
            r_sq = fxx**2 + fzz**2  # Squared distance from center (Z,X)
            gaussian_attenuation = 1 - torch.exp(-r_sq / (2 * (2 * np.pi * self.gaussian_sigma) ** 2))

            # Combine to create 2D filter
            filter_2d = buff * gaussian_attenuation

            # Expand to 3D by adding T dimension (shape will be [Z,X,1])
            filter_3d = filter_2d.unsqueeze(-1)

            # Broadcast along T dimension to match [Z,X,T//2+1]
            self.filter = filter_3d.expand(-1, -1, len(self.freq_t))

    def _gaussian_filter_2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D Gaussian blur without for loops"""
        Z, X, T = x.shape

        # Convert sigmas to grid points
        sigma_z_grid = float(self.sigma_z / self.dz)
        sigma_x_grid = float(self.sigma_x / self.dx)

        # Create kernels
        kernel_z = self._gaussian_kernel_1d(sigma_z_grid) if sigma_z_grid > 0 else None
        kernel_x = self._gaussian_kernel_1d(sigma_x_grid) if sigma_x_grid > 0 else None

        # Reshape for efficient convolution
        # Combine Z and T dimensions for Z blur
        x_zt = x.permute(1, 0, 2).reshape(1, 1, X, Z * T)  # (1,1,X,Z*T)

        if kernel_z is not None:
            padding_z = len(kernel_z) // 2
            x_zt = torch.nn.functional.conv2d(x_zt, weight=kernel_z.view(1, 1, -1, 1), padding=(padding_z, 0))

        # Reshape back and prepare for X blur
        x_zt = x_zt.view(X, Z, T).permute(1, 0, 2)  # Back to (Z,X,T)
        x_xt = x_zt.reshape(1, 1, Z, X * T)  # (1,1,Z,X*T)

        if kernel_x is not None:
            padding_x = len(kernel_x) // 2
            x_xt = torch.nn.functional.conv2d(x_xt, weight=kernel_x.view(1, 1, 1, -1), padding=(0, padding_x))

        # Final reshape
        return x_xt.view(Z, X, T)

    def _gaussian_kernel_1d(self, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel"""
        radius = ceil(3 * sigma)
        x = torch.linspace(-radius, radius, 2 * radius + 1, device=self.device)
        kernel = torch.exp(-(x**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    # def __call__(self, input: np.ndarray):
    #     """Process single (1, Z, X, T) or (Z, X, T) input"""
    #     # Ensure correct shape (1, Z, X, T)
    #     input_t = torch.as_tensor(input, device=self.device)
    #     if input_t.ndim == 3:
    #         input_t = input_t.unsqueeze(0)
    #     _, Z, X, T = input_t.shape

    #     # Initialize filter if needed
    #     if self.current_shape != (Z, X, T):
    #         self._compute_filter(Z, X, T)
    #         self.current_shape = (Z, X, T)

    #     # Forward FFT (real along last dim)
    #     spectrum = fft.rfftn(input_t)
    #     spectrum = fft.fftshift(spectrum, dim=(-3, -2))  # Shift Z,X

    #     # Apply filter
    #     spectrum = spectrum * self.filter

    #     return fft.irfftn(fft.ifftshift(spectrum, dim=(-3, -2))).squeeze().cpu().numpy()  # Returns (Z, X, T) or (1, Z, X, T)


    def __call__(self, input: torch.Tensor):
        """Process batch of (B, Z, X, T) or single (Z, X, T) input"""
        # Ensure correct shape (B, Z, X, T)
        if input.ndim == 3:
            input = input.unsqueeze(0)  # Add batch dimension if single input
        B, Z, X, T = input.shape

        # Initialize filter if needed
        if self.current_shape != (Z, X, T):
            self._compute_filter(Z, X, T)
            self.current_shape = (Z, X, T)

        # Forward FFT (real along last dim)
        spectrum = fft.rfftn(input)
        spectrum = fft.fftshift(spectrum, dim=(-3, -2))  # Shift Z,X

        # Apply filter - the filter automatically broadcasts to batch dimension
        spectrum = spectrum * self.filter.unsqueeze(0)  # Add batch dimension to filter

        # Inverse FFT and return
        result = fft.irfftn(fft.ifftshift(spectrum, dim=(-3, -2)))
        
        # Return shape matches input shape
        if input.ndim == 3:  # If input was (Z,X,T)
            return result.squeeze(0).cpu()  # Return (Z,X,T)
        return result.cpu()  # Return (B,Z,X,T)

    def plot_filter_slice(self, t_idx=0, lims=None):
        """Visualize filter slice"""
        if self.filter is None:
            raise ValueError("Filter not initialized")

        plt.figure(figsize=(10, 6))
        slice = self.filter[..., t_idx].cpu().numpy()
        freq_x = self.freq_x.cpu().numpy()
        freq_z = self.freq_z.cpu().numpy()
        plt.imshow(
            slice,
            extent=[freq_x.min(), freq_x.max(), freq_z.min(), freq_z.max()],
            origin="lower",
            cmap="viridis",
        )

        if lims:
            plt.xlim([-lims[0], lims[0]])
            plt.ylim([-lims[1], lims[1]])

        plt.xlabel("kx (rad/m)")
        plt.ylabel("kz (rad/m)")
        plt.colorbar(label="Filter amplitude")
        plt.title(f"FK Filter Slice at freq index {t_idx}")
        plt.show()
