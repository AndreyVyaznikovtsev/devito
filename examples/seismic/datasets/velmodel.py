import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt


class VelocityModel:
    def __init__(self, path, dx, dz, clip=False, xmin=None, xmax=None, zmin=None, zmax=None):
        """
        Initialize the VelocityModel with a path to the model file.
        
        Parameters:
        path (str): Path to the velocity model file
        dx (float): Desired grid spacing in x-direction
        dz (float): Desired grid spacing in z-direction
        """
        self._path = path
        self._dx = dx
        self._dz = dz
        self.clip = clip
        # Initialize internal state variables
        self._model_loaded = False
        self._current_model = None
        self._xmin = xmin
        self._xmax = xmax
        self._zmin = zmin
        self._zmax = zmax

        
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, new_path):
        if new_path != self._path:
            self._path = new_path
            self._model_loaded = False
    
    @property
    def dx(self):
        return self._dx
    
    @dx.setter
    def dx(self, new_dx):
        if new_dx != self._dx:
            self._dx = new_dx
            self._model_loaded = False
    
    @property
    def dz(self):
        return self._dz
    
    @dz.setter
    def dz(self, new_dz):
        if new_dz != self._dz:
            self._dz = new_dz
            self._model_loaded = False
    
    def _load_and_interpolate_model(self):
            """Load the model from file and interpolate to regular grid"""
            # Read CSV file into DataFrame
            df = pd.read_csv(self._path, sep=r'\s+')
            df_array = df.to_numpy()

            ind = np.ones_like(df_array[:,0]).astype('bool')
            if self._xmin is not None:
                ind = np.logical_and(df_array[:, 0] >= self._xmin, ind)
            if self._xmax is not None:
                ind = np.logical_and(df_array[:, 0] <= self._xmax, ind)
            if self._zmin is not None:
                ind = np.logical_and(df_array[:, 1] >= self._zmin, ind)
            if self._zmax is not None:
                ind = np.logical_and(df_array[:, 1] <= self._zmax, ind)

            df_array = df_array[ind, :] 
            
            x_coords = df_array[:, 0]
            z_coords = df_array[:, 1]  # Assuming z needs to be inverted
            
            
            # Get velocity values
            vel_values = df_array[:, 4]
            vxvz_values = df_array[:, 5]
            if self.clip:
                vel_values = np.clip(vel_values, *np.quantile(vel_values, [0.002, 0.998]))
                vxvz_values = np.clip(vxvz_values, *np.quantile(vxvz_values, [0.001, 0.999]))

            # Create interpolators for irregular grid
            points = np.column_stack((x_coords, z_coords))
            vel_interp = NearestNDInterpolator(points, vel_values)
            vxvz_interp = NearestNDInterpolator(points, vxvz_values)
            
            # Determine grid boundaries
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            
            # Create new regular grid
            new_x = np.arange(x_min, x_max + self._dx, self._dx)
            new_z = np.arange(z_min, z_max + self._dz, self._dz)
            new_xx, new_zz = np.meshgrid(new_x, new_z, indexing='xy')
            
            # Interpolate onto regular grid
            new_vel = vel_interp(new_xx, new_zz)
            new_vxvz = vxvz_interp(new_xx, new_zz)
            print(new_vel.shape)
            
            self._current_model = {
                'vel': new_vel,
                'vxvz': new_vxvz,
                'x': new_x,
                'z': new_z,
                'dx': self._dx,
                'dz': self._dz
            }
            
            self._model_loaded = True
    
    def _interpolate_model(self):
        """Interpolate the model to new grid spacing if needed"""
        if not self._base_model_loaded:
            self._load_base_model()
            
        if self._dx is None and self._dz is None:
            # No interpolation needed
            self._current_model = {
                'vel': self._vel_padded,
                'vxvz': self._vxvz_padded,
                'x': self._original_x,
                'z': self._original_z,
                'dx': self._original_dx,
                'dz': self._original_dz
            }
            return
            
        # Determine new grid dimensions and coordinates
        x_min, x_max = self._original_x[0], self._original_x[-1]
        z_min, z_max = self._original_z[0], self._original_z[-1]
        
        dx = self._original_dx if self._dx is None else self._dx
        dz = self._original_dz if self._dz is None else self._dz
        
        new_x = np.arange(x_min, x_max + dx, dx)
        new_z = np.arange(z_min, z_max + dz, dz)
        
        # Create grid points for interpolation
        xx, zz = np.meshgrid(self._original_x, self._original_z, indexing='xy')
        points = np.column_stack((xx.ravel(), zz.ravel()))
        
        # Create interpolators
        vel_interp = NearestNDInterpolator(points, self._vel_padded.ravel())
        vxvz_interp = NearestNDInterpolator(points, self._vxvz_padded.ravel())
        
        # Create new grid for evaluation
        new_xx, new_zz = np.meshgrid(new_x, new_z, indexing='xy')
        new_points = np.column_stack((new_xx.ravel(), new_zz.ravel()))
        
        # Interpolate
        new_vel = vel_interp(new_points).reshape(len(new_z), len(new_x))
        new_vxvz = vxvz_interp(new_points).reshape(len(new_z), len(new_x))
        
        self._current_model = {
            'vel': new_vel,
            'vxvz': new_vxvz,
            'x': new_x,
            'z': new_z,
            'dx': dx,
            'dz': dz
        }
        
        self._interpolated = True
    
    def _ensure_model_ready(self):
        """Ensure the model is loaded and interpolated"""
        if not self._model_loaded:
            self._load_and_interpolate_model()
    
    @property
    def vp(self):
        self._ensure_model_ready()
        epsilon = self._current_model['vxvz'] - 1
        epsilon[epsilon < 0] = 0.
        return (2 * self._current_model['vel']) / (2 + epsilon)
        # vp = self._current_model['vel'] * 0 + 2.5
        # vp[:vp.shape[0]//2, :] = 3.5
        # return vp
    
    @property
    def epsilon(self):
        self._ensure_model_ready()
        epsilon = self._current_model['vxvz'] - 1
        epsilon[epsilon < 0] = 0.
        return epsilon
    
    @property
    def delta(self):
        self._ensure_model_ready()
        return np.zeros_like(self._current_model['vxvz'])
    
    @property
    def x(self):
        self._ensure_model_ready()
        return self._current_model['x']
    
    @property
    def z(self):
        self._ensure_model_ready()
        return self._current_model['z']
    
    @property
    def shape(self):
        self._ensure_model_ready()
        return self._current_model['vel'].shape
    
    def pad_left(self, n_cells):
        """
        Add padding to the left side of the model using edge values.
        
        Parameters:
        n_cells (int): Number of cells to add
        """
        self._ensure_model_ready()
        
        # Update velocity arrays with edge padding
        self._current_model['vel'] = np.pad(
            self._current_model['vel'],
            ((0, 0), (n_cells, 0)),
            mode='edge'
        )
        self._current_model['vxvz'] = np.pad(
            self._current_model['vxvz'],
            ((0, 0), (n_cells, 0)),
            mode='edge'
        )
        
        # Update x coordinates
        new_x = np.arange(
            self._current_model['x'][0] - n_cells * self._dx,
            self._current_model['x'][-1] + self._dx,
            self._dx
        )
        self._current_model['x'] = new_x

    def pad_right(self, n_cells):
        """
        Add padding to the right side of the model using edge values.
        
        Parameters:
        n_cells (int): Number of cells to add
        """
        self._ensure_model_ready()
        
        # Update velocity arrays with edge padding
        self._current_model['vel'] = np.pad(
            self._current_model['vel'],
            ((0, 0), (0, n_cells)),
            mode='edge'
        )
        self._current_model['vxvz'] = np.pad(
            self._current_model['vxvz'],
            ((0, 0), (0, n_cells)),
            mode='edge'
        )
        
        # Update x coordinates
        new_x = np.arange(
            self._current_model['x'][0],
            self._current_model['x'][-1] + (n_cells + 1) * self._dx,
            self._dx
        )
        self._current_model['x'] = new_x

    def pad_top(self, n_cells):
        """
        Add padding to the top of the model using edge values.
        
        Parameters:
        n_cells (int): Number of cells to add
        """
        self._ensure_model_ready()
        
        # Update velocity arrays with edge padding
        self._current_model['vel'] = np.pad(
            self._current_model['vel'],
            ((n_cells, 0), (0, 0)),
            mode='edge'
        )
        self._current_model['vxvz'] = np.pad(
            self._current_model['vxvz'],
            ((n_cells, 0), (0, 0)),
            mode='edge'
        )
        
        # Update z coordinates
        new_z = np.arange(
            self._current_model['z'][0] - n_cells * self._dz,
            self._current_model['z'][-1] + self._dz,
            self._dz
        )
        self._current_model['z'] = new_z

    def pad_bottom(self, n_cells):
        """
        Add padding to the bottom of the model using edge values.
        
        Parameters:
        n_cells (int): Number of cells to add
        """
        self._ensure_model_ready()
        
        # Update velocity arrays with edge padding
        self._current_model['vel'] = np.pad(
            self._current_model['vel'],
            ((0, n_cells), (0, 0)),
            mode='edge'
        )
        self._current_model['vxvz'] = np.pad(
            self._current_model['vxvz'],
            ((0, n_cells), (0, 0)),
            mode='edge'
        )
        
        # Update z coordinates
        new_z = np.arange(
            self._current_model['z'][0],
            self._current_model['z'][-1] + (n_cells + 1) * self._dz,
            self._dz
        )
        self._current_model['z'] = new_z
    
    def plot_whole(self, show=False, title=r"Модель упругих параметров"):
        """
        Plot the velocity model parameters (Vp, epsilon, delta) using imshow
        
        Parameters:
        show (bool): If True, immediately displays the plot. If False, returns figure and axes
        title (str): Title for the plot
        
        Returns:
        If show=False: (fig, axs) matplotlib figure and axes objects
        If show=True: None
        """        
        # Ensure model is loaded and ready
        self._ensure_model_ready()
        
        # Create figure and axes
        fig, axs = plt.subplots(1, 3, dpi=200, figsize=(15, 10))
        plt.subplots_adjust(wspace=0.01)
        
        # Plot each parameter
        models = [self.vp, self.epsilon, self.delta]
        cmaps = ["turbo", "copper", "bone"]
        params = [r"$V_p$, м/с", r"${\epsilon}$, у.е.", r"${\delta}$, у.е."]
        
        handles = []
        for ax, model, cmap in zip(axs, models, cmaps):
            im = ax.imshow(
                model,
                extent=[self.x[0], self.x[-1], self.z[-1], self.z[0]],
                cmap=cmap,
                aspect='equal',
                interpolation='bilinear'
            )
            handles.append(im)
            # ax.invert_yaxis()
        
        # Configure axes
        axs[0].set_ylabel(r"Глубина, м", fontsize=12)
        for ax, handle, param in zip(axs, handles, params):
            ax.set_xlabel(r"Расстояние, м", fontsize=12)
            cbar = fig.colorbar(handle, shrink=0.4, pad=0.02, ax=ax)
            cbar.ax.get_yaxis().labelpad = 10
            cbar.ax.set_ylabel(param, rotation=90)
        
        fig.suptitle(title, fontsize=14,)
        
        if show:
            plt.show()
        else:
            return fig, axs
        
    
    def plot_vp(self, show=False, title=r"Модель скорости продольных волн", dpi=200, figsize=(5, 10)):
        """
        Plot the velocity model parameters (Vp, epsilon, delta) using imshow
        
        Parameters:
        show (bool): If True, immediately displays the plot. If False, returns figure and axes
        title (str): Title for the plot
        
        Returns:
        If show=False: (fig, axs) matplotlib figure and axes objects
        If show=True: None
        """        
        # Ensure model is loaded and ready
        self._ensure_model_ready()
        
        # Create figure and axes
        fig, axs = plt.subplots(1, 1, dpi=dpi, figsize=figsize)
        axs = [axs]
        plt.subplots_adjust(wspace=0.01)
        
        # Plot each parameter
        models = [self.vp]
        cmaps = ["turbo"]
        params = [r"$V_p$, м/с"]
        
        handles = []
        for ax, model, cmap in zip(axs, models, cmaps):
            im = ax.imshow(
                model,
                extent=[self.x[0], self.x[-1], self.z[-1], self.z[0]],
                cmap=cmap,
                aspect='equal',
                interpolation='bilinear'
            )
            handles.append(im)
            # ax.invert_yaxis()
        
        # Configure axes
        axs[0].set_ylabel(r"Абс. отм., м", fontsize=12)
        for ax, handle, param in zip(axs, handles, params):
            ax.set_xlabel(r"Расстояние, м", fontsize=12)
            cbar = fig.colorbar(handle, shrink=0.4, pad=0.02, ax=ax)
            cbar.ax.get_yaxis().labelpad = 10
            cbar.ax.set_ylabel(param, rotation=90)
        
        ax.set_title(title, fontsize=14,)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            return fig, axs