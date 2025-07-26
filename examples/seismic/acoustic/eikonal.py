import numpy as np
from typing import Tuple, Dict
from pykonal.solver import PointSourceSolver
from examples.seismic.datasets import VelocityModel, SeismogramDataset

class EikonalSolver:
    """
    A class for solving the eikonal equation to compute traveltimes from sources to receivers
    using a velocity model and seismic dataset.
    """
    
    def __init__(self, velocity_model: VelocityModel, seismogram_dataset: SeismogramDataset):
        """
        Initialize the Eikonal solver with a velocity model and seismic dataset.
        
        Args:
            velocity_model: An instance of the VelocityModel class
            seismogram_dataset: An instance of the SeismogramDataset class
        """
        self.velocity_model = velocity_model
        self.seismogram_dataset = seismogram_dataset
        
        # Validate that the velocity model covers the seismic survey area
        self._validate_model_coverage()
    
    def _validate_model_coverage(self):
        """Check that the velocity model covers the seismic survey area"""
        # Get survey coordinates from the dataset
        min_survey_x = min(self.seismogram_dataset.x_coords.min(), 
                          self.seismogram_dataset.opposite_x.min())
        max_survey_x = max(self.seismogram_dataset.x_coords.max(), 
                          self.seismogram_dataset.opposite_x.max())
        min_survey_z = min(self.seismogram_dataset.elevations.min(), 
                          self.seismogram_dataset.opposite_elev.min())
        max_survey_z = max(self.seismogram_dataset.elevations.max(), 
                          self.seismogram_dataset.opposite_elev.max())
        
        model_x = self.velocity_model.x
        model_z = self.velocity_model.z
        
        # Check coverage
        if (min_survey_x < model_x[0] or max_survey_x > model_x[-1] or
            min_survey_z < model_z[0] or max_survey_z > model_z[-1]):
            raise ValueError("Velocity model does not cover the entire survey area. "
                           "Consider padding the model.")
    
    
    def _initialize_solver(self) -> PointSourceSolver:
        """Initialize and configure the pykonal solver"""
        solver = PointSourceSolver(coord_sys="cartesian")
        
        # Set up the grid
        solver.velocity.min_coords = self.velocity_model.z[0], self.velocity_model.x[0], 0
        solver.velocity.node_intervals = self.velocity_model.dz, self.velocity_model.dx, 1
        solver.velocity.npts = len(self.velocity_model.z), len(self.velocity_model.x), 1
        
        # Assign velocity model (adding dummy dimension for 3D)
        velocity_3d = self.velocity_model.vp[..., np.newaxis]
        solver.velocity.values = velocity_3d
        
        return solver

    def solve_single(self, gather_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the eikonal equation for a single gather (source-receiver pair).
        
        Args:
            gather_idx: Index of the gather to process
            
        Returns:
            Tuple containing:
            - traveltime_field: 2D array of traveltimes from source to all points
            - traveltimes: 1D array of traveltimes from source to each receiver
        """
        # Get gather information
        _, source_x, source_z, receiver_x, receiver_z = self.seismogram_dataset[gather_idx]
        
        # Initialize and configure solver
        solver = self._initialize_solver()
        
        # Set source location (adding dummy z-coordinate for 3D)
        src_loc = np.array([source_z, source_x, 0])
        solver.src_loc = src_loc
        
        # Solve the eikonal equation
        solver.solve()
        
        # Get the traveltime field (remove dummy dimension)
        traveltime_field = solver.tt.values[..., 0]
        
        # Interpolate traveltimes at receiver locations
        traveltimes = np.zeros_like(receiver_x)
        for i, (rx, rz) in enumerate(zip(receiver_x, receiver_z)):
            # Find nearest grid point
            x_idx = np.argmin(np.abs(self.velocity_model.x - rx))
            z_idx = np.argmin(np.abs(self.velocity_model.z - rz))
            traveltimes[i] = traveltime_field[z_idx, x_idx]
        
        return traveltime_field, traveltimes
    
    def solve_all(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Solve the eikonal equation for all gathers in the dataset.
        
        Returns:
            Dictionary with gather indices as keys and tuples of 
            (traveltime_field, traveltimes) as values
        """
        results = {}
        for gather_idx in range(len(self.seismogram_dataset)):
            results[gather_idx] = self.solve_single(gather_idx)
        return results

