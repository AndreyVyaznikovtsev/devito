import segyio
from segyio import TraceField
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


class SeismogramDataset:
    def __init__(self, sgy_path: str, sort_key: str = "sou"):
        """
        Initialize the dataset with a SEG-Y file and sorting key.

        Args:
            sgy_path: Path to the SEG-Y file
            sort_key: 'sou' to sort by source or 'rec' to sort by receiver ('sou' by default)
        """
        if sort_key not in ["sou", "rec"]:
            raise ValueError("sort_key must be either 'sou' or 'rec'")

        self.sgy_path = sgy_path
        self.sort_key = sort_key

        # Open the file and read headers
        with segyio.open(sgy_path, ignore_geometry=True) as f:
            # Get scalars from the first trace (assuming they're consistent across all traces)
            source_group_scalar = f.header[0][TraceField.SourceGroupScalar]
            elevation_scalar = f.header[0][TraceField.ElevationScalar]

            # Apply scalars (convert to 1 if 0 as per SEG-Y standard)
            source_group_scalar = source_group_scalar if source_group_scalar != 0 else 1
            elevation_scalar = elevation_scalar if elevation_scalar != 0 else 1

            # Read all headers with scaling applied
            if sort_key == "sou":
                self.elevations = np.array([f.header[trace][TraceField.SourceSurfaceElevation] / elevation_scalar for trace in range(f.tracecount)])
                self.x_coords = np.array([f.header[trace][TraceField.SourceX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_x = np.array([f.header[trace][TraceField.GroupX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_elev = np.array([f.header[trace][TraceField.ReceiverGroupElevation] / elevation_scalar for trace in range(f.tracecount)])
            else:  # 'rec'
                self.elevations = np.array([f.header[trace][TraceField.ReceiverGroupElevation] / elevation_scalar for trace in range(f.tracecount)])
                self.x_coords = np.array([f.header[trace][TraceField.GroupX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_x = np.array([f.header[trace][TraceField.SourceX] / source_group_scalar for trace in range(f.tracecount)])
                self.opposite_elev = np.array([f.header[trace][TraceField.SourceSurfaceElevation] / elevation_scalar for trace in range(f.tracecount)])

            # Get unique elevations and their indices
            self.unique_elevations = np.unique(self.elevations)

            # For each unique elevation, store the trace indices that belong to it
            self.elev_groups = {}
            for i, elev in enumerate(self.unique_elevations):
                mask = self.elevations == elev
                self.elev_groups[i] = {
                    "trace_indices": np.where(mask)[0],
                    "x_coord": self.x_coords[mask][0],  # Take first value (all should be same for this group)
                    "elev": elev,
                    "opposite_x": self.opposite_x[mask],
                    "opposite_elev": self.opposite_elev[mask],
                }

            # Store samples and dt for later use
            self.samples = f.samples
            self.n_samples = len(self.samples)
            self._dt = segyio.tools.dt(f) / 1000  # Convert microseconds to seconds

    @property
    def dt(self) -> float:
        """Return the sample interval in seconds"""
        return self._dt

    def __len__(self) -> int:
        """Return the number of unique source/receiver elevations"""
        return len(self.unique_elevations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        """
        Get a gather by index.

        Returns:
            Tuple containing:
            - 2D numpy array of the gather (traces x samples)
            - X coordinate of the gather (source or receiver depending on sort_key)
            - Elevation of the gather
            - 1D array of opposite X coordinates
            - 1D array of opposite elevations
        """
        group = self.elev_groups[idx]
        trace_indices = group["trace_indices"]

        # Sort traces by opposite elevation
        sort_order = np.argsort(group["opposite_elev"])
        sorted_trace_indices = trace_indices[sort_order]

        # Read the traces
        with segyio.open(self.sgy_path, ignore_geometry=True) as f:
            gather = np.stack([f.trace[i] for i in sorted_trace_indices])

        return (
            gather,  # 2D array (traces x samples)
            group["x_coord"],  # X coordinate
            group["elev"],  # Elevation
            group["opposite_x"][sort_order],  # Sorted opposite X coordinates
            group["opposite_elev"][sort_order],  # Sorted opposite elevations
        )

    def plot_gather(self, idx: int, quantile: float = 0.98, figsize=(10, 6)):
        """
        Plot the gather with automatic scaling based on data quantile.

        Args:
            idx: Index of the gather to plot
            quantile: Quantile value for scaling (0.98 means 98% of data will be within range)
            figsize: Figure size
        """
        gather, _, _, opposite_elev, _ = self.__getitem__(idx)

        # Calculate vmin/vmax based on quantile
        vmax = np.quantile(np.abs(gather), quantile)
        vmin = -vmax

        # Create figure
        plt.figure(figsize=figsize)

        # Plot with proper extent
        extent = [
            opposite_elev[0],
            opposite_elev[-1],  # x-axis: first and last opposite elevation
            self.samples[-1],
            self.samples[0],  # y-axis: time (reversed for seismic display)
        ]

        plt.imshow(
            gather.T,  # Transpose to have traces on x-axis and time on y-axis
            aspect="auto",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )

        plt.xlabel("Opposite Elevation" if self.sort_key == "sou" else "Source Elevation")
        plt.ylabel("Time (ms)")
        title_type = "Source" if self.sort_key == "sou" else "Receiver"
        plt.title(f"{title_type} gather #{idx}")
        plt.colorbar(label="Amplitude")
        plt.show()
