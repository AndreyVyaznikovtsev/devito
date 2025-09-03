from devito import Function, TimeFunction, ConditionalDimension, DevitoCheckpoint, CheckpointOperator, Revolver, Grid, Dimension
from devito.tools import memoized_meth
from examples.seismic.acoustic.operators import (
    ForwardOperator,
    ForwardOperatorUnt,
    AdjointOperator,
    GradientOperator,
    BornOperator,
    GreenOperator,
)
import numpy as np


class AcousticWaveSolver:
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Physical model with domain parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    kernel : str, optional
        Type of discretization, centered or shifted.
    space_order: int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    """

    def __init__(self, model, geometry, kernel="OT2", space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry = geometry

        assert self.model.grid == geometry.grid

        self.space_order = space_order
        self.kernel = kernel

        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        if self.kernel == "OT4":
            return self.model.dtype(1.73 * self.model.critical_dt)
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(
            self.model,
            save=save,
            geometry=self.geometry,
            kernel=self.kernel,
            space_order=self.space_order,
            **self._kwargs,
        )

    @memoized_meth
    def op_fwd_unt(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperatorUnt(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_adj(self, save=None):
        """Cached operator for adjoint runs"""
        return AdjointOperator(
            self.model,
            save=save,
            geometry=self.geometry,
            kernel=self.kernel,
            space_order=self.space_order,
            **self._kwargs,
        )

    @memoized_meth
    def op_grad(self, save=True):
        """Cached operator for gradient runs"""
        return GradientOperator(
            self.model,
            save=save,
            geometry=self.geometry,
            kernel=self.kernel,
            space_order=self.space_order,
            **self._kwargs,
        )

    @memoized_meth
    def op_born(self):
        """Cached operator for born runs"""
        return BornOperator(
            self.model,
            save=None,
            geometry=self.geometry,
            kernel=self.kernel,
            space_order=self.space_order,
            **self._kwargs,
        )

    @memoized_meth
    def op_green(self, save=False, nfreqs=1):
        """Cached operator for green runs"""
        return GreenOperator(
            self.model,
            save=save,
            geometry=self.geometry,
            nfreqs=nfreqs,
            kernel=self.kernel,
            space_order=self.space_order,
            **self._kwargs,
        )

    def forward(self, src=None, rec=None, u=None, model=None, save=None, **kwargs):
        src = src or self.geometry.src
        rec = rec or self.geometry.rec
        u = u or TimeFunction(name="u", grid=self.model.grid, time_order=2, save=None, space_order=self.space_order)
        model = model or self.model

        # Update physical parameters from model
        kwargs.update(model.physical_params(**kwargs))

        # Extract common parameters
        dt = kwargs.pop("dt", self.dt)

        if save:
            # Configure subsampling
            nsnaps = kwargs.pop("nsnaps", 500)
            space_subsample = kwargs.pop("space_subsample", (10, 10))
            time_subsample = kwargs.pop("time_subsample", None)
            nx, nz = model.grid.shape  # Original dimensions
            sub_nx, sub_nz = nx//space_subsample[0] + 1, nz//space_subsample[1] + 1
            # Calculate time subsampling if not provided
            if time_subsample is None:
                time_subsample = max(1, self.geometry.nt // nsnaps + 1)
            # Create subsampled dimensions and storage
            x_sub = ConditionalDimension("x_sub", parent=model.grid.dimensions[0], factor=space_subsample[0])
            z_sub = ConditionalDimension("z_sub", parent=model.grid.dimensions[1], factor=space_subsample[1])
            time_sub = ConditionalDimension("t_sub", parent=model.grid.time_dim, factor=time_subsample)

            usave = TimeFunction(
                name="usave",
                grid=model.grid,
                dimensions=(time_sub, x_sub, z_sub),
                shape=(nsnaps, sub_nx, sub_nz),
                time_dim=time_sub,
                save=nsnaps,
            )

            # Store subsampling parameters for reference
            self._kwargs.update(
                {
                    "nsnaps": nsnaps,
                    "time_subsample": time_subsample,
                    "space_subsample": space_subsample,
                }
            )

            summary = self.op_fwd(True).apply(src=src, rec=rec, u=u, usave=usave, dt=dt, **kwargs)
            return rec, usave, summary
        else:
            summary = self.op_fwd().apply(src=src, rec=rec, u=u, dt=dt, **kwargs)
            return rec, u, summary

    def forward_untouched(self, src=None, rec=None, u=None, model=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            Stores the computed wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.

        Returns
        -------
        Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd_unt(save).apply(src=src, rec=rec, u=u,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)

        return rec, u, summary


    def adjoint(self, rec, srca=None, v=None, model=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        rec : SparseTimeFunction or array-like
            The receiver data. Please note that
            these act as the source term in the adjoint run.
        srca : SparseTimeFunction or array-like
            The resulting data for the interpolated at the
            original source location.
        v: TimeFunction, optional
            The computed wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        srca = srca or self.geometry.new_src(name="srca", src_type=None)
        v = v or TimeFunction(name="v", grid=self.model.grid, time_order=2, space_order=self.space_order)
        model = model or self.model

        # Update physical parameters from model
        kwargs.update(model.physical_params(**kwargs))

        # Extract subsampling parameters with defaults
        save = kwargs.pop("save", False)
        dt = kwargs.pop("dt", self.dt)

        if save:
            # Configure subsampling
            nsnaps = kwargs.pop("nsnaps", 500)
            space_subsample = kwargs.pop("space_subsample", (10, 10))
            time_subsample = kwargs.pop("time_subsample", None)

            # Calculate time subsampling if not provided
            if time_subsample is None:
                time_subsample = max(1, self.geometry.nt // nsnaps + 1)
            # Create subsampled dimensions and storage
            
            nx, nz = model.grid.shape  # Original dimensions
            sub_nx, sub_nz = nx//space_subsample[0] + 1, nz//space_subsample[1] + 1
            x_sub, z_sub = self._create_space_subsampling_dims(model, *space_subsample)

            time_sub = ConditionalDimension("t_sub", parent=model.grid.time_dim, factor=time_subsample)

            vsave = TimeFunction(
                name="vsave",
                grid=model.grid,
                dimensions=(time_sub, x_sub, z_sub),
                shape=(nsnaps, sub_nx, sub_nz),
                time_dim=time_sub,
                save=nsnaps,
            )

            # Store subsampling parameters for reference
            self._kwargs.update(
                {
                    "nsnaps": nsnaps,
                    "time_subsample": time_subsample,
                    "space_subsample": space_subsample,
                }
            )

            summary = self.op_adj(True).apply(srca=srca, rec=rec, v=v, vsave=vsave, dt=dt, **kwargs)
            return srca, vsave, summary
        else:
            summary = self.op_adj().apply(srca=srca, rec=rec, v=v, dt=dt, **kwargs)
            return srca, v, summary

    def jacobian_adjoint(
        self,
        rec,
        u,
        src=None,
        v=None,
        grad=None,
        model=None,
        checkpointing=False,
        **kwargs,
    ):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        u : TimeFunction
            Full wavefield `u` (created with save=True).
        v : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the gradient field.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Gradient field and performance summary.
        """
        dt = kwargs.pop("dt", self.dt)
        # Gradient symbol
        grad = grad or Function(name="grad", grid=self.model.grid)

        # Create the forward wavefield
        v = v or TimeFunction(name="v", grid=self.model.grid, time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        if checkpointing:
            u = TimeFunction(
                name="u",
                grid=self.model.grid,
                time_order=2,
                space_order=self.space_order,
            )
            cp = DevitoCheckpoint([u])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(
                self.op_fwd(save=False),
                src=src or self.geometry.src,
                u=u,
                dt=dt,
                **kwargs,
            )
            wrap_rev = CheckpointOperator(self.op_grad(save=False), u=u, v=v, rec=rec, dt=dt, grad=grad, **kwargs)

            # Run forward
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0] - 2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            summary = self.op_grad().apply(rec=rec, grad=grad, v=v, u=u, dt=dt, **kwargs)
        return grad, summary

    def jacobian(self, dmin, src=None, rec=None, u=None, U=None, model=None, **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            The forward wavefield.
        U : TimeFunction, optional
            The linearized wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefields u and U if not provided
        u = u or TimeFunction(name="u", grid=self.model.grid, time_order=2, space_order=self.space_order)
        U = U or TimeFunction(name="U", grid=self.model.grid, time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_born().apply(dm=dmin, u=u, U=U, src=src, rec=rec, dt=kwargs.pop("dt", self.dt), **kwargs)
        return rec, u, U, summary

    def green(self, frequencies, src=None, rec=None, u=None, model=None, save=None, **kwargs):

        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name="u", grid=self.model.grid, save=self.geometry.nt if save else None, time_order=2, space_order=self.space_order)

        model = model or self.model

        f = Dimension(name="f")
        nfreq = len(frequencies)
        freqs = Function(name="frequencies", dimensions=(f,), shape=(nfreq,), dtype=np.float32)
        freqs.data[:] = frequencies

        freq_modes = Function(
            name="freq_modes",
            grid=model.grid,
            space_order=0,
            dtype=np.complex64,
            dimensions=(f, *model.grid.dimensions),
            shape=(nfreq, *model.grid.shape),
        )

        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        kwargs.update({"frequencies": freqs, "freq_modes": freq_modes})

        # Execute operator and return wavefield and receiver data
        summary = self.op_green(save, nfreq).apply(src=src, rec=rec, u=u, dt=kwargs.pop("dt", self.dt), **kwargs)

        return rec, u, freq_modes, summary

    def _create_space_subsampling_dims(self, model, x_factor, z_factor):
        """Helper to create space subsampling dimensions."""
        return (
            ConditionalDimension("x_sub", parent=model.grid.dimensions[0], factor=x_factor),
            ConditionalDimension("z_sub", parent=model.grid.dimensions[1], factor=z_factor),
        )

    # Backward compatibility
    born = jacobian
    gradient = jacobian_adjoint
