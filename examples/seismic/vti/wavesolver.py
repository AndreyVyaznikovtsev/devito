# coding: utf-8
from devito import (
    Function,
    TimeFunction,
    warning,
    NODE,
    DevitoCheckpoint,
    CheckpointOperator,
    Revolver,
    ConditionalDimension,
)
from devito.tools import memoized_meth
from examples.seismic.vti.operators import ForwardOperator, AdjointOperator
from examples.seismic.vti.operators import JacobianOperator, JacobianAdjOperator


class VTIWaveSolver:
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Order of the spatial stencil discretisation. Defaults to 4.

    Notes
    -----
    space_order must be even and it is recommended to be a multiple of 4
    """

    def __init__(self, model, geometry, space_order=4, kernel="centered", **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry = geometry
        self.kernel = kernel

        if space_order % 2 != 0:
            raise ValueError("space_order must be even but got %s" % space_order)

        if space_order % 4 != 0:
            warning("It is recommended for space_order to be a multiple of 4" + "but got %s" % space_order)

        self.space_order = space_order

        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=False):
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
    def op_jac(self):
        """Cached operator for born runs"""
        return JacobianOperator(
            self.model,
            save=None,
            geometry=self.geometry,
            space_order=self.space_order,
            **self._kwargs,
        )

    @memoized_meth
    def op_jacadj(self, save=True):
        """Cached operator for gradient runs"""
        return JacobianAdjOperator(
            self.model,
            save=save,
            geometry=self.geometry,
            space_order=self.space_order,
            **self._kwargs,
        )

    def forward(
        self,
        src=None,
        rec=None,
        p=None,
        model=None,
        save=False,
        **kwargs,
    ):

        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        model = model or self.model

        p = p or TimeFunction(
            name="p",
            grid=self.model.grid,
            save=None,
            time_order=2,
            space_order=self.space_order,
        )

        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        # Execute operator and return wavefield and receiver data
        dt = kwargs.pop("dt", self.dt)

        if save:
            nsnaps = kwargs.pop("nsnaps", 500)
            space_subsample = kwargs.pop("space_subsample", (10, 10))
            time_subsample = kwargs.pop("time_subsample", None)
            nx, nz = model.grid.shape  # Original dimensions
            sub_nx, sub_nz = nx // space_subsample[0] + 1, nz // space_subsample[1] + 1
            # Calculate time subsampling if not provided
            if time_subsample is None:
                time_subsample = max(1, self.geometry.nt // nsnaps + 1)
            # Create subsampled dimensions and storage
            x_sub = ConditionalDimension("x_sub", parent=model.grid.dimensions[0], factor=space_subsample[0])
            z_sub = ConditionalDimension("z_sub", parent=model.grid.dimensions[1], factor=space_subsample[1])
            time_sub = ConditionalDimension("t_sub", parent=model.grid.time_dim, factor=time_subsample)

            psave = TimeFunction(
                name="psave",
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

            summary = self.op_fwd(True).apply(src=src, rec=rec, p=p, psave=psave, dt=dt, **kwargs)
            return rec, psave, summary
        else:
            summary = self.op_fwd(save).apply(src=src, rec=rec, p=p, dt=dt, **kwargs)
            return rec, p, summary

    def adjoint(self, rec, srca=None, v=None, model=None, **kwargs):
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
            sub_nx, sub_nz = nx // space_subsample[0] + 1, nz // space_subsample[1] + 1
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

    def _create_space_subsampling_dims(self, model, x_factor, z_factor):
        """Helper to create space subsampling dimensions."""
        return (
            ConditionalDimension("x_sub", parent=model.grid.dimensions[0], factor=x_factor),
            ConditionalDimension("z_sub", parent=model.grid.dimensions[1], factor=z_factor),
        )

    def jacobian(
        self,
        dm,
        src=None,
        rec=None,
        p0=None,
        du=None,
        model=None,
        save=None,
        kernel="centered",
        **kwargs,
    ):
        dt = kwargs.pop("dt", self.dt)
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefields u, v du and dv if not provided
        p0 = p0 or TimeFunction(name="p0", grid=self.model.grid, time_order=2, space_order=self.space_order)
        dp = dp or TimeFunction(name="dp", grid=self.model.grid, time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        # Execute operator and return wavefield and receiver data
        summary = self.op_jac().apply(dm=dm, p0=p0, dp=dp, src=src, rec=rec, dt=dt, **kwargs)
        return rec, p0, dp, summary

    def jacobian_adjoint(
        self,
        rec,
        p0,
        dp=None,
        dm=None,
        model=None,
        checkpointing=False,
        kernel="centered",
        **kwargs,
    ):

        dt = kwargs.pop("dt", self.dt)
        # Gradient symbol
        dm = dm or Function(name="dm", grid=self.model.grid)

        # Create the perturbation wavefields if not provided
        dp = dp or TimeFunction(name="dp", grid=self.model.grid, time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        if checkpointing:
            p0 = TimeFunction(
                name="p0",
                grid=self.model.grid,
                time_order=2,
                space_order=self.space_order,
            )
            cp = DevitoCheckpoint([p0])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(self.op_fwd(save=False), src=self.geometry.src, p=p0, dt=dt, **kwargs)

            wrap_rev = CheckpointOperator(
                self.op_jacadj(save=False),
                p0=p0,
                dp=dp,
                rec=rec,
                dm=dm,
                dt=dt,
                **kwargs,
            )

            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0] - 2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            summary = self.op_jacadj().apply(rec=rec, dm=dm, p0=p0, dp=dp, dt=dt, **kwargs)
        return dm, summary
