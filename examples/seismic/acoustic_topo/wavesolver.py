import numpy as np
from devito import Function, TimeFunction
from devito.tools import memoized_meth

from examples.seismic.acoustic_topo.operators import (
    ForwardOperator, GradientOperator, ibm_step, ibm_step_numba
)

__all__ = ['AcousticTopoSolver']

# Rolling-buffer size for TimeFunction(time_order=2, save=None).
_TIME_BUF = 3


class AcousticTopoSolver:
    """
    Forward modelling solver for variable-density acoustic waves with an
    irregular free surface handled by the Immersed Boundary Method.

    Time stepping is manual: each FD step is followed by an IBM ghost-point
    correction before the next step reads the updated values.

    Parameters
    ----------
    model : ModelTopo
        Physical model (v, rho, topo, damp, b, m, and IBM geometry arrays).
    geometry : AcquisitionGeometry
        Source and receiver positions and time axis.
    space_order : int, optional
        Spatial discretisation order (default 4).
    """

    def __init__(self, model, geometry, space_order=4, use_numba=False, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="mask")
        self.geometry = geometry
        self.space_order = space_order
        self._use_numba = use_numba
        self._kwargs = kwargs

    def _call_ibm(self, data_buf, model, n_iter, tol):
        """Dispatch IBM correction to NumPy or Numba implementation."""
        if self._use_numba:
            ibm_step_numba(np.asarray(data_buf), model.ghost_coords,
                           model.coeff_indices, model.coeff_weights,
                           n_iter=n_iter, tol=tol)
        else:
            ibm_step(data_buf, model, n_iter=n_iter, tol=tol)

    @property
    def dt(self):
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=False):
        """Cached compiled forward operator (FD only, no IBM)."""
        return ForwardOperator(self.model, geometry=self.geometry,
                               space_order=self.space_order, save=save,
                               **self._kwargs)

    def forward(self, src=None, rec=None, p=None, model=None, save=False,
                ibm_n_iter=20, ibm_tol=1e-6, **kwargs):
        """
        Run the IBM acoustic forward propagation.

        For each time step t = 0 … nt−2:
          1. FD pressure update: writes p(t+1) into the rolling buffer.
          2. IBM ghost correction on p(t+1): enforces p=0 at the surface.

        The receiver `rec` samples `p(t)` inside the FD operator, so it
        records the wavefield *before* the FD advance — consistent with the
        standard Devito acoustic convention.

        Parameters
        ----------
        src : SparseTimeFunction, optional
        rec : SparseTimeFunction, optional
        p : TimeFunction, optional
            Pre-allocated pressure field.  Created internally if None.
        model : ModelTopo, optional
            Override the stored model.
        save : bool, optional
            Allocate the full time history of p (use for imaging / snaps).
        ibm_n_iter : int
            Maximum IBM iterations per time step (default 20).
        ibm_tol : float
            IBM convergence tolerance (default 1e-6).

        Note
        ----
        Unlike the standard acoustic solver, ``ModelTopo`` stores ``m``
        (slowness-squared) and ``b`` (buoyancy) as separate Function objects.
        Passing ``vp=...`` is **not supported** — use ``model=`` or update
        ``model.m.data`` directly if you need to change velocity.

        Returns
        -------
        rec : SparseTimeFunction — recorded pressure traces.
        p   : TimeFunction — pressure field.
        summary : object — performance summary from the last FD step.
        """
        src = src or self.geometry.src
        rec = rec or self.geometry.rec
        model = model or self.model
        nt = self.geometry.nt

        # ModelTopo uses m (not vp) in the PDE; strip vp from kwargs so it
        # doesn't reach the compiled operator as an unrecognized argument.
        kwargs.pop('vp', None)

        if p is None:
            p = TimeFunction(name='p', grid=model.grid,
                             save=nt if save else None,
                             time_order=2, space_order=self.space_order)

        dt = kwargs.pop('dt', self.dt)
        kwargs.update(model.physical_params(**kwargs))

        op = self.op_fwd(save)
        summary = None

        # With save=True the full-history buffer has no p[-1], so time_m must
        # start at 1 (reads p[0] as the valid "backward" initial condition).
        t_start = 1 if save else 0
        for t in range(t_start, nt - 1):
            summary = op.apply(time_m=t, time_M=t, src=src, rec=rec, p=p,
                               dt=dt, **kwargs)
            # Select the buffer index that holds the freshly computed p(t+1).
            # Rolling buffer: buf = (t+1) % 3.
            # Full time history: buf = t+1 directly.
            buf = t + 1 if save else (t + 1) % _TIME_BUF
            self._call_ibm(p.data[buf], model, ibm_n_iter, ibm_tol)

        return rec, p, summary

    @memoized_meth
    def op_grad(self, save=True):
        """Cached compiled gradient operator (single-step)."""
        return GradientOperator(self.model, geometry=self.geometry,
                                space_order=self.space_order, save=save,
                                **self._kwargs)

    def gradient(self, rec, p, v=None, grad=None, model=None,
                 ibm_n_iter=20, ibm_tol=1e-6, **kwargs):
        """
        Compute the gradient of the misfit functional with respect to
        slowness-squared ``m`` via the adjoint-state method.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver residuals (observed - predicted data).
        p : TimeFunction
            Forward wavefield saved at all time steps (save=True).
        v : TimeFunction, optional
            Adjoint wavefield (created internally if None).
        grad : Function, optional
            Gradient field (created internally if None).
        model : ModelTopo, optional
            Override the stored model.
        ibm_n_iter : int
            Maximum IBM iterations per time step (default 20).
        ibm_tol : float
            IBM convergence tolerance (default 1e-6).

        Note
        ----
        ``ModelTopo`` uses ``m`` (slowness-squared) and ``b`` (buoyancy) in
        the PDE, not ``vp``.  Pass ``model=`` or update ``model.m.data``
        directly if you need to change velocity.

        Returns
        -------
        grad : Function
        summary : object
        """
        model = model or self.model
        nt = self.geometry.nt

        # ModelTopo uses m (not vp) in the PDE; strip vp from kwargs so it
        # doesn't reach the compiled operator as an unrecognized argument.
        kwargs.pop('vp', None)

        grad = grad or Function(name='grad', grid=model.grid)
        v = v or TimeFunction(name='v', grid=model.grid,
                              time_order=2, space_order=self.space_order)

        dt = kwargs.pop('dt', self.dt)
        kwargs.update(model.physical_params(**kwargs))

        op = self.op_grad(save=True)

        for t in range(nt - 2, 0, -1):
            op.apply(time_m=t, time_M=t, rec=rec, v=v, u=p, grad=grad,
                     dt=dt, **kwargs)
            buf = (t - 1) % 3
            self._call_ibm(v.data[buf], model, ibm_n_iter, ibm_tol)

        return grad, None

    jacobian_adjoint = gradient
