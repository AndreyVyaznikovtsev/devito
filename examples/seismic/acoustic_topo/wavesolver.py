import numpy as np
from devito import TimeFunction
from devito.tools import memoized_meth

from examples.seismic.acoustic_topo.operators import ForwardOperator, ibm_step

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

    def __init__(self, model, geometry, space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="mask")
        self.geometry = geometry
        self.space_order = space_order
        self._kwargs = kwargs

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
            if save:
                buf = t + 1
            else:
                buf = (t + 1) % _TIME_BUF
            ibm_step(p.data[buf], model, n_iter=ibm_n_iter, tol=ibm_tol)

        return rec, p, summary
