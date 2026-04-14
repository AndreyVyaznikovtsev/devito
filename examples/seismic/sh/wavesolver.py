from devito.tools import memoized_meth
from devito import TimeFunction, NODE

from examples.seismic.sh.operators import ForwardOperator


class SHWaveSolver:
    """
    Solver object for SH (Shear Horizontal) wave forward modelling.

    Parameters
    ----------
    model : ModelSH
        Physical model with domain parameters (mu, b).
    geometry : AcquisitionGeometry
        Source and receiver geometry.
    space_order : int, optional
        Spatial discretisation order. Defaults to 4.
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
    def op_fwd(self, save=None):
        """Cached forward operator."""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec=None, v=None, tau_xy=None, tau_zy=None,
                model=None, save=None, **kwargs):
        """
        Run the SH forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction, optional
            Source wavelet.
        rec : SparseTimeFunction, optional
            Receiver array (samples the velocity field v).
        v : TimeFunction, optional
            Particle velocity field (staggered=NODE).
        tau_xy : TimeFunction, optional
            Stress component staggered in x.
        tau_zy : TimeFunction, optional
            Stress component staggered in z.
        model : ModelSH, optional
            Override the model stored on the solver.
        save : bool, optional
            Save the full wavefield history.

        Returns
        -------
        rec, v, tau_xy, tau_zy, summary
        """
        src = src or self.geometry.src
        rec = rec or self.geometry.new_rec(name='rec')

        save_t = src.nt if save else None
        x, z = self.model.grid.dimensions

        v = v or TimeFunction(name='v', grid=self.model.grid, save=save_t,
                              space_order=self.space_order, time_order=1,
                              staggered=NODE)
        tau_xy = tau_xy or TimeFunction(name='tau_xy', grid=self.model.grid,
                                        save=save_t, space_order=self.space_order,
                                        time_order=1, staggered=(x,))
        tau_zy = tau_zy or TimeFunction(name='tau_zy', grid=self.model.grid,
                                        save=save_t, space_order=self.space_order,
                                        time_order=1, staggered=(z,))

        kwargs.update({'v': v, 'tau_xy': tau_xy, 'tau_zy': tau_zy})

        model = model or self.model
        kwargs.update(model.physical_params(**kwargs))

        summary = self.op_fwd(save).apply(src=src, rec=rec,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, v, tau_xy, tau_zy, summary
