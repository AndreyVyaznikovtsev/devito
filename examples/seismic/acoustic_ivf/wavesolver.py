from devito.tools import memoized_meth
from devito import TimeFunction, NODE

from examples.seismic.acoustic_ivf.operators import ForwardOperator


class AcousticIVFWaveSolver:
    """
    Solver object for first-order acoustic wave forward modelling with IVF
    free-surface support.

    Parameters
    ----------
    model : ModelAcousticIVF
        Physical model with domain parameters (kappa, b_x, b_z).
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

    def forward(self, src=None, rec=None, p=None, vx=None, vz=None,
                model=None, save=None, **kwargs):
        """
        Run the acoustic IVF forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction, optional
            Source wavelet (pressure source).
        rec : SparseTimeFunction, optional
            Receiver array (samples pressure p).
        p : TimeFunction, optional
            Pressure field (staggered=NODE).
        vx : TimeFunction, optional
            Horizontal velocity staggered in x.
        vz : TimeFunction, optional
            Vertical velocity staggered in z.
        model : ModelAcousticIVF, optional
            Override the model stored on the solver.
        save : bool, optional
            If True, allocate p/vx/vz with full time history (save=src.nt).

        Returns
        -------
        rec, p, vx, vz, summary
        """
        src = src or self.geometry.src
        rec = rec or self.geometry.new_rec(name='rec')

        save_t = src.nt if save else None
        x, z = self.model.grid.dimensions

        p  = p  or TimeFunction(name='p',  grid=self.model.grid, save=save_t,
                                space_order=self.space_order, time_order=1,
                                staggered=NODE)
        vx = vx or TimeFunction(name='vx', grid=self.model.grid, save=save_t,
                                space_order=self.space_order, time_order=1,
                                staggered=(x,))
        vz = vz or TimeFunction(name='vz', grid=self.model.grid, save=save_t,
                                space_order=self.space_order, time_order=1,
                                staggered=(z,))

        kwargs.update({'p': p, 'vx': vx, 'vz': vz})
        model = model or self.model
        kwargs.update(model.physical_params(**kwargs))

        summary = self.op_fwd(save).apply(src=src, rec=rec,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, p, vx, vz, summary
