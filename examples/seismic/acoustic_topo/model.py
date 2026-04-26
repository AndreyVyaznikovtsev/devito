import numpy as np
from sympy import finite_diff_weights as fd_w

from devito.builtins import mmax

from examples.seismic.model import GenericModel
from examples.seismic.acoustic_topo.ibm_setup import setup_ibm

__all__ = ['ModelTopo']


class ModelTopo(GenericModel):
    """
    Variable-density acoustic model with irregular free-surface topography.

    Governing equation (2nd-order pressure, non-staggered):

        (1/v²) ∂²p/∂t² = ∇·(b ∇p) + src,   b = 1/rho

    The free surface is handled by the Immersed Boundary Method (IBM) of
    Li et al. (2020, J. Geophys. Eng. 17, 643–660).

    Absorbing boundaries (damp sponge, bcs="mask") are applied at the bottom
    and lateral edges only.  The top boundary is open — IBM sets ghost-point
    values each time step to enforce p=0 at the surface.

    Parameters
    ----------
    origin : tuple of float
        Physical origin (x0, z0) in m.
    spacing : tuple of float
        Grid spacing (dx, dz) in m.
    shape : tuple of int
        Physical grid size (nx, nz).
    space_order : int
        Spatial discretisation order.
    v : array_like
        P-wave velocity in km/s, shape (nx, nz).
    rho : array_like
        Density in g/cm³, shape (nx, nz).
    topo : array_like of float
        Physical z-coordinate of the free surface at each x column,
        shape (nx,).  Grid points strictly above this depth are ghost points.
    nbl : int, optional
        Number of absorbing boundary layers (default 20).
    bcs : str, optional
        Absorbing BC type.  Must be "mask" (default) for the IBM forward
        operator formulation ``damp * solve(pde, p.forward)``.
    dtype : data-type, optional
        Defaults to np.float32.
    """

    def __init__(self, origin, spacing, shape, space_order, v, rho, topo,
                 nbl=20, dtype=np.float32, bcs="mask", subdomains=(),
                 grid=None, topology=None, **kwargs):
        # fs=True: suppresses top PML padding and top sponge.
        # initialize_damp would divide by zero if nbl=0 for the top;
        # the fs guard in GenericModel skips that branch safely.
        super().__init__(origin, spacing, shape, space_order, nbl=nbl,
                         dtype=dtype, subdomains=subdomains, bcs=bcs,
                         grid=grid, fs=True, topology=topology)

        v = np.asarray(v, dtype=dtype)
        rho = np.asarray(rho, dtype=dtype)

        # Buoyancy b = 1/rho and slowness squared m = 1/v².
        # Both are stored as Function objects so IBM setup can write
        # ghost-point values directly into .data before the time loop.
        b_arr = np.where(rho > 0, np.float32(1) / rho, dtype(0)).astype(dtype)
        m_arr = np.where(v > 0, np.float32(1) / (v * v), dtype(0)).astype(dtype)

        self.b = self._gen_phys_param(b_arr, 'b', space_order)
        self.m = self._gen_phys_param(m_arr, 'm', space_order)

        # vp kept for CFL computation only; not used in the wave equation.
        self.vp = self._gen_phys_param(v, 'vp', space_order)
        self._physical_parameters.discard('vp')

        # Topography as physical z-coordinates (float64 for geometry accuracy).
        self.topo = np.asarray(topo, dtype=np.float64)

        self._dt = kwargs.get('dt')
        self._dt_scale = 1

        # Precompute ghost/mirror geometry and Lagrange coefficients.
        # Stores: ghost_coords, mirror_coords, coeff_indices, coeff_weights.
        setup_ibm(self)

    @property
    def _max_vp(self):
        return float(mmax(self.vp))

    @property
    def _cfl_coeff(self):
        """CFL stability coefficient for 2nd-order-in-time acoustic FD."""
        a1 = 4  # 2nd order in time
        coeffs = fd_w(2, range(-self.space_order, self.space_order + 1), 0)[-1][-1]
        return np.sqrt(a1 / float(self.grid.dim * sum(np.abs(coeffs))))

    @property
    def critical_dt(self):
        """Critical time step from the acoustic CFL condition."""
        dt = self._cfl_coeff * np.min(self.spacing) / self._max_vp
        dt = self.dtype("%.3e" % (self._dt_scale * dt))
        return self._dt if self._dt else dt

    @property
    def dt_scale(self):
        return self._dt_scale

    @dt_scale.setter
    def dt_scale(self, val):
        self._dt_scale = val

    def _build_surface_mask(self):
        """
        Boolean ghost-point mask over the full padded grid.

        Returns
        -------
        is_ghost : ndarray of bool, shape (nx_pad, nz_pad)
            is_ghost[ix, iz] is True when the physical z-coordinate of grid
            point (ix, iz) is strictly above the topographic surface, i.e.
            ``origin[1] + iz*dz < topo[ix_phys]``.

            Only physical x-columns (padded x-indices nbl … nbl+nx-1) can
            ever be True; PML x-columns are always False.
        """
        nx_pad, nz_pad = self.grid.shape
        dz = self.spacing[1]
        # With fs=True the padded grid has no top offset in z:
        # grid point iz has physical z-coordinate origin[1] + iz*dz.
        oz = self.origin[1]

        is_ghost = np.zeros((nx_pad, nz_pad), dtype=bool)

        # z-coordinates of every iz layer in the padded grid
        z_coords = oz + np.arange(nz_pad) * dz  # (nz_pad,)

        # Physical x-column ix_phys sits at padded x-index ix_phys + nbl
        for ix_phys in range(self.shape[0]):
            ix_pad = ix_phys + self.nbl
            is_ghost[ix_pad, :] = z_coords < float(self.topo[ix_phys])

        return is_ghost
