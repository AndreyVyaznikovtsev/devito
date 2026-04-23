from devito import Eq, Operator, TimeFunction, solve
from devito import NODE
from devito import ConditionalDimension


def src_rec(vz, p, model, geometry):
    """
    Vertical body-force source injected into vz.forward; receivers sample p.

    Injecting into vz (staggered in z) mirrors the "vertical source" convention
    used for Rayleigh-wave P-SV modelling (Pan et al. 2018, Fig. 7) and is the
    acoustic analogue of the elastic body-force approach.  Pressure p (at NODE)
    is recorded, consistent with hydrophone / pressure-sensor receivers.
    """
    s = model.grid.time_dim.spacing
    src = geometry.src
    rec = geometry.new_rec(name='rec')
    src_term = src.inject(vz.forward, expr=src * s)
    rec_term = rec.interpolate(expr=p)
    return src_term + rec_term


def ForwardOperator(model, geometry, space_order=4, save=False, **kwargs):
    """
    Forward modelling operator for first-order acoustic waves with IVF free surface.

    Velocity-pressure staggered-grid formulation (acoustic limit of Virieux 1986
    P-SV equations with mu=0):

        vx^{n+1/2} = vx^{n-1/2} - dt * b_x * dp^n/dx
        vz^{n+1/2} = vz^{n-1/2} - dt * b_z * dp^n/dz  [+ body-force source]
        p^{n+1}    = p^n         - dt * kappa * (dvx^{n+1/2}/dx + dvz^{n+1/2}/dz)

    Grid staggering:
        p     -- NODE        (x,     z    )
        vx    -- staggered x (x+h/2, z    )
        vz    -- staggered z (x,     z+h/2)
        kappa -- NODE        (zero in vacuum -> dp/dt=0 -> p stays 0; IVF)
        b_x   -- staggered x (x+h/2, z    )  IVF buoyancy avg (Pan 2018 eq. 9)
        b_z   -- staggered z (x,     z+h/2)  IVF buoyancy avg (Pan 2018 eq. 10)

    Source / receiver convention:
        Source  -- vertical body force injected into vz.forward (analogous to
                   the "vertical source for Rayleigh wave" in Pan et al. 2018).
        Receiver -- pressure p interpolated at receiver positions.

    Parameters
    ----------
    model : ModelAcousticIVF
        Physical model with kappa, b_x, b_z, damp.
    geometry : AcquisitionGeometry
        Source and receiver geometry.
    space_order : int, optional
        Spatial discretisation order.
    save : bool, optional
        If True, create sparse snapshot fields (p_save) using nsnaps and
        time_subsample kwargs.
    """
    nt = geometry.nt
    x, z = model.grid.dimensions

    p  = TimeFunction(name='p',  grid=model.grid, save=None,
                      space_order=space_order, time_order=1, staggered=NODE)
    vx = TimeFunction(name='vx', grid=model.grid, save=None,
                      space_order=space_order, time_order=1, staggered=(x,))
    vz = TimeFunction(name='vz', grid=model.grid, save=None,
                      space_order=space_order, time_order=1, staggered=(z,))

    kappa, b_x, b_z = model.kappa, model.b_x, model.b_z

    # ∂vx/∂t = -b_x * ∂p/∂x
    eq_vx = vx.dt + b_x * p.dx
    # ∂vz/∂t = -b_z * ∂p/∂z
    eq_vz = vz.dt + b_z * p.dy
    # ∂p/∂t = -kappa * (∂vx/∂x + ∂vz/∂z)  uses vx.forward, vz.forward
    eq_p  = p.dt  + kappa * (vx.forward.dx + vz.forward.dy)

    u_vx = Eq(vx.forward, model.damp * solve(eq_vx, vx.forward))
    u_vz = Eq(vz.forward, model.damp * solve(eq_vz, vz.forward))
    u_p  = Eq(p.forward,  model.damp * solve(eq_p,  p.forward))

    srcrec = src_rec(vz, p, model, geometry)
    stencils = [u_vx, u_vz, u_p] + srcrec

    if save:
        nsnaps = kwargs.pop("nsnaps", 1)
        time_subsample = kwargs.pop("time_subsample", round(nt / nsnaps))
        time_sub = ConditionalDimension("t_sub", parent=model.grid.time_dim,
                                        factor=time_subsample)
        p_save = TimeFunction(name="p_save", grid=model.grid, time_dim=time_sub,
                              save=nsnaps, staggered=NODE)
        stencils += [Eq(p_save, p)]

    return Operator(stencils, subs=model.spacing_map, name="ForwardAcousticIVF", **kwargs)
