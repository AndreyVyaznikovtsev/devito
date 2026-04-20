from devito import Eq, Operator, TimeFunction, solve
from devito import NODE
from devito import ConditionalDimension


def src_rec(v, model, geometry):
    """
    Source injection into v and receiver interpolation from v.
    """
    s = model.grid.time_dim.spacing
    src = geometry.src
    rec = geometry.new_rec(name='rec')

    src_term = src.inject(v.forward, expr=src * s)
    rec_term = rec.interpolate(expr=v)

    return src_term + rec_term


def ForwardOperator(model, geometry, space_order=4, save=False, **kwargs):
    """
    Forward modelling operator for SH (Shear Horizontal) waves.

    Staggered velocity-stress formulation (Virieux 1984):

        v^{n+1/2}    = v^{n-1/2}  + dt * b    * (d/dx tau_xy  + d/dz tau_zy)
        tau_xy^{n+1} = tau_xy^{n} + dt * mu_x * d/dx v^{n+1/2}
        tau_zy^{n+1} = tau_zy^{n} + dt * mu_z * d/dz v^{n+1/2}

    Grid staggering:
        v      -- NODE        (x,     z    )
        b      -- NODE        (x,     z    )
        tau_xy -- staggered x (x+h/2, z    )
        tau_zy -- staggered z (x,     z+h/2)
        mu_x   -- staggered x (x+h/2, z    )  pre-averaged on ModelSH
        mu_z   -- staggered z (x,     z+h/2)  pre-averaged on ModelSH

    mu_x and mu_z are 2-point harmonic averages computed by ModelSH.  With IVF
    (Pan 2018, topo != None), any staggered point that borders a vacuum cell
    gets mu_x=0 or mu_z=0, which drives the corresponding stress to zero and
    automatically satisfies the traction-free boundary condition.

    Parameters
    ----------
    model : ModelSH
        Physical model with mu_x, mu_z, and b.
    geometry : AcquisitionGeometry
        Source and receiver geometry.
    space_order : int, optional
        Spatial discretisation order.
    save : bool, optional
        Save the full wavefield history.
    """
    nt = geometry.nt

    x, z = model.grid.dimensions
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     space_order=space_order, time_order=1, staggered=NODE)
    tau_xy = TimeFunction(name='tau_xy', grid=model.grid, save=None,
                          space_order=space_order, time_order=1, staggered=(x,))
    tau_zy = TimeFunction(name='tau_zy', grid=model.grid, save=None,
                          space_order=space_order, time_order=1, staggered=(z,))

    mu_x, mu_z, b = model.mu_x, model.mu_z, model.b

    eq_v = v.dt - b * (tau_xy.dx + tau_zy.dy)
    # mu_x is already at (x+h/2, z) — same position as tau_xy; no re-averaging.
    eq_tau_xy = tau_xy.dt - mu_x * v.forward.dx
    # mu_z is already at (x, z+h/2) — same position as tau_zy; no re-averaging.
    eq_tau_zy = tau_zy.dt - mu_z * v.forward.dy

    u_v = Eq(v.forward, model.damp * solve(eq_v, v.forward))
    u_tau_xy = Eq(tau_xy.forward, model.damp * solve(eq_tau_xy, tau_xy.forward))
    u_tau_zy = Eq(tau_zy.forward, model.damp * solve(eq_tau_zy, tau_zy.forward))

    srcrec = src_rec(v, model, geometry)
    stencils = [u_v, u_tau_xy, u_tau_zy] + srcrec
    if save:
        nsnaps = kwargs.pop("nsnaps", 1)
        time_subsample = kwargs.pop("time_subsample", round(nt / nsnaps))
        time_sub = ConditionalDimension("t_sub", parent=model.grid.time_dim, factor=time_subsample)
        v_save = TimeFunction(
            name="v_save",
            grid=model.grid,
            time_dim=time_sub,
            save=nsnaps,
            staggered=NODE,
        )
        tau_xy_save = TimeFunction(
            name="tau_xy_save",
            grid=model.grid,
            time_dim=time_sub,
            save=nsnaps,
            staggered=(x,),
        )
        tau_zy_save = TimeFunction(
            name="tau_zy_save",
            grid=model.grid,
            time_dim=time_sub,
            save=nsnaps,
            staggered=(z,),
        )
        stencils += [Eq(v_save, v)]
        stencils += [Eq(tau_xy_save, tau_xy)]
        stencils += [Eq(tau_zy_save, tau_zy)]
    
    return Operator(stencils, subs=model.spacing_map, name="ForwardSH", **kwargs)
