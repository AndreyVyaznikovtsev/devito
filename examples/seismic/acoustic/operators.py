from devito import Eq, Operator, Function, TimeFunction, Inc, solve, sign, ConditionalDimension, exp, Dimension
from devito.symbolics import retrieve_functions, INT, retrieve_derivatives
from math import floor
import numpy as np


def freesurface(model, eq):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation.

    Parameters
    ----------
    model : Model
        Physical model.
    eq : Eq
        Time-stepping stencil (time update) to mirror at the freesurface.
    """
    lhs, rhs = eq.args
    # Get vertical dimension and corresponding subdimension
    fsdomain = model.grid.subdomains["fsdomain"]
    zfs = fsdomain.dimensions[-1]
    z = zfs.parent

    # Retrieve vertical derivatives
    dzs = {d for d in retrieve_derivatives(rhs) if z in d.dims}
    # Remove inner duplicate
    dzs = dzs - {d for D in dzs for d in retrieve_derivatives(D.expr) if z in d.dims}
    dzs = {d: d._eval_at(lhs).evaluate for d in dzs}

    # Finally get functions for evaluated derivatives
    funcs = {f for f in retrieve_functions(dzs.values())}

    mapper = {}
    # Antisymmetric mirror at negative indices
    # TODO: Make a proper "mirror_indices" tool function
    for f in funcs:
        zind = f.indices[-1]
        if (zind - z).as_coeff_Mul()[0] < 0:
            s = sign(zind.subs({z: zfs, z.spacing: 1}))
            mapper.update({f: s * f.subs({zind: INT(abs(zind))})})

    # Mapper for vertical derivatives
    dzmapper = {d: v.subs(mapper) for d, v in dzs.items()}

    fs_eq = [eq.func(lhs, rhs.subs(dzmapper), subdomain=fsdomain)]
    fs_eq.append(eq.func(lhs._subs(z, 0), 0, subdomain=fsdomain))

    return fs_eq


def laplacian(field, model, kernel):
    """
    Spatial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    """
    if kernel not in ["OT2", "OT4"]:
        raise ValueError("Unrecognized kernel")
    s = model.grid.time_dim.spacing
    biharmonic = field.biharmonic(1 / model.m) if kernel == "OT4" else 0
    return field.laplace + s**2 / 12 * biharmonic


def iso_stencil(field, model, kernel, **kwargs):
    """
    Stencil for the acoustic isotropic wave-equation:
    u.dt2 - H + damp*u.dt = 0.

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    q : TimeFunction, Function or float
        Full-space/time source of the wave-equation.
    forward : bool, optional
        Whether to propagate forward (True) or backward (False) in time.
    """
    # Forward or backward
    forward = kwargs.get("forward", True)
    # Define time step to be updated
    unext = field.forward if forward else field.backward
    udt = field.dt if forward else field.dt.T
    # Get the spacial FD
    lap = laplacian(field, model, kernel)
    # Get source
    q = kwargs.get("q", 0)
    # Define PDE and update rule
    eq_time = solve(model.m * field.dt2 - lap - q + model.damp * udt, unext)

    # Time-stepping stencil.
    eqns = [Eq(unext, eq_time, subdomain=model.grid.subdomains["physdomain"])]

    # Add free surface
    if model.fs:
        eqns.append(freesurface(model, Eq(unext, eq_time)))
    return eqns


def ForwardOperator(model, geometry, space_order=4, save=False, kernel="OT2", **kwargs):
    m = model.m
    u = TimeFunction(name="u", grid=model.grid, save=None, time_order=2, space_order=space_order)
    src = geometry.src
    rec = geometry.rec

    # Create wave equation stencils
    s = model.grid.stepping_dim.spacing
    stencils = iso_stencil(u, model, kernel)
    stencils += src.inject(field=u.forward, expr=src * s**2 / m)
    stencils += rec.interpolate(expr=u)

    # Handle subsampling if requested
    if save:
        nsnaps = kwargs.pop("nsnaps", 1)
        space_subsample = kwargs.pop("space_subsample", (1, 1))
        time_subsample = kwargs.pop("time_subsample", round(geometry.nt / nsnaps))
        nx, nz = model.grid.shape  # Original dimensions
        sub_nx, sub_nz = nx//space_subsample[0] + 1, nz//space_subsample[1] + 1
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
        stencils += [Eq(usave, u)]
    return Operator(stencils, subs=model.spacing_map, name="Forward", opt=("advanced", {"gpu-fit": usave if save else None}), **kwargs)


def AdjointOperator(model, geometry, save=None, space_order=4, kernel="OT2", **kwargs):
    """
    Construct an adjoint modelling operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    v = TimeFunction(name="v", grid=model.grid, save=None, time_order=2, space_order=space_order)
    srca = geometry.new_src(name="srca", src_type=None)
    rec = geometry.rec

    # Create wave equation stencils
    s = model.grid.stepping_dim.spacing
    stencils = iso_stencil(v, model, kernel, forward=False)
    stencils += rec.inject(field=v.backward, expr=rec * s**2 / model.m)
    stencils += srca.interpolate(expr=v)

    # Handle subsampling if requested
    if save:
        nsnaps = kwargs.pop("nsnaps", 1)
        space_subsample = kwargs.pop("space_subsample", (1, 1))
        time_subsample = kwargs.pop("time_subsample", round(geometry.nt / nsnaps))
        nx, nz = model.grid.shape  # Original dimensions
        sub_nx, sub_nz = nx//space_subsample[0] + 1, nz//space_subsample[1] + 1
        x_sub = ConditionalDimension("x_sub", parent=model.grid.dimensions[0], factor=space_subsample[0])
        z_sub = ConditionalDimension("z_sub", parent=model.grid.dimensions[1], factor=space_subsample[1])
        time_sub = ConditionalDimension("t_sub", parent=model.grid.time_dim, factor=time_subsample)

        vsave = TimeFunction(
            name="vsave",
            grid=model.grid,
            dimensions=(time_sub, x_sub, z_sub),
            shape=(nsnaps, sub_nx, sub_nz),
            time_dim=time_sub,
            save=nsnaps,
        )
        stencils += [Eq(vsave, v)]

    return Operator(stencils, subs=model.spacing_map, name="Adjoint", opt=("advanced", {"gpu-fit": vsave if save else None}), **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True, kernel="OT2", **kwargs):
    """
    Construct a gradient operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Option to store the entire (unrolled) wavefield.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m = model.m

    # Gradient symbol and wavefield symbols
    grad = Function(name="grad", grid=model.grid)
    u = TimeFunction(
        name="u",
        grid=model.grid,
        save=geometry.nt if save else None,
        time_order=2,
        space_order=space_order,
    )
    v = TimeFunction(name="v", grid=model.grid, save=None, time_order=2, space_order=space_order)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, forward=False)

    if kernel == "OT2":
        gradient_update = Inc(grad, -u * v.dt2)
    elif kernel == "OT4":
        gradient_update = Inc(grad, -u * v.dt2 - s**2 / 12.0 * u.biharmonic(m ** (-2)) * v)
    # Add expression for receiver injection
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Substitute spacing terms to reduce flops
    return Operator(
        eqn + receivers + [gradient_update],
        subs=model.spacing_map,
        name="Gradient",
        **kwargs,
    )


def BornOperator(model, geometry, space_order=4, kernel="OT2", **kwargs):
    """
    Construct an Linearized Born operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m = model.m

    # Create source and receiver symbols
    src = geometry.src
    rec = geometry.rec

    # Create wavefields and a dm field
    u = TimeFunction(name="u", grid=model.grid, save=None, time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None, time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, model, kernel)
    eqn2 = iso_stencil(U, model, kernel, q=-dm * u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map, name="Born", **kwargs)


def GreenOperator(model, geometry, nfreqs, space_order=4, save=False, kernel="OT2", **kwargs):
    m = model.m

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name="u", grid=model.grid, save=geometry.nt if save else None, time_order=2, space_order=space_order)
    src = geometry.src
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, model, kernel)

    src_term = src.inject(field=u.forward, expr=src * s**2 / m)
    rec_term = rec.interpolate(expr=u)

    f = Dimension(name="f")
    freqs = Function(name="frequencies", dimensions=(f,), shape=(nfreqs,), dtype=np.float32)
    freq_modes = Function(
        name="freq_modes",
        grid=model.grid,
        space_order=0,
        dtype=np.complex64,
        dimensions=(f, *model.grid.dimensions),
        shape=(nfreqs, *model.grid.shape),
    )
    omega = 2 * np.pi * freqs
    basis = exp(-1j * omega * model.grid.time_dim * model.grid.time_dim.spacing)
    dfts = [Inc(freq_modes, basis * u)]

    return Operator(eqn + src_term + rec_term + dfts, subs=model.spacing_map, name="Forward", **kwargs)
