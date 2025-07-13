from devito import (Eq, Operator, Function, TimeFunction, Inc, solve)
from examples.seismic.acoustic.operators import freesurface


def second_order_stencil_vti(model, p, H, q, forward=True):
    m, damp = model.m, model.damp

    pnext = p.forward if forward else p.backward
    pdt = p.dt if forward else p.dt.T

    # Stencils
    stencil = solve(m * p.dt2 - H - q + damp * pdt, pnext)
    print("New stencil: ", stencil)
    eq = Eq(pnext, stencil, subdomain=model.grid.subdomains['physdomain'])
    stencils = [eq]
    
    # Add free surface
    if model.fs:
        stencils.append(freesurface(model, eq))
    return stencils


def vti_kernel_centered(model, p, **kwargs):
    """
    VTI finite difference kernel using single-component wavefield and frequency-domain
    anisotropy treatment.
    """
    # Model parameters
    vp = model.vp
    epsilon = model.epsilon
    delta = model.delta
    
    forward = kwargs.get('forward', True)
    # Get source if provided
    q = kwargs.get('q', 0)

    
    # Spatial derivatives
    nabla_x = p.dx2
    nabla_z = p.dz2
    
    kx = model.kx
    kz = model.kz
    
    # Anisotropy term - all real-valued
    numerator = -2*(epsilon-delta)*kx**2*kz**2
    denominator = (1+2*epsilon)*kx**4 + kz**4 + 2*(1+delta)*kx**2*kz**2
    sk = numerator / (denominator + 1e-26)  # Small constant to avoid division by zero
    
    H = vp**2 * (1+2*epsilon + sk)*nabla_x + (1 + sk)*nabla_z
    
    return second_order_stencil_vti(model, p, H, q, forward=forward)


def ForwardOperator(model, geometry, space_order=4, save=False, **kwargs):
    """
    Construct a forward modeling operator for VTI media using single-component wavefield.
    """
    dt = model.grid.time_dim.spacing
    time_order = 2
    
    # Create wavefield
    p = TimeFunction(name='p', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    
    src = geometry.src
    rec = geometry.rec
    
    # FD kernel
    stencils = vti_kernel_centered(model, p)
    
    # Source and receivers
    stencils += src.inject(field=p.forward, expr=src * dt**2 / model.m)
    stencils += rec.interpolate(expr=p)
    
    return Operator(stencils, subs=model.spacing_map, name='ForwardVTI', **kwargs)



def AdjointOperator(model, geometry, space_order=4, **kwargs):
    """
    Construct an adjoint modeling operator for VTI media.
    """
    dt = model.grid.time_dim.spacing
    time_order = 2
    
    # Create wavefield
    p = TimeFunction(name='p', grid=model.grid,
                     time_order=time_order, space_order=space_order)
    
    srca = geometry.new_src(name='srca', src_type=None)
    rec = geometry.rec
    
    # FD kernel
    stencils = vti_kernel_centered(model, p, forward=False)
    
    # Receiver injection
    stencils += rec.inject(field=p.backward, expr=rec * dt**2 / model.m)
    
    # Adjoint source
    stencils += srca.interpolate(expr=p)
    
    return Operator(stencils, subs=model.spacing_map, name='AdjointVTI', **kwargs)


def JacobianOperator(model, geometry, space_order=4, **kwargs):
    """
    Linearized Born operator for VTI media.
    
    Parameters:
        model : Model
            VTI model with physical parameters
        geometry : AcquisitionGeometry
            Source and receiver geometry
        space_order : int
            Spatial discretization order
    """
    dt = model.grid.stepping_dim.spacing
    time_order = 2

    # Create wavefields
    p0 = TimeFunction(name='p0', grid=model.grid, save=None, 
                     time_order=time_order, space_order=space_order)
    dp = TimeFunction(name="dp", grid=model.grid, save=None,
                     time_order=time_order, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    # Source and receiver terms
    src = geometry.src
    rec = geometry.rec

    # Background wave equation
    eqn1 = vti_kernel_centered(model, p0)

    # Perturbed wave equation with linearized source
    lin_src = -dm * p0.dt2
    eqn2 = vti_kernel_centered(model, dp, q=lin_src)

    # Source injection and receiver interpolation
    src_term = src.inject(field=p0.forward, expr=src * dt**2 / model.m)
    rec_term = rec.interpolate(expr=dp)

    return Operator(eqn1 + src_term + eqn2 + rec_term, 
                   subs=model.spacing_map, name='BornVTI', **kwargs)


def JacobianAdjOperator(model, geometry, space_order=4, save=True, **kwargs):
    """
    Jacobian adjoint operator for VTI media.
    
    Parameters:
        model : Model
            VTI model with physical parameters
        geometry : AcquisitionGeometry
            Source and receiver geometry
        space_order : int
            Spatial discretization order
        save : bool
            Whether to save the entire wavefield
    """
    dt = model.grid.stepping_dim.spacing
    time_order = 2

    # Wavefields and gradient
    p0 = TimeFunction(name='p0', grid=model.grid, 
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    dp = TimeFunction(name="dp", grid=model.grid, save=None,
                     time_order=time_order, space_order=space_order)
    dm = Function(name="dm", grid=model.grid)

    # Receiver term
    rec = geometry.rec

    # Adjoint wave equation
    eqn = vti_kernel_centered(model, dp, forward=False)

    # Gradient update
    dm_update = Inc(dm, - p0 * dp.dt2)

    # Receiver injection
    rec_term = rec.inject(field=dp.backward, expr=rec * dt**2 / model.m)

    return Operator(eqn + rec_term + [dm_update], 
                   subs=model.spacing_map, name='GradientVTI', **kwargs)

kernels = {('centered', 2): vti_kernel_centered}
