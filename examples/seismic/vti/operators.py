from devito import (Eq, Operator, Function, TimeFunction, Inc, solve, ConditionalDimension)
from examples.seismic.acoustic.operators import freesurface
import numpy as np

def second_order_stencil_vti(model, p, H, q, forward=True):
    vp, damp = model.vp, model.damp

    pnext = p.forward if forward else p.backward
    pdt = p.dt if forward else p.dt.T

    # Stencils
    stencil = solve(p.dt2 - H - q + vp**2 * damp * pdt, pnext)
    eq = Eq(pnext, stencil, subdomain=model.grid.subdomains['physdomain'])
    stencils = [eq]
    
    # Add free surface
    if model.fs:
        stencils.append(freesurface(model, Eq(pnext, stencil)))
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

    px = p.dx
    pz = p.dy
    px2 = px**2
    pz2 = pz**2
    px4 = px**4
    pz4 = pz**4

    dxx = p.dx2
    dzz = p.dy2
  
    numerator = -2 * (epsilon - delta) * px2 * pz2
    denominator = (1 + 2 * epsilon) * px4 + pz4 + 2 * (1 + delta) * px2 * pz2
    sn = numerator / (denominator + 1e-26)  # Small constant to avoid division by zero
    H = vp**2 * ((1+2*epsilon + sn)*dxx + (1 + sn)*dzz) 

    return second_order_stencil_vti(model, p, H, q, forward=forward)

def habc_higdon_stencils(model, p, forward=True):
    """
    Generate HABC stencils using Higdon boundary condition (3rd method)
    """
    pnext, pprev = p.forward, p.backward if forward else p.backward, p.forward
    # Get model parameters
    v = model.vp
    weightsx = model.weightsx
    weightsz = model.weightsz
    dt = model.grid.time_dim.spacing
    hx, hz = model.spacing
    
    # Define angles for Higdon (order 2)
    alpha1 = 0.0
    alpha2 = np.pi/4
    
    # Coefficients for Higdon boundary condition
    a1, b1 = 0.5, 0.5  # Coefficients for first angle
    a2, b2 = 0.5, 0.5  # Coefficients for second angle
    
    # Get dimensions
    x, z = model.grid.dimensions
    t = model.grid.time_dim
    
    # Higdon boundary condition implementation
    stencils = []
    
    # Left boundary (d1)
    gamma111 = np.cos(alpha1)*(1-a1)*(1/dt)
    gamma121 = np.cos(alpha1)*(a1)*(1/dt)
    gamma131 = np.cos(alpha1)*(1-b1)*(1/hx)*v
    gamma141 = np.cos(alpha1)*(b1)*(1/hx)*v
    
    gamma211 = np.cos(alpha2)*(1-a2)*(1/dt)
    gamma221 = np.cos(alpha2)*(a2)*(1/dt)
    gamma231 = np.cos(alpha2)*(1-b2)*(1/hx)*v
    gamma241 = np.cos(alpha2)*(b2)*(1/hx)*v
    
    c111 = gamma111 + gamma131
    c121 = -gamma111 + gamma141
    c131 = gamma121 - gamma131
    c141 = -gamma121 - gamma141
    
    c211 = gamma211 + gamma231
    c221 = -gamma211 + gamma241
    c231 = gamma221 - gamma231
    c241 = -gamma221 - gamma241
    
    # Left boundary condition
    aux1 = (p *(-c111*c221-c121*c211) + pnext[x+1,z]*(-c111*c231-c131*c211) + 
            p[x+1,z]*(-c111*c241-c121*c231-c141*c211-c131*c221) + 
            pprev*(-c121*c221) + pprev[x+1,z]*(-c121*c241-c141*c221) + 
            pnext[x+2,z]*(-c131*c231) + p[x+2,z]*(-c131*c241-c141*c231) +
            pprev[x+2,z]*(-c141*c241))/(c111*c211)
    
    pde1 = (1-weightsx)*pnext + weightsx*aux1
    stencils.append(Eq(pnext, pde1, subdomain=model.grid.subdomain['d1']))
    
    # Right boundary (d2) - similar but with x-1, x-2
    aux2 = (p*(-c111*c221-c121*c211) + pnext[x-1,z]*(-c111*c231-c131*c211) + 
            p[x-1,z]*(-c111*c241-c121*c231-c141*c211-c131*c221) + 
            pprev*(-c121*c221) + pprev[x-1,z]*(-c121*c241-c141*c221) + 
            pnext[x-2,z]*(-c131*c231) + p[x-2,z]*(-c131*c241-c141*c231) +
            pprev[x-2,z]*(-c141*c241))/(c111*c211)
    
    pde2 = (1-weightsx)*pnext + weightsx*aux2
    stencils.append(Eq(pnext, pde2, subdomain=model.grid.subdomain['d2']))
    
    # Bottom boundary (d3)
    gamma113 = np.cos(alpha1)*(1-a1)*(1/dt)
    gamma123 = np.cos(alpha1)*(a1)*(1/dt)
    gamma133 = np.cos(alpha1)*(1-b1)*(1/hz)*v
    gamma143 = np.cos(alpha1)*(b1)*(1/hz)*v
    
    gamma213 = np.cos(alpha2)*(1-a2)*(1/dt)
    gamma223 = np.cos(alpha2)*(a2)*(1/dt)
    gamma233 = np.cos(alpha2)*(1-b2)*(1/hz)*v
    gamma243 = np.cos(alpha2)*(b2)*(1/hz)*v
    
    c113 = gamma113 + gamma133
    c123 = -gamma113 + gamma143
    c133 = gamma123 - gamma133
    c143 = -gamma123 - gamma143
    
    c213 = gamma213 + gamma233
    c223 = -gamma213 + gamma243
    c233 = gamma223 - gamma233
    c243 = -gamma223 - gamma243
    
    aux3 = (p*(-c113*c223-c123*c213) + pnext[x,z-1]*(-c113*c233-c133*c213) + 
            p[x,z-1]*(-c113*c243-c123*c233-c143*c213-c133*c223) + 
            pprev*(-c123*c223) + pprev[x,z-1]*(-c123*c243-c143*c223) + 
            pnext[x,z-2]*(-c133*c233) + p[x,z-2]*(-c133*c243-c143*c233) +
            pprev[x,z-2]*(-c143*c243))/(c113*c213)
    
    pde3 = (1-weightsz)*pnext + weightsz*aux3
    stencils.append(Eq(p.forward, pde3, subdomain=model.grid.subdomain['d3'])) # stencil 3
    
    return stencils

def second_order_stencil_vti_habc(model, ps, H, q, forward=True):
    m = model.m
    p = ps[0]
    
    pnext = p.forward if forward else p.backward
    pdt = p.dt if forward else p.dt.T

    # Main domain stencil (without damping)
    stencil = solve(p.dt2 - H - q, pnext)
    eq = Eq(pnext, stencil, subdomain=model.grid.subdomains['physdomain'])
    stencils = [eq]
    
    # Add free surface if needed
    if model.fs:
        stencils.append(freesurface(model, Eq(pnext, stencil)))
    
    # Add HABC stencils if enabled
    if hasattr(model, 'habc_type') and model.habc_type == 'higdon':
        stencils.extend(habc_higdon_stencils(model, p, forward=forward))
    
    return stencils

def vti_kernel_centered_habc(model, ps, **kwargs):
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
    
    p = ps[0]
    
    px = p.dx
    pz = p.dy
    px2 = px**2
    pz2 = pz**2
    px4 = px**4
    pz4 = pz**4

    dxx = p.dx2
    dzz = p.dy2
  
    numerator = -2 * (epsilon - delta) * px2 * pz2
    denominator = (1 + 2 * epsilon) * px4 + pz4 + 2 * (1 + delta) * px2 * pz2
    sn = numerator / (denominator + 1e-26)  # Small constant to avoid division by zero
    H = vp**2 * ((1+2*epsilon + sn)*dxx + (1 + sn)*dzz) 

    return second_order_stencil_vti_habc(model, ps, H, q, forward=forward)

def ForwardOperatorHABC(model, geometry, space_order=4, save=False, **kwargs):
    """
    Construct a forward modeling operator with snapshotting capability.
    """
    dt = model.grid.time_dim.spacing
    time_order = 2
    
    p = TimeFunction(name='p', grid=model.grid,
                    save=None,
                    time_order=time_order, 
                    space_order=space_order)
    
    p1 = TimeFunction(name='p1', grid=model.grid,
                    save=None,
                    time_order=time_order, 
                    space_order=space_order)
    
    p2 = TimeFunction(name='p2', grid=model.grid,
                    save=None,
                    time_order=time_order, 
                    space_order=space_order)
    
    p3 = TimeFunction(name='p3', grid=model.grid,
                    save=None,
                    time_order=time_order, 
                    space_order=space_order)
    
    src = geometry.src
    rec = geometry.rec
    
    # FD kernel
    stencils = vti_kernel_centered_habc(model, [p, p1, p2, p3])
    
    # Source and receivers

    stencils += src.inject(field=p.forward, expr=src * dt**2 / model.m)
    stencils += rec.interpolate(expr=p)

    if save:
        nsnaps = kwargs.get('nsnaps', 5)    
        factor = round(geometry.nt / nsnaps)
        time_subsampled = ConditionalDimension('t_sub', parent=model.grid.time_dim, factor=factor)
        psave = TimeFunction(name='psave', grid=model.grid,
                            time_order=time_order, space_order=space_order,
                            save=nsnaps, time_dim=time_subsampled
                            )
        stencils += [Eq(psave, p)]

    return Operator(stencils, subs=model.spacing_map, 
                   name='ForwardVTI', **kwargs)

def ForwardOperator(model, geometry, space_order=4, save=False, **kwargs):
    """
    Construct a forward modeling operator with snapshotting capability.
    """
    dt = model.grid.time_dim.spacing
    time_order = 2
    
    p = TimeFunction(name='p', grid=model.grid,
                    save=None,
                    time_order=time_order, 
                    space_order=space_order)
    
    src = geometry.src
    rec = geometry.rec
    
    # FD kernel
    stencils = vti_kernel_centered(model, p)
    
    # Source and receivers

    stencils += src.inject(field=p.forward, expr=src * dt**2 / model.m)
    stencils += rec.interpolate(expr=p)

    if save:
        nsnaps = kwargs.get('nsnaps', 5)    
        factor = round(geometry.nt / nsnaps)
        time_subsampled = ConditionalDimension('t_sub', parent=model.grid.time_dim, factor=factor)
        psave = TimeFunction(name='psave', grid=model.grid,
                            time_order=time_order, space_order=space_order,
                            save=nsnaps, time_dim=time_subsampled
                            )
        stencils += [Eq(psave, p)]

    return Operator(stencils, subs=model.spacing_map, 
                   name='ForwardVTI', **kwargs)


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
